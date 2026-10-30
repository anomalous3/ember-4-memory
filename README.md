# ember-4

Persistent memory for Claude Code and Codex — graph-aware retrieval over a single
SQLite database, using Nomic-Embed-Text-v1.5 for semantic embedding and
BM25 (via SQLite FTS5) for lexical retrieval. Designed as the successor to
ember-3 (the HESTIA/MiniLM-based memory system).

## What changed from ember-3

- **Nomic-Embed-Text-v1.5 with task-prefix discipline** — separate vector
  spaces for clustering (`clustering:`) and retrieval (`search_document:` /
  `search_query:`). Same text in different roles produces different
  vectors; this is a feature, not a footgun.
- **Graph recall** — dense (Nomic) and lexical (BM25) seeds, fused with RRF,
  teleport into personalized PageRank over the memory graph. PPR widens
  the candidate set; the final score blends dense, BM25, engagement
  (cond_activity) and a small PPR term, then applies recency demotion.
  Memories superseded by `ember_contradict` are left out unless you ask
  for historical recall (`include_stale=True`).
- **Update-activity importance signal** — replaces ember-3's
  `cell_vitality` / `shadow_load`. Has a strict contribution filter:
  `(event_type, origin)` distinguishes substantive engagement from system-
  initiated bookkeeping, preventing the basin-pathology feedback loop the
  old system silently relied on.
- **Tag-cohort edges** — pseudo-tags extracted from ember names by regex
  (project / type / version patterns) create explicit project-cohort
  adjacency that the kNN graph alone under-samples.
- **Query preparation** — proper English + corpus-specific stop-word
  filtering, OR-with-BM25-ranking instead of implicit AND. Applies to
  both ember recall and archive full-text search.

See `MIGRATION.md` for the in-place ember-3 → ember-4 cutover procedure.

## Fresh install on a new machine

```sh
# Clone
git clone https://github.com/anomalous3/ember-4-memory.git ~/.ember4-src
cd ~/.ember4-src

# Venv (uv recommended; pip works too)
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python3 -e .

# Data directory — by default ~/.ember4/, overridable via EMBER4_DATA_DIR.
# Models cached here too; ~140MB Nomic int8 ONNX downloaded on first use.
mkdir -p ~/.ember4

# Option A: start fresh (empty memory)
.venv/bin/python3 -c "
import asyncio
from ember4.db import Database
async def init():
    db = Database()
    await db.connect()
    await db.close()
asyncio.run(init())
"

# Option B: migrate from an existing ember-3 database (online backup;
# safe to run while ember-3 is still serving requests)
EMBER3_DB=$HOME/.ember/ember3.db  # adjust if your ember-3 lives elsewhere
.venv/bin/python3 -c "
import sqlite3
src = sqlite3.connect('$EMBER3_DB')
dst = sqlite3.connect('$HOME/.ember4/ember4.db')
with dst: src.backup(dst)
src.close(); dst.close()
print('ember-3 → ember-4 data copy complete')
"

# Run the post-install migrations (idempotent; --dry-run available)
.venv/bin/python3 ember4_migrate_update_log.py   # schema additions
.venv/bin/python3 ember4_migrate.py --force      # Nomic vectors
.venv/bin/python3 ember4_build_fts.py            # BM25 index
.venv/bin/python3 ember4_build_graph.py          # kNN + temporal
.venv/bin/python3 ember4_build_tag_edges.py      # project-cohort (patterns: tag_patterns.example.json)
.venv/bin/python3 ember4_rebuild_stopwords.py    # corpus stop words

# Smoke test
.venv/bin/python3 ember4_test_wireup.py
# Expected: ✓ Wire-up smoke test PASSED

# Inspect baseline state
.venv/bin/python3 ember4_cli.py count
```

## MCP server registration

Claude Code: edit `~/.claude.json` (or `~/Library/Application Support/Claude/`
on macOS) to register the ember-4 MCP server:

```json
{
  "mcpServers": {
    "ember": {
      "command": "/absolute/path/to/.ember4-src/.venv/bin/python3",
      "args": ["-m", "ember4"],
      "env": {
        "EMBER4_READ_ONLY": "0"
      }
    }
  }
}
```

`EMBER4_READ_ONLY=0` is critical — the freeze flag exists for development
and would block all writes if left set at production.

If you want **multiple ember instances** (e.g., separate memory for
different agents on the same machine), use distinct `EMBER4_DATA_DIR`
values and register them under different MCP server names:

```json
{
  "mcpServers": {
    "ember-primary": {
      "command": "...",
      "args": ["-m", "ember4"],
      "env": {"EMBER4_DATA_DIR": "/home/me/.ember4-primary"}
    },
    "ember-scratch": {
      "command": "...",
      "args": ["-m", "ember4"],
      "env": {"EMBER4_DATA_DIR": "/home/me/.ember4-scratch"}
    }
  }
}
```

Each instance gets its own DB, its own model dir, its own update_log.

## Codex MCP registration

Add this to `~/.codex/config.toml`, using your clone's absolute Python path:

```toml
[mcp_servers.ember]
command = "/absolute/path/to/.ember4-src/.venv/bin/python3"
args = ["-m", "ember4"]

[mcp_servers.ember.env]
EMBER_AGENT = "codex"
EMBER4_READ_ONLY = "0"
# EMBER4_DATA_DIR = "/absolute/path/to/your/store"  # optional
```

The same server/tools and database work with either client. MCP registration
alone does **not** install lifecycle hooks.

## Lifecycle hooks: Claude Code and Codex

The portable runner `ember4/hooks/lifecycle.py` supports both clients. It uses
that client's explicit `transcript_path`, never guesses another recent session,
keeps the raw JSONL, and publishes readable transcripts to the Ember archive.
Initialize the store using the fresh-install steps above first.

```sh
# Preview only (no changes). Run with the venv used by the Ember server.
.venv/bin/python3 ember4_install_hooks.py --client codex --timing

# Back up and merge into ~/.codex/hooks.json:
.venv/bin/python3 ember4_install_hooks.py --client codex --timing --install

# Or merge into ~/.claude/settings.json:
.venv/bin/python3 ember4_install_hooks.py --client claude --timing --install
```

`--config PATH` selects a project-local/other config file. `--data-dir PATH`
pins the hook to a specific store. **MCP server environment variables do not
configure hook subprocesses:** use that option if the MCP store is not the
normal `~/.ember4`. Existing hooks/settings are preserved; repeated installation
with identical options is a no-op. Changing the clone, Python, or data path
creates a new command: remove the old definition when relocating. Do not run
both legacy Ember exporters/timing hooks and the portable equivalents for the
same event. Existing custom wrappers can keep calling `export_session()`.

In Codex, open **`/hooks` and review/trust the new definitions**. Installation
is not trust approval; restart/resume if the running client has not loaded the
new config. No hook-trust bypass is used. Requires a Codex version supporting
`PreCompact` and `SessionEnd` command hooks (tested with CLI 0.163.0).
See [official Codex hook documentation](https://learn.chatgpt.com/docs/hooks).

| Event | Behavior |
|---|---|
| `PreCompact` | Snapshot the transcript before compaction; queue archive publication |
| `SessionEnd` | Snapshot and queue archive publication, then refresh the store's graph if needed |
| `Stop` (with `--timing`) | Save a per-client, per-session timestamp |
| `UserPromptSubmit` (with `--timing`) | Add a time-delta marker after at least one minute |

Codex limits `SessionEnd` handlers to three seconds. The hook writes a fixed-size
raw snapshot and a persistent job, then detaches a worker with independent
stdio. No embedding/model load occurs. Very large/slow transcript copies can
still exceed the client deadline; hook errors are reported rather than claiming
an export succeeded. A hard kill need not emit SessionEnd. Codex also does not
end a session merely when you switch conversations; see the official event
semantics. These hooks run where the client runs and require local command-hook
support; remote/cloud orchestration may not support local shell hooks.

Archive behavior:

- Claude messages/tool calls/results and Codex response items/tool calls/results
  are rendered. Codex event-stream mirrors are not duplicated; older event-only
  message logs have a fallback. Visible reasoning summaries are included, not
  encrypted reasoning. Raw snapshots retain the source data, including content
  omitted or shortened in the readable view. Transcript formats are not stable
  APIs; unsupported/invalid logs retain their raw copies and report errors.
- All chunks publish atomically, with UTF-8-safe size bounds. A failed part
  cannot retire the preceding complete export.
- Same-session, same-client repeats deduplicate. A newer snapshot retires an
  older one **only with byte-prefix proof** and matching extra header text.
  Rewritten/forked transcripts, truncated final records, and older legacy
  exports lacking proof are retained conservatively. No source logs are deleted.
- Raw copies live in `<data dir>/archive/exports/<client>/`; pending jobs in
  `archive/pending/`; worker diagnostics in `archive/hook.log`. Runtime files
  are local data, not repository assets. They can contain sensitive conversation
  and tool output; no redaction is promised. Retention is currently manual.
- Archive rows carry `client:<client>` and `session:<id>` tags. A small
  `transcript_exports` manifest table records coverage for safe supersession.
  These exports are FTS-searchable; the hook does not generate semantic vectors.
- Persona routing via `<data dir>/persona_stores.json` remains supported when
  no explicit data-directory override is set. The transcript's initial cwd,
  when present in the first record, identifies the session's home.

If a worker fails (e.g. store missing/locked), the snapshot and job remain. Fix
the cause and replay jobs explicitly:

```sh
.venv/bin/python3 ember4/hooks/lifecycle.py --drain
# For a nondefault store, add --data-dir /path/to/store.
```

For a manual export/test, feed a hook JSON object containing `session_id`,
`transcript_path`, `cwd` and `hook_event_name` to
`lifecycle.py --client codex --foreground` (or `--client claude`). Foreground
mode is for diagnostics, **not** a time-limited SessionEnd registration.

Run the offline regression suite (synthetic transcripts/temporary stores only):

```sh
.venv/bin/python3 -m unittest -v ember4_test_hooks
```

## Importing claude.ai and ChatGPT exports

`ember4_import_exports.py` brings your web-chat history into the archive.
Request an export (claude.ai: Settings → Privacy → Export data; ChatGPT:
Settings → Data controls → Export data), download the zips, and point the
script at them. It detects each file's format from the JSON itself, so
claude.ai's split export (`conversations-000.zip`, `memories-000.zip`, ...),
older single-zip exports and ChatGPT's `conversations.json` all work.

```sh
# Preview, then import. Exports are untrusted input: keep -I.
.venv/bin/python3 -I ember4_import_exports.py ~/Downloads/claude-export/ --dry-run
.venv/bin/python3 -I ember4_import_exports.py ~/Downloads/claude-export/ chatgpt-export.zip
```

- Each conversation becomes one readable archive transcript (split only when
  very long), dated to when it happened, so `archive_search` date filters and
  `deep_recall` find it. Chunks are tagged `imported` and `source:claude-ai`
  or `source:chatgpt`, and filed under project `imported` (`--project` to change).
- Edits and regenerations: both services keep every branch. The importer
  keeps the one you ended on (ChatGPT's `current_node`; for claude.ai, the
  path to the most recent message).
- claude.ai memories (the general summary, per-project summaries, memory
  files) and project instructions/knowledge become reference chunks.
  `--memories-as-embers` stores the memories as embers instead.
- No model calls and no embers from conversations: turning history into
  embers is a separate, judgment-heavy pass.
- Re-running is safe. Unchanged conversations are skipped; one that grew gets
  its new version and the old chunks are set to `archived`, not deleted.
- Other options: `--since YYYY-MM-DD`, `--include-thinking`, `--data-dir`.

## Configuration knobs

Env var control:

| Variable | Default | Purpose |
|---|---|---|
| `EMBER4_DATA_DIR` | `~/.ember4` | Override data directory (DB + models) |
| `EMBER4_READ_ONLY` | unset (read-write) | Set to `1` to open DB read-only; useful for diagnostic scripts that should never accidentally write |

Recency in `ember4/config.py`: `RECENCY_STRENGTH` (0.75) and `RECENCY_TAU_DAYS`
(21) demote embers whose last creation or content edit is old, so a superseded
approach doesn't rank as if current. Embers tagged `foundation` or `principle`
are exempt. A per-project override lives in the DB config table under
`recency_strength:<project>`, and a store-wide one under `recency_strength`
(useful when a store's memories are mostly old but still current); `deep_recall` and wonder reach back with strength 0.

Code-side weights in `ember4/recall_v4.py`:

- `FEATURE_WEIGHTS`: relative contribution of dense / BM25 / cond_activity / PPR
- `EDGE_TYPE_WEIGHTS`: per-type stochastic-matrix mix for the graph traversal
- `PPR_ALPHA`: restart probability (0.25)
- `RRF_K`: rank-normalization damping constant (60; standard from Cormack et al.)

See `recall_v4.py` for the empirical rationale behind each.

## Directory layout

```
~/.ember4-src/             ← this repo
  ember4/                  ← package
    __init__.py
    __main__.py            ← MCP entry point (python -m ember4)
    config.py
    db.py                  ← schema, queries, log_update
    embedder.py            ← Nomic + MiniLM-L6 specs
    query_prep.py          ← BM25 stop-word handling
    recall_v4.py           ← graph-based retrieval pipeline
    server.py              ← MCP tools
    utils.py
    hooks/                 ← optional Claude Code/Codex hooks (see above)
  ember4_*.py              ← migration / build / test scripts
  tag_patterns.example.json
  pyproject.toml
  README.md                ← this file
  MIGRATION.md             ← ember-3 → ember-4 cutover plan

~/.ember4/                 ← per-machine data (NOT in git)
  ember4.db                ← SQLite DB
  tag_patterns.json        ← your project pseudo-tag regexes (optional)
  archive/exports/         ← raw JSONL transcript copies (from the hooks)
  models/
    nomic-embed-text-v1.5/ ← downloaded on first embedder use
    all-MiniLM-L6-v2/      ← legacy, kept for back-compat
```

## Status

ember-4 is the live memory system on our machines (cut over 2026-05-25).
Live recall (`ember_recall`) runs the graph engine in `recall_v4.py`:
dense + BM25 + conditioned activity, rank-fused, with recency demotion.

- An ablation on our corpus confirmed `dense + BM25 + cond_activity`
  matches the ember-3 baseline on a diagnostic benchmark
- Cold start: an empty update_log contributes nothing and the remaining
  weights redistribute proportionally
- Write paths log to update_log with `(event_type, origin, content_changed)`
  so bookkeeping never counts as engagement
- Archive full-text search and v4 BM25 share one stop-word +
  OR-with-BM25-ranking pipeline

The smoke tests create and delete their own test embers, so they run on any
store.

## License

MIT. See [LICENSE](./LICENSE).
