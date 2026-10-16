# ember-4

Persistent memory for Claude instances — graph-aware retrieval over a single
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

## Claude Code hooks (optional)

`ember4/hooks/` keeps full session transcripts in the archive, so
`archive_search` / `archive_read` can reach back past a compaction or into an
earlier session. Register them in `~/.claude/settings.json` with absolute paths
(`$SRC` below is your clone):

| Event | Command | What it does |
|---|---|---|
| `PreCompact` | `$SRC/.venv/bin/python3 $SRC/ember4/hooks/pre_compact_export.py` | Archives the transcript before compaction; prints one status line |
| `SessionEnd` | `$SRC/.venv/bin/python3 $SRC/ember4/hooks/session_end_export.py` | Archives the transcript on exit (forks, so exit isn't delayed) |
| `Stop` | `bash $SRC/ember4/hooks/stop_heartbeat.sh` | Records when the turn ended |
| `UserPromptSubmit` | `bash $SRC/ember4/hooks/time_delta_marker.sh` | Adds `[time: HH:MM · +Nm since last turn]` when time has passed |

```json
{
  "hooks": {
    "PreCompact": [{"hooks": [{"type": "command", "timeout": 30,
      "command": "/abs/path/.ember4-src/.venv/bin/python3 /abs/path/.ember4-src/ember4/hooks/pre_compact_export.py"}]}]
  }
}
```

Raw JSONL copies go to `<data dir>/archive/exports/`. Transcripts land in the
same database the MCP server reads; a session run from a directory listed in
`<data dir>/persona_stores.json` (`{"<home dir>": "<data dir>"}`) is archived
to that store instead.

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
`recency_strength:<project>`; `deep_recall` and wonder reach back with strength 0.

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
    hooks/                 ← optional Claude Code hooks (see above)
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
