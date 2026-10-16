# ember-4 Migration / Live Cutover Plan

Last updated: 2026-05-15

This document describes how to promote ember-4 from a development copy
running against frozen data into the live MCP memory server on a machine
that already runs ember-3. Once cutover happens, ember-3 stops accumulating
new data and ember-4 becomes the system of record. For a **fresh install**
with no ember-3 history, see [README.md](./README.md) instead.

> **Retrieval status:** the live MCP tools (`ember_recall`, `ember_auto`,
> `deep_recall`, `ember_graph_search`) rank with the ember-4 graph pipeline
> in `recall_v4.py` (via `server.py:_graph_rerank`). The ember-3 scoring
> path `_fetch_and_rerank` remains in the source but has no callers.

Commands below use these variables; set them for your machine:

```sh
EMBER3_SRC=~/ember3            # ember-3 source checkout (has its own .venv)
EMBER3_DB=~/.ember/ember3.db   # live ember-3 database
EMBER4_SRC=~/ember-4           # this repo
EMBER4_DB=~/.ember4/ember4.db  # ember-4 database (EMBER4_DATA_DIR overrides ~/.ember4)
```

## Pre-cutover state assumptions

- The ember-3 MCP server is live (stdio processes spawned by Claude Code),
  reading and writing `$EMBER3_DB`.
- An ember-4 checkout exists at `$EMBER4_SRC`, possibly with a stale
  snapshot at `$EMBER4_DB`.
- ember-3 keeps working at all times; the cutover is reversible.

## Pre-cutover checklist

```sh
# 1. ember-4 update_log must be empty (cold start preserved; no synthetic
#    or test data lingering).
sqlite3 $EMBER4_DB "SELECT COUNT(*) FROM update_log;"
# Expected: 0

# 2. No 'synthetic-' entries in update_log
sqlite3 $EMBER4_DB \
    "SELECT COUNT(*) FROM update_log WHERE details LIKE 'synthetic-%' OR details LIKE 'test-%';"
# Expected: 0

# 3. Wire-up smoke test passes
cd $EMBER4_SRC
.venv/bin/python3 ember4_test_wireup.py
# Expected: "✓ Wire-up smoke test PASSED"
```

If any of these fail, fix before continuing. **Do not migrate live data
into a database with synthetic or test contamination.**

## Cutover procedure

### Step 1 — Refresh ember-4 data from live ember-3

`sqlite3 .backup` is online-safe; ember-3 can keep running during the
backup. The resulting DB is a snapshot, not a live mirror.

```sh
# Stop ember-3 MCP would be cleaner but unnecessary for backup safety.
# (Optional) snapshot the current ember-4 DB before overwriting, in case
# we need to roll back to the earlier snapshot.
cp $EMBER4_DB $EMBER4_DB.pre-cutover

# Online backup from live source → new ember-4 data file
$EMBER3_SRC/.venv/bin/python3 -c "
import sqlite3
src = sqlite3.connect('$EMBER3_DB')
dst = sqlite3.connect('$EMBER4_DB.new')
with dst:
    src.backup(dst)
src.close(); dst.close()
print('backup complete')
"

# Verify counts before swap
sqlite3 $EMBER4_DB.new "SELECT COUNT(*) FROM embers"
sqlite3 $EMBER3_DB   "SELECT COUNT(*) FROM embers"
# These should match.

# Swap in
mv $EMBER4_DB $EMBER4_DB.previous
mv $EMBER4_DB.new $EMBER4_DB
```

### Step 2 — Apply ember-4 schema and populate parallel structures

Run the migration scripts in order. Each is idempotent and has `--dry-run`.

```sh
cd $EMBER4_SRC

# 2a. Schema additions (update_log shape, new vec tables, embers_fts,
#     fiber_assignments). Idempotent.
.venv/bin/python3 ember4_migrate_update_log.py
#     For a fresh data refresh from ember-3, this will say "already migrated"
#     if the schema slots already exist, OR drop+recreate update_log if the
#     refresh brought back the old shape.

# 2b. Generate Nomic clustering + retrieval vectors for all embers,
#     plus retrieval vectors for archive summaries. ~150 seconds.
.venv/bin/python3 ember4_migrate.py --force
#     --force is required because the previous migration left tables
#     populated; we want to regenerate against current data.

# 2c. Populate FTS5 index for BM25 retrieval over embers content.
.venv/bin/python3 ember4_build_fts.py

# 2d. Build kNN + temporal proximity graph over Nomic clustering vectors.
.venv/bin/python3 ember4_build_graph.py

# 2e. Build tag_cohort edges from name-pattern IDF weighting.
.venv/bin/python3 ember4_build_tag_edges.py
```

### Step 3 — Switch MCP config

Claude Code reads its MCP server config from `~/.claude.json`. The
ember entry currently points at the ember-3 venv:

```json
{
  "ember": {
    "command": "$EMBER3_SRC/.venv/bin/python3",
    "args": ["-m", "ember"]
  }
}
```

After cutover it should point at ember-4:

```json
{
  "ember": {
    "command": "/path/to/ember-4/.venv/bin/python3",
    "args": ["-m", "ember4"],
    "env": {
      "EMBER4_READ_ONLY": "0"
    }
  }
}
```

`EMBER4_READ_ONLY=0` is critical — the freeze flag exists for development
and would block all writes if accidentally left on at cutover.

After editing `~/.claude.json`, restart Claude Code (or `/mcp` reload)
for the change to take effect.

### Step 4 — Other systems referencing ember

Anything else that opens `$EMBER3_DB` directly (hooks, scripts, other
agents) keeps seeing ember-3 data until it is pointed at ember-4. The hooks
in `ember4/hooks/` already target ember-4. Each additional machine needs its
own clone, model download, and the migration scripts run locally.

### Step 5 — Post-cutover verification

```sh
# (Optional) Mark the cutover boundary explicitly in update_log so we
# can audit "first ember-4 event" later.
sqlite3 $EMBER4_DB <<'SQL'
INSERT INTO update_log (ember_id, event_type, origin, content_changed, details)
SELECT ember_id, 'auto_enrich', 'migration', 0,
       'EMBER-4 CUTOVER: live MCP server promoted from ember-3'
FROM embers LIMIT 1;
SQL
#   (Note: this single row uses origin='migration' so it does NOT
#    contribute to the importance signal. It's an audit landmark only.)

# Smoke test: make a fresh ember through the live MCP, then verify it
# landed correctly. From Claude Code:
#   ember_store(name="cutover smoke", content="...", importance="context")
# Then inspect:
sqlite3 $EMBER4_DB <<'SQL'
SELECT ember_id, name, source FROM embers
ORDER BY created_at DESC LIMIT 3;
SQL

# Confirm the new ember has Nomic vectors
sqlite3 $EMBER4_DB <<'SQL'
SELECT
  (SELECT COUNT(*) FROM vec_embers_nomic_v15_clustering) AS n_cluster,
  (SELECT COUNT(*) FROM vec_embers_nomic_v15_retrieval) AS n_retrieval,
  (SELECT COUNT(*) FROM embers) AS n_embers;
SQL
# n_cluster and n_retrieval should equal n_embers.

# Confirm log_update is accumulating from real writes
sqlite3 $EMBER4_DB \
  "SELECT event_type, origin, COUNT(*) FROM update_log GROUP BY event_type, origin;"
# Should show edit/agent rows from the new ember_store calls.
```

## Rollback procedure

If anything goes wrong:

```sh
# 1. Revert ~/.claude.json to point at ember-3.
# 2. Restart Claude Code.
# 3. Live ember-3 takes over again; no data lost (its DB was never modified).
# 4. (Optional) Inspect ember-4 to understand what failed; restore the
#    pre-cutover snapshot if needed:
mv $EMBER4_DB $EMBER4_DB.failed-cutover
mv $EMBER4_DB.previous $EMBER4_DB
```

## Things that explicitly do NOT survive into ember-4

These are intentional ablations. Do not
re-enable them as "kludge fixes" if recall feels off post-cutover.

1. **cell_id / Voronoi cells**: column remains for ember-3 compatibility
   but ember-4 code never reads it. New embers store cell_id = -1 and
   the fiber assignment lives in the (deferred) `fiber_assignments` table.

2. **shadow_load auto-archive**: `_shadow_on_insert` is now a shim that
   routes to `_post_insert_v4` + `_relate_on_insert`. No shadow_load
   writes, no auto-archive. Existing shadow_load values in the table are
   frozen historical data.

3. **region_stats / cell_vitality**: not read by recall_v4. The principled
   replacement is `compute_conditioned_activity` (task 6, 2026-05-13)
   which conditions engagement on query-cohort overlap to avoid the
   basin pathology global popularity would create.

If post-cutover recall behavior is meaningfully worse than ember-3
HESTIA, the fix is **NOT** to restore these — it's to tune the v4
feature weights and/or accumulate more update_log signal. The
contribution filter is load-bearing for preventing the feedback loops
the old system silently relied on.

## After cutover: ember-5 watchlist

Things expected to break or feel cracked over time. When they do,
that's the trigger for ember-5 design work.

- **Tag pseudo-extraction is hand-coded.** When new project names show
  up, the patterns in `tag_patterns.json` will silently miss them until
  someone adds them.
  Either move to LLM-driven tagging (Haiku batch over the corpus) or
  add a self-registering pattern table.

- **PPR is a sidecar, not primary retrieval.** The two-mode reframe is
  in place but the "wonder mode" pathway isn't built. When users want
  to explore a neighborhood instead of finding an answer, they currently
  have no API for that.

- **Cold-start has paid off; long-tail bias will appear.** As update_log
  accumulates, embers that get touched a lot will dominate recall for
  their cohorts. The hub-penalty in the PPR layer is the only defense;
  if conditioned_activity starts looking like global activity in
  practice, the conditioning is failing.

- **Embedder lock-in.** Nomic v1.5 is great in 2026-05; not great
  forever. Schema supports multi-embedder (vec_embers_nomic_v15_* tables
  could sit beside future vec_embers_qwen_v5_*) but no migration path
  is built. Plan ember-5 with the embedder-swap story.
