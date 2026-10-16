"""ember-4 schema migration: extend update_log with origin + content_changed.

The importance-contribution filter for engagement events needs two more columns
than the original schema had. This script:

  1. Verifies update_log is empty (refuses destructive migration if rows exist)
  2. Drops the old table
  3. Recreates it with the new shape (origin + content_changed + 'auto_enrich')

The new SCHEMA_SQL in db.py has the target shape — but CREATE IF NOT EXISTS
is a no-op when the table already exists, so a connect alone wouldn't fix
existing DBs. Hence this one-shot migration.

After this runs successfully, connecting via Database() will see the new
shape (because IF NOT EXISTS now matches the just-created table).

Idempotent: safe to re-run; detects "already migrated" via column presence
and exits no-op.

Usage:
    .venv/bin/python3 ember4_migrate_update_log.py
    .venv/bin/python3 ember4_migrate_update_log.py --dry-run
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys

# Schema migration writes; turn off freeze.
os.environ["EMBER4_READ_ONLY"] = "0"

import aiosqlite  # noqa: E402

# We intentionally do NOT import ember4.db.Database here — its connect()
# would run the new SCHEMA_SQL which references columns that don't exist
# on the old update_log table yet (the indexes on origin/ember_time fail
# before our migration logic runs). We open SQLite raw, do the swap, and
# subsequent Database() connects will then succeed because the table
# matches the new schema.
from ember4.config import get_db_path  # noqa: E402


NEW_UPDATE_LOG_SQL = """
CREATE TABLE update_log (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    ember_id        TEXT NOT NULL,
    event_type      TEXT NOT NULL CHECK (event_type IN
        ('edit', 'contradict_old', 'contradict_new', 'edge_form',
         'tag_update', 'auto_enrich')),
    origin          TEXT NOT NULL DEFAULT 'agent' CHECK (origin IN
        ('user', 'agent', 'auto_system', 'migration')),
    content_changed INTEGER NOT NULL DEFAULT 0,
    session_id      TEXT NOT NULL DEFAULT '',
    details         TEXT NOT NULL DEFAULT '',
    timestamp       TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
    FOREIGN KEY (ember_id) REFERENCES embers(ember_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_update_log_ember
    ON update_log(ember_id);
CREATE INDEX IF NOT EXISTS idx_update_log_event
    ON update_log(event_type);
CREATE INDEX IF NOT EXISTS idx_update_log_origin
    ON update_log(origin);
CREATE INDEX IF NOT EXISTS idx_update_log_time
    ON update_log(timestamp);
CREATE INDEX IF NOT EXISTS idx_update_log_ember_time
    ON update_log(ember_id, timestamp);
"""


async def main_async(args: argparse.Namespace) -> int:
    db_path = str(get_db_path())
    print(f"db_path: {db_path}")

    # Open raw — do NOT use Database() which runs SCHEMA_SQL, which
    # references columns that don't exist on the old table yet.
    conn = await aiosqlite.connect(db_path)
    conn.row_factory = aiosqlite.Row

    # ─── Detect current schema ───────────────────────────────────────────
    cur = await conn.execute("PRAGMA table_info(update_log)")
    cols = [r[1] for r in await cur.fetchall()]
    print(f"current update_log columns: {cols}")

    have_new = "origin" in cols and "content_changed" in cols
    if have_new:
        print("Already migrated. No-op.")
        await conn.close()
        return 0

    # ─── Safety check: must be empty ─────────────────────────────────────
    cur = await conn.execute("SELECT COUNT(*) FROM update_log")
    n = (await cur.fetchone())[0]
    print(f"update_log row count: {n}")
    if n > 0:
        print(
            "REFUSING destructive migration: update_log has rows. "
            "Either back up + clear manually, or write a non-destructive ALTER."
        )
        await conn.close()
        return 1

    if args.dry_run:
        print("\n(dry-run; would DROP update_log and recreate with new shape)")
        print("New shape:")
        print(NEW_UPDATE_LOG_SQL)
        await conn.close()
        return 0

    # ─── Migrate ─────────────────────────────────────────────────────────
    print("\nDropping old update_log...")
    await conn.execute("DROP TABLE update_log")
    await conn.commit()

    print("Creating new update_log...")
    await conn.executescript(NEW_UPDATE_LOG_SQL)
    await conn.commit()

    # Verify
    cur = await conn.execute("PRAGMA table_info(update_log)")
    new_cols = [r[1] for r in await cur.fetchall()]
    print(f"new update_log columns: {new_cols}")

    # Smoke test: insert + read a sample row of each contribution case
    print("\nSmoke test — inserting sample rows for each (event_type, origin):")
    samples = [
        ("edit",           "user",        1, "user edit; counts"),
        ("edit",           "agent",       1, "agent edit; counts"),
        ("contradict_old", "agent",       0, "old-side of contradict; counts"),
        ("edge_form",      "agent",       0, "edge formation; counts"),
        ("tag_update",     "user",        0, "human curating tags; small contribution"),
        ("tag_update",     "auto_system", 0, "tag-on-recall; does NOT contribute"),
        ("auto_enrich",    "migration",   0, "Haiku batch enrich; does NOT contribute"),
    ]
    cur = await conn.execute("SELECT ember_id FROM embers LIMIT 1")
    real_id = (await cur.fetchone())["ember_id"]
    for ev, origin, cc, detail in samples:
        await conn.execute(
            "INSERT INTO update_log (ember_id, event_type, origin, "
            "content_changed, details) VALUES (?, ?, ?, ?, ?)",
            (real_id, ev, origin, cc, detail),
        )
    await conn.commit()

    cur = await conn.execute("""
        SELECT
            event_type, origin, content_changed,
            CASE WHEN
                (event_type IN ('edit','contradict_old','contradict_new','edge_form')
                 AND origin IN ('user','agent'))
                OR
                (event_type = 'tag_update' AND origin IN ('user','agent')
                 AND content_changed = 0)
            THEN 1 ELSE 0 END AS contributes,
            details
        FROM update_log
        WHERE ember_id = ?
        ORDER BY id
    """, (real_id,))
    rows = await cur.fetchall()
    print(f"\nContribution filter check ({len(rows)} rows):")
    print(f"  {'event_type':<16} {'origin':<12} {'cc':>3} {'contributes':>11}  details")
    for r in rows:
        print(
            f"  {r['event_type']:<16} {r['origin']:<12} {r['content_changed']:>3} "
            f"{r['contributes']:>11}  {r['details']}"
        )

    print(f"\nCleaning up {len(rows)} test rows...")
    await conn.execute(
        "DELETE FROM update_log WHERE ember_id = ?", (real_id,)
    )
    await conn.commit()

    cur = await conn.execute("SELECT COUNT(*) FROM update_log")
    final_n = (await cur.fetchone())[0]
    print(f"final update_log row count: {final_n} (expect 0 for cold start)")

    await conn.close()
    print("\nMigration complete. Cold-start active; signal accumulates from here.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    sys.exit(main())
