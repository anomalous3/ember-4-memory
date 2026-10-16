"""ember-4 FTS5 backfill: populate embers_fts from existing embers rows.

The schema (db.py SCHEMA_SQL) creates the FTS5 virtual table and the
ai/ad/au triggers idempotently. The triggers fire on future INSERT/UPDATE/
DELETE but cannot retroactively index existing rows. This script issues the
canonical FTS5 'rebuild' command so the index reflects current embers
content.

Idempotent — re-runs rebuild from scratch each time. Use after any bulk
edit to embers content if you want the FTS index re-aligned.

Usage:
    .venv/bin/python3 ember4_build_fts.py
    .venv/bin/python3 ember4_build_fts.py --verify-query "fiber bundle manifold"
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time

# FTS5 rebuild is a write operation.
os.environ["EMBER4_READ_ONLY"] = "0"

from ember4.db import Database  # noqa: E402


async def main_async(args: argparse.Namespace) -> int:
    db = Database()
    await db.connect()
    assert not db._read_only, "freeze must be off for FTS rebuild"

    print(f"db_path: {db.db_path}")

    # ─── Confirm tables exist (the schema CREATE IF NOT EXISTS runs at
    #     connect() time when not read-only; so by now embers_fts exists) ──
    cur = await db.conn.execute(
        "SELECT name FROM sqlite_master "
        "WHERE type IN ('table','view') AND name IN ('embers_fts','archive_fts')"
    )
    have = {r["name"] for r in await cur.fetchall()}
    if "embers_fts" not in have:
        print("ERROR: embers_fts not in schema — did SCHEMA_SQL run?")
        await db.close()
        return 1
    print(f"FTS tables present: {sorted(have)}")

    # Pre-state
    cur = await db.conn.execute("SELECT COUNT(*) FROM embers_fts")
    n_before = (await cur.fetchone())[0]
    cur = await db.conn.execute("SELECT COUNT(*) FROM embers")
    n_embers = (await cur.fetchone())[0]
    print(f"embers:       {n_embers}")
    print(f"embers_fts:   {n_before}  (pre-rebuild)")

    # ─── Rebuild ─────────────────────────────────────────────────────────
    print("\nRebuilding embers_fts...")
    t0 = time.time()
    await db.conn.execute(
        "INSERT INTO embers_fts(embers_fts) VALUES('rebuild')"
    )
    await db.conn.commit()
    print(f"  rebuild done in {time.time()-t0:.2f}s")

    cur = await db.conn.execute("SELECT COUNT(*) FROM embers_fts")
    n_after = (await cur.fetchone())[0]
    print(f"embers_fts:   {n_after}  (post-rebuild)")

    if n_after != n_embers:
        print(f"  WARN: index count {n_after} != embers count {n_embers}")
    else:
        print("  OK: index aligned with embers count")

    # ─── Verify with a sample BM25 query ─────────────────────────────────
    if args.verify_query:
        print(f"\n─── BM25 sample query: {args.verify_query!r} ─────────")
        # bm25(table) returns negative scores; smaller (more negative) = better
        cur = await db.conn.execute(
            "SELECT ember_id, name, bm25(embers_fts) AS score "
            "FROM embers_fts WHERE embers_fts MATCH ? "
            "ORDER BY bm25(embers_fts) LIMIT 10",
            (args.verify_query,),
        )
        results = await cur.fetchall()
        if not results:
            print("  (no FTS5 matches)")
        for i, r in enumerate(results, 1):
            print(f"  #{i:>2}  bm25={r['score']:.3f}  {r['name'][:65]}")

    await db.close()
    print("\nDone. Set EMBER4_READ_ONLY=1 for normal use.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--verify-query", type=str, default="",
        help="optional BM25 query to run after rebuild for sanity check",
    )
    args = parser.parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    sys.exit(main())
