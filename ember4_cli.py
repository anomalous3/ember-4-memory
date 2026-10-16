"""ember-4 test harness: read frozen ember4.db, A/B current vs new logic.

Always operates on the frozen copy at ~/.ember4/ember4.db. Reads only.
The harness is intentionally separate from the MCP server so we can iterate
on clustering/scoring without touching the live ember-3 broker.

Usage:
    python ember4_cli.py count
    python ember4_cli.py recall "fiber bundles knowledge geometry"
    python ember4_cli.py dump 20
    python ember4_cli.py read <ember_id>
    python ember4_cli.py cells          # current Voronoi cell distribution
    python ember4_cli.py timespan       # oldest/newest, age histogram
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime, timezone

# Force read-only freeze before any ember4 import touches config
os.environ.setdefault("EMBER4_READ_ONLY", "1")

from ember4.config import get_db_path, READ_ONLY  # noqa: E402
from ember4.db import Database  # noqa: E402


async def _connect() -> Database:
    db = Database()
    await db.connect()
    assert db._read_only, "expected READ_ONLY freeze to be active"
    return db


async def cmd_count(_: argparse.Namespace) -> None:
    db = await _connect()
    cur = await db.conn.execute("SELECT COUNT(*) FROM embers")
    embers = (await cur.fetchone())[0]
    cur = await db.conn.execute("SELECT COUNT(*) FROM archive")
    archive = (await cur.fetchone())[0]
    cur = await db.conn.execute("SELECT COUNT(*) FROM vec_centroids")
    centroids = (await cur.fetchone())[0]
    cur = await db.conn.execute("SELECT COUNT(*) FROM recall_log")
    recalls = (await cur.fetchone())[0]
    cur = await db.conn.execute("SELECT COUNT(*) FROM edges")
    edges = (await cur.fetchone())[0]
    print(f"db_path:    {get_db_path()}")
    print(f"read_only:  {READ_ONLY}")
    print(f"embers:     {embers}")
    print(f"archive:    {archive}")
    print(f"centroids:  {centroids}")
    print(f"recall_log: {recalls}")
    print(f"edges:      {edges}")
    await db.close()


async def cmd_cells(_: argparse.Namespace) -> None:
    """Show current Voronoi cell distribution — the partition we're replacing."""
    db = await _connect()
    cur = await db.conn.execute(
        "SELECT cell_id, COUNT(*) AS n FROM embers GROUP BY cell_id ORDER BY cell_id"
    )
    rows = await cur.fetchall()
    total = sum(r["n"] for r in rows)
    print(f"current cell distribution (k-means / Voronoi, total {total}):")
    for r in rows:
        bar = "█" * (r["n"] * 40 // max(1, max(rr["n"] for rr in rows)))
        print(f"  cell {r['cell_id']:>3}  n={r['n']:>4}  {bar}")
    await db.close()


async def cmd_timespan(_: argparse.Namespace) -> None:
    """Show temporal extent of the ember corpus — fibers extend along this axis."""
    db = await _connect()
    cur = await db.conn.execute(
        "SELECT MIN(created_at) AS lo, MAX(created_at) AS hi, COUNT(*) AS n FROM embers"
    )
    row = await cur.fetchone()
    lo = datetime.fromisoformat(row["lo"].replace("Z", "+00:00"))
    hi = datetime.fromisoformat(row["hi"].replace("Z", "+00:00"))
    span_days = (hi - lo).total_seconds() / 86400.0
    print(f"oldest ember:  {lo.isoformat()}")
    print(f"newest ember:  {hi.isoformat()}")
    print(f"span:          {span_days:.1f} days")
    print(f"n_embers:      {row['n']}")

    # Crude age histogram in weeks
    cur = await db.conn.execute(
        """SELECT
             CAST((julianday('now') - julianday(created_at)) / 7 AS INTEGER) AS weeks_ago,
             COUNT(*) AS n
           FROM embers
           GROUP BY weeks_ago
           ORDER BY weeks_ago"""
    )
    rows = await cur.fetchall()
    max_n = max(r["n"] for r in rows) if rows else 1
    print("\nage histogram (weeks ago → n_embers):")
    for r in rows:
        bar = "█" * (r["n"] * 50 // max_n)
        print(f"  {r['weeks_ago']:>3}w  n={r['n']:>4}  {bar}")
    await db.close()


async def cmd_dump(args: argparse.Namespace) -> None:
    db = await _connect()
    cur = await db.conn.execute(
        "SELECT ember_id, name, importance, cell_id, created_at, access_count, "
        "shadow_load, utility_score "
        "FROM embers ORDER BY created_at DESC LIMIT ?",
        (args.n,),
    )
    rows = await cur.fetchall()
    for r in rows:
        print(
            f"{r['created_at'][:10]}  cell={r['cell_id']:>3}  "
            f"imp={r['importance']:<10}  acc={r['access_count']:>3}  "
            f"shadow={r['shadow_load']:.2f}  util={r['utility_score']:.2f}  "
            f"id={r['ember_id'][:8]}  {r['name'][:60]}"
        )
    await db.close()


async def cmd_read(args: argparse.Namespace) -> None:
    db = await _connect()
    e = await db.get_ember(args.ember_id)
    if not e:
        print(f"no ember {args.ember_id}")
        return
    print(json.dumps(e, indent=2, default=str))
    await db.close()


async def cmd_recall(args: argparse.Namespace) -> None:
    """Run current ember-3 HESTIA recall against the frozen DB.

    This is the baseline we compare against. Once new scoring is implemented,
    we'll add `recall_v4` and run both for the same query side-by-side.
    """
    # Import server-side helpers; they expect to initialize module-level _db
    # and _embedder. We monkey-load against our frozen DB.
    from ember4 import server as srv

    # Initialize the singleton against the frozen DB
    if srv._db is None:
        srv._db = Database()
        await srv._db.connect()
    if srv._embedder is None:
        from ember4.embedder import Embedder
        srv._embedder = Embedder()

    # In read-only mode, increment_access would raise. Stub it to a no-op
    # so the recall pipeline runs end-to-end without mutating the frozen DB.
    async def _noop_access(_ember_id):
        return None
    srv._db.increment_access = _noop_access  # type: ignore[assignment]

    scored = await srv._fetch_and_rerank(args.query, top_k=args.top_k)
    if not scored:
        print("(no results)")
        return

    now = datetime.now(timezone.utc)
    for i, (ember, score, breakdown) in enumerate(scored, 1):
        created = datetime.fromisoformat(ember["created_at"].replace("Z", "+00:00"))
        age_days = (now - created).total_seconds() / 86400.0
        freshness = "fresh" if age_days < 7 else f"{int(age_days)}d ago"
        cos = breakdown.get("cos_sim", float("nan"))
        sf = breakdown.get("shadow_factor", float("nan"))
        vf = breakdown.get("vitality_factor", float("nan"))
        print(
            f"#{i:>2}  score={score:.3f}  cos={cos:.3f}  shadow_f={sf:.3f}  "
            f"vital_f={vf:.3f}  cell={ember['cell_id']:>3}  {freshness}\n"
            f"     id={ember['ember_id']}\n"
            f"     {ember['name'][:90]}\n"
        )

    await srv._db.close()


async def cmd_recall_v4(args: argparse.Namespace) -> None:
    """ember-4 graph-based recall: dense + BM25 seeds → PPR diffusion → blend.

    Compares directly against `recall` (ember-3 HESTIA baseline) on the
    same frozen DB. Outputs feature decomposition per result so we can
    see why each ember surfaced.
    """
    from ember4.embedder import Embedder, NOMIC_V1_5_SPEC
    from ember4.recall_v4 import graph_recall

    db = await _connect()
    embedder = Embedder(NOMIC_V1_5_SPEC)
    # Force warmup so the first call isn't dominated by model init
    _ = embedder.embed_query("warmup")

    results, trace = await graph_recall(
        db, embedder, args.query, top_k=args.top_k, seed_k=args.seed_k,
    )

    if not results:
        print("(no results)")
        await db.close()
        return

    if trace:
        print(f"trace: seeds(dense={trace.n_seeds_dense}, "
              f"bm25={trace.n_seeds_bm25}), candidates={trace.n_candidates}, "
              f"ppr_iters={trace.ppr_iters}, "
              f"ppr_delta={trace.ppr_final_delta:.2e}")
        if args.verbose:
            edges = ", ".join(f"{t}={n}" for t, n in trace.edge_type_used.items())
            print(f"edges loaded: {edges}")
        print()

    for i, r in enumerate(results, 1):
        f = r.features
        print(
            f"#{i:>2}  score={r.score:.4f}  "
            f"ppr#{f['ppr_rank']:<4} dense#{f['dense_rank']:<4} "
            f"bm25#{f['bm25_rank']:<4}  ppr={f['ppr_score']:.2e}\n"
            f"    id={r.ember_id}\n"
            f"    {r.name[:90]}\n"
        )

    await db.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("count", help="row counts across tables")
    sub.add_parser("cells", help="current Voronoi cell distribution")
    sub.add_parser("timespan", help="temporal extent + age histogram")

    p_dump = sub.add_parser("dump", help="most recent N embers")
    p_dump.add_argument("n", nargs="?", type=int, default=20)

    p_read = sub.add_parser("read", help="full ember by id")
    p_read.add_argument("ember_id")

    p_recall = sub.add_parser("recall", help="current HESTIA recall (baseline)")
    p_recall.add_argument("query")
    p_recall.add_argument("--top-k", type=int, default=5)

    p_v4 = sub.add_parser(
        "recall-v4", help="ember-4 graph-based recall (Nomic + BM25 + PPR)"
    )
    p_v4.add_argument("query")
    p_v4.add_argument("--top-k", type=int, default=10)
    p_v4.add_argument("--seed-k", type=int, default=20)
    p_v4.add_argument("--verbose", "-v", action="store_true")

    args = parser.parse_args()
    handler = {
        "count": cmd_count,
        "cells": cmd_cells,
        "timespan": cmd_timespan,
        "dump": cmd_dump,
        "read": cmd_read,
        "recall": cmd_recall,
        "recall-v4": cmd_recall_v4,
    }[args.cmd]
    asyncio.run(handler(args))
    return 0


if __name__ == "__main__":
    sys.exit(main())
