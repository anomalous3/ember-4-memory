"""Stale policy and rank determinism, checked on a copy of the live store.

    .venv/bin/python3 ember4_test_stale_policy.py

1. Rebuilds the graph on the copy: no knn/temporal/tag_cohort edge touches a
   stale ember, and the session-end trigger is satisfied afterwards.
2. Recalls by the names of stale embers: active recall returns none of them;
   include_stale=True brings them back.
3. Runs the same recalls under two hash seeds: identical rankings, scores
   equal up to embedding float jitter.

Read-only on the live store (sqlite backup API). Needs a store with stale
embers; skips otherwise.
"""
import argparse
import asyncio
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
GRAPH_TYPES = ("knn_nomic_v15", "temporal_nomic", "tag_cohort")


async def recall_dump(names, include_stale):
    from ember4.db import Database
    from ember4.embedder import Embedder
    from ember4.recall_v4 import graph_recall
    db = Database()
    await db.connect()
    emb = Embedder()
    out = []
    for q in names:
        res, _ = await graph_recall(db, emb, q, top_k=10, include_stale=include_stale)
        out.append([(r.ember_id, round(r.score, 12)) for r in res])
    await db.close()
    return out


def run_child(data_dir, names, include_stale, hashseed):
    env = dict(os.environ, EMBER4_DATA_DIR=str(data_dir), PYTHONHASHSEED=str(hashseed),
               PYTHONUNBUFFERED="1")
    args = [sys.executable, __file__, "--child", json.dumps(names)]
    if include_stale:
        args.append("--include-stale")
    p = subprocess.run(args, env=env, capture_output=True, text=True, cwd=HERE)
    if p.returncode:
        sys.exit(f"child failed:\n{p.stderr[-2000:]}")
    return json.loads(p.stdout.strip().splitlines()[-1])


def main():
    sys.path.insert(0, str(HERE))
    from ember4.config import get_db_path

    tmp = Path(tempfile.mkdtemp(prefix="ember4_stale_test_"))
    try:
        src = sqlite3.connect(f"file:{get_db_path()}?mode=ro", uri=True)
        dst = sqlite3.connect(tmp / "ember4.db")
        src.backup(dst)
        src.close()
        stale = [r for r in dst.execute(
            "SELECT ember_id, name FROM embers WHERE is_stale = 1 ORDER BY ember_id")]
        dst.close()
        if not stale:
            print("skip: no stale embers in this store")
            return 0
        stale_ids = {eid for eid, _ in stale}
        fails = []

        # 1. graph rebuild on the copy
        env = dict(os.environ, EMBER4_DATA_DIR=str(tmp), PYTHONUNBUFFERED="1")
        for script, extra in (("ember4_build_graph.py", []), ("ember4_build_tag_edges.py", [])):
            p = subprocess.run([sys.executable, script, *extra], env=env, cwd=HERE,
                               capture_output=True, text=True)
            if p.returncode:
                sys.exit(f"{script} failed:\n{p.stderr[-2000:]}")
        db = sqlite3.connect(tmp / "ember4.db")
        ph = ",".join("?" * len(GRAPH_TYPES))
        n_touch = db.execute(
            f"SELECT COUNT(*) FROM edges WHERE edge_type IN ({ph}) AND "
            "(source_id IN (SELECT ember_id FROM embers WHERE is_stale = 1) OR "
            " target_id IN (SELECT ember_id FROM embers WHERE is_stale = 1))",
            GRAPH_TYPES).fetchone()[0]
        if n_touch:
            fails.append(f"{n_touch} rebuilt graph edges touch stale embers")
        db.close()
        os.environ["EMBER4_DATA_DIR"] = str(tmp)
        sys.path.insert(0, str(HERE / "ember4" / "hooks"))
        from session_end_export import graph_is_stale
        if graph_is_stale(tmp / "ember4.db"):
            fails.append("session-end trigger still fires right after a full rebuild")

        # 2 + 3. recall by stale names, active vs historical, two hash seeds
        names = [n for _, n in stale[:12]]
        active_a = run_child(tmp, names, False, 1)
        active_b = run_child(tmp, names, False, 2)
        hist = run_child(tmp, names, True, 1)
        leaked = sum(1 for res in active_a for eid, _ in res if eid in stale_ids)
        if leaked:
            fails.append(f"active recall returned {leaked} stale embers")
        back = sum(1 for res in hist for eid, _ in res if eid in stale_ids)
        if back == 0:
            fails.append("include_stale=True returned no stale embers for their own names")
        # Embedding floats jitter ~1e-8 between runs (threaded BLAS), so
        # compare the ranking exactly and scores loosely.
        if [[e for e, _ in r] for r in active_a] != [[e for e, _ in r] for r in active_b]:
            fails.append("rankings differ between hash seeds")
        elif any(abs(s1 - s2) > 1e-6 for a, b in zip(active_a, active_b)
                 for (_, s1), (_, s2) in zip(a, b)):
            fails.append("scores differ between hash seeds beyond float jitter")

        print(f"stale embers: {len(stale)}; queries: {len(names)}; "
              f"historical hits on stale: {back}; active leaks: {leaked}")
        if fails:
            print("FAIL\n  " + "\n  ".join(fails))
            return 1
        print("ok: stale embers out of graph and active recall, back with include_stale; "
              "ranks identical across hash seeds")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--child")
    ap.add_argument("--include-stale", action="store_true")
    a = ap.parse_args()
    if a.child:
        sys.path.insert(0, str(HERE))
        print(json.dumps(asyncio.run(recall_dump(json.loads(a.child), a.include_stale))))
        sys.exit(0)
    sys.exit(main())
