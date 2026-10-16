"""Non-mutating prerequisite checks for dedup/PPR calibration.

Run with .venv/bin/python3 ember4_calibration_audit.py [--db PATH].
Without --db, only tiny in-memory probes run; no model is loaded. With --db,
SQLite mode=ro + query_only and one read transaction protect the live corpus.
Output is diagnostic JSON, not a relevance benchmark or a threshold fit.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import aiosqlite
import numpy as np
import scipy.sparse as sp
import sqlite_vec

from ember4.embedder import serialize_vector
from ember4.recall_v4 import (
    build_transition_matrix, load_graph_state, personalized_pagerank,
)
from ember4.utils import l2_to_cosine


def distance_probe() -> dict:
    vectors = np.array([[1., 0.], [0., 1.], [-1., 0.]], dtype=np.float32)
    conn = sqlite3.connect(":memory:")
    try:
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.execute("CREATE VIRTUAL TABLE v USING vec0(id INTEGER PRIMARY KEY, embedding float[2])")
        conn.executemany("INSERT INTO v VALUES (?, ?)", [
            (i, serialize_vector(v)) for i, v in enumerate(vectors)
        ])
        rows = conn.execute(
            "SELECT id, distance FROM v WHERE embedding MATCH ? AND k=3 ORDER BY distance",
            (serialize_vector(vectors[0]),),
        ).fetchall()
    finally:
        conn.close()
    distances = np.array([d for _, d in rows])
    dots = np.array([float(vectors[0] @ vectors[i]) for i, _ in rows])
    return {
        "sqlite_vec_version": sqlite_vec.__version__,
        "distance": distances.tolist(),
        "true_cosine": dots.tolist(),
        "current_helper": [l2_to_cosine(float(d)) for d in distances],
        "matches_l2": bool(np.allclose(distances, np.linalg.norm(vectors - vectors[0], axis=1))),
        "squared_conversion_matches_dot": bool(np.allclose(1 - distances**2 / 2, dots, atol=1e-6)),
        "learn_distance_cutoffs_as_true_cosine": {str(d): 1-d*d/2 for d in (0.1, 0.3)},
    }


def transition_probe() -> dict:
    # Only one edge type is available, so its mixture weight leaves missing mass.
    matrix = sp.csr_matrix([[0., 1.], [1., 0.]])
    transition = build_transition_matrix({"knn_nomic_v15": matrix})
    p0 = np.array([1., 0.])
    score, iterations, delta = personalized_pagerank(transition, p0)
    empty_score, _, _ = personalized_pagerank(sp.csr_matrix((2, 2)), p0)
    return {
        "row_sums": np.asarray(transition.sum(axis=1)).ravel().tolist(),
        "ppr_sum": float(score.sum()), "ppr": score.tolist(),
        "iterations": iterations, "delta": delta,
        "empty_graph_ppr": empty_score.tolist(),
        "empty_graph_ppr_sum": float(empty_score.sum()),
    }


async def database_audit(path: Path) -> dict:
    async with aiosqlite.connect(path.resolve().as_uri() + "?mode=ro", uri=True) as conn:
        conn.row_factory = aiosqlite.Row
        await conn.enable_load_extension(True)
        await conn.load_extension(sqlite_vec.loadable_path())
        await conn.execute("PRAGMA query_only=ON")
        await conn.execute("BEGIN")

        async def rows(sql):
            async with conn.execute(sql) as cursor:
                return [dict(row) for row in await cursor.fetchall()]

        result = {"db": str(path.resolve())}
        result["snapshot_time"] = await rows("SELECT datetime('now') AS utc")
        result["embers"] = await rows("SELECT COUNT(*) AS total, SUM(is_stale) AS stale FROM embers")
        result["missing_knn"] = await rows("""SELECT COUNT(*) AS total FROM embers e
            WHERE NOT EXISTS (SELECT 1 FROM edges g WHERE g.source_id=e.ember_id
                AND g.edge_type='knn_nomic_v15')""")
        result["events"] = await rows("""SELECT origin,event_type,COUNT(*) AS total,
            MAX(timestamp) AS newest FROM update_log GROUP BY origin,event_type""")
        result["recent_qualifying_events"] = await rows("""SELECT COUNT(*) AS total,
            COUNT(DISTINCT ember_id) AS embers FROM update_log
            WHERE origin IN ('agent','user') AND (
                event_type IN ('edit','contradict_old','contradict_new','edge_form')
                OR (event_type='tag_update' AND content_changed=0))
            AND julianday(timestamp)>julianday('now')-42""")
        result["vectors"] = {}
        for table in ("vec_embers", "vec_embers_nomic_v15_clustering", "vec_embers_nomic_v15_retrieval"):
            result["vectors"][table] = await rows(f"""SELECT COUNT(*) AS total,
                SUM(e.ember_id IS NULL) AS orphaned, SUM(e.is_stale=1) AS stale
                FROM {table} v LEFT JOIN embers e ON e.ember_id=v.ember_id""")
        ids, _, matrices = await load_graph_state(SimpleNamespace(conn=conn))
        transition = build_transition_matrix(matrices)
        row_sums = np.asarray(transition.sum(axis=1)).ravel()
        result["transition"] = {
            "nodes": len(ids),
            "row_sum_quantiles": np.quantile(row_sums, [0, .25, .5, .75, 1]).tolist() if len(ids) else [],
            "rows_below_one": int(np.sum(row_sums < 1-1e-9)),
            "empty_rows": int(np.sum(row_sums == 0)),
        }
        await conn.rollback()
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path)
    args = parser.parse_args()
    result = {"distance_probe": distance_probe(), "transition_probe": transition_probe()}
    if args.db:
        result["database_audit"] = asyncio.run(database_audit(args.db))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
