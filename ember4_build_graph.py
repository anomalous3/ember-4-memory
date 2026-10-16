"""ember-4 graph builder: kNN (semantic) + temporal-proximity edges.

Framing: the graph is the primary indexing primitive for
ember-4 — not a step toward clustering. Recall traverses this graph; fibers
(if useful) are derived from it via community detection, not built into it.

Two edge types are produced, both stored in the existing `edges` table so
the schema doesn't need to change:

- `knn_nomic_v15`: for each ember, edges to its K nearest semantic
  neighbors in Nomic clustering-vector space. Weight = cosine similarity.
- `temporal_nomic`: for each ember, edges to its N nearest temporal
  neighbors (smallest |Δt|). Weight = exp(-|Δt| / τ_days).

Cosine similarity is bounded [-1, 1], and exp(-x/τ) is bounded (0, 1], so
both edge weights are on comparable scales. The blend weight between them
becomes a recall-time hyperparameter, not a build-time bake-in.

Pre-existing edges (`shadow`, `kg`) are left alone. Re-runs of this script
delete only its own edge types before re-inserting, so it's idempotent.

Usage:
    .venv/bin/python3 ember4_build_graph.py
    .venv/bin/python3 ember4_build_graph.py --k-knn 20 --n-temporal 5 --tau-days 7
    .venv/bin/python3 ember4_build_graph.py --dry-run
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Graph build is additive writes; turn off freeze before importing config.
os.environ["EMBER4_READ_ONLY"] = "0"

import numpy as np  # noqa: E402

from ember4.db import Database  # noqa: E402


KNN_EDGE_TYPE = "knn_nomic_v15"
TEMPORAL_EDGE_TYPE = "temporal_nomic"


def _parse_iso(ts: str) -> datetime:
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))


async def load_clustering_vectors(db: Database) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Load clustering vectors + created_at for all migrated embers.

    Returns (ids, vectors (N, 768), age_days (N,) where age_days[i] is days
    since some reference timestamp — used only for relative deltas).
    """
    cur = await db.conn.execute(
        "SELECT v.ember_id, v.embedding, e.created_at "
        "FROM vec_embers_nomic_v15_clustering v "
        "JOIN embers e ON e.ember_id = v.ember_id "
        "WHERE e.is_stale = 0 "
        "ORDER BY e.created_at"
    )
    rows = await cur.fetchall()
    if not rows:
        return [], np.zeros((0, 768), dtype=np.float32), np.zeros(0)

    ids = [r["ember_id"] for r in rows]
    vecs = np.stack(
        [np.frombuffer(r["embedding"], dtype=np.float32) for r in rows]
    )
    # Sanity check normalization
    norms = np.linalg.norm(vecs, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-3):
        bad = (np.abs(norms - 1.0) > 1e-3).sum()
        print(f"  WARN: {bad}/{len(vecs)} vectors not unit-norm "
              f"(min={norms.min():.4f} max={norms.max():.4f})")

    # Build age vector in days relative to oldest ember
    times = np.array([_parse_iso(r["created_at"]).timestamp() for r in rows])
    age_days = (times - times.min()) / 86400.0
    return ids, vecs, age_days


def compute_knn_edges(
    vecs: np.ndarray, k: int
) -> tuple[np.ndarray, np.ndarray]:
    """Brute-force kNN over L2-normalized clustering vectors.

    Returns (indices (N, k), similarities (N, k)) where indices[i] are the
    k nearest neighbors of i (excluding i itself) and similarities[i] are
    their cosine similarities.
    """
    # cos_sim = A @ A.T for L2-normalized A
    sims = vecs @ vecs.T  # (N, N) float32
    np.fill_diagonal(sims, -np.inf)  # exclude self
    # argpartition is faster than argsort for top-k
    nn_idx = np.argpartition(-sims, k, axis=1)[:, :k]
    # Sort the top-k by descending similarity per-row
    rows = np.arange(vecs.shape[0])[:, None]
    sim_topk = sims[rows, nn_idx]
    order = np.argsort(-sim_topk, axis=1)
    nn_idx = nn_idx[rows, order]
    sim_topk = sim_topk[rows, order]
    return nn_idx, sim_topk


def compute_temporal_edges(
    age_days: np.ndarray, n: int, tau_days: float
) -> tuple[np.ndarray, np.ndarray]:
    """For each ember, find N nearest temporal neighbors.

    Returns (indices (N_embers, n), weights (N_embers, n)) where weight =
    exp(-|Δt| / τ).
    """
    # Pairwise |Δt|
    dt = np.abs(age_days[:, None] - age_days[None, :])  # (N, N)
    np.fill_diagonal(dt, np.inf)  # exclude self
    nn_idx = np.argpartition(dt, n, axis=1)[:, :n]
    rows = np.arange(age_days.shape[0])[:, None]
    dt_topn = dt[rows, nn_idx]
    order = np.argsort(dt_topn, axis=1)
    nn_idx = nn_idx[rows, order]
    dt_topn = dt_topn[rows, order]
    weights = np.exp(-dt_topn / tau_days)
    return nn_idx, weights


async def clear_existing_edges(db: Database, edge_types: list[str]) -> int:
    """Delete prior runs of these edge types so re-runs are idempotent.

    Doesn't commit: main_async commits once after the new edges are in.

    Returns count deleted.
    """
    total = 0
    for et in edge_types:
        cur = await db.conn.execute(
            "DELETE FROM edges WHERE edge_type = ?", (et,)
        )
        # aiosqlite doesn't expose rowcount easily; track via pre-count
        total += cur.rowcount if cur.rowcount is not None else 0
    return total


async def insert_edges(
    db: Database,
    sources: list[str],
    nn_idx: np.ndarray,
    weights: np.ndarray,
    edge_type: str,
    batch_size: int = 500,
) -> int:
    """Bulk-insert edges. Returns count inserted."""
    rows: list[tuple] = []
    n_inserted = 0
    for i, sid in enumerate(sources):
        for j_pos, j in enumerate(nn_idx[i]):
            w = float(weights[i, j_pos])
            rows.append((sid, sources[int(j)], edge_type, w))

    for k in range(0, len(rows), batch_size):
        chunk = rows[k : k + batch_size]
        await db.conn.executemany(
            "INSERT OR REPLACE INTO edges "
            "(source_id, target_id, edge_type, weight) VALUES (?, ?, ?, ?)",
            chunk,
        )
        n_inserted += len(chunk)
    return n_inserted


def graph_properties(
    n: int, knn_idx: np.ndarray, knn_sim: np.ndarray,
    temp_idx: np.ndarray, temp_weight: np.ndarray,
) -> dict:
    """Compute basic health/diagnostic properties of the constructed graph."""
    # Degree distribution: out-degree is fixed (k or n), but in-degree varies
    # which tells us which nodes are hubs
    knn_in_deg = np.zeros(n, dtype=np.int32)
    for row in knn_idx:
        for j in row:
            knn_in_deg[int(j)] += 1

    temp_in_deg = np.zeros(n, dtype=np.int32)
    for row in temp_idx:
        for j in row:
            temp_in_deg[int(j)] += 1

    # Connectivity via union-find on the kNN graph (undirected projection)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(n):
        for j in knn_idx[i]:
            union(i, int(j))

    components = len(set(find(i) for i in range(n)))

    return {
        "n_nodes": n,
        "knn_components": components,
        "knn_in_degree_max": int(knn_in_deg.max()),
        "knn_in_degree_mean": float(knn_in_deg.mean()),
        "knn_in_degree_p95": int(np.percentile(knn_in_deg, 95)),
        "knn_sim_mean": float(knn_sim.mean()),
        "knn_sim_min": float(knn_sim.min()),
        "knn_sim_max": float(knn_sim.max()),
        "temp_in_degree_max": int(temp_in_deg.max()),
        "temp_in_degree_mean": float(temp_in_deg.mean()),
        "temp_weight_mean": float(temp_weight.mean()),
        "temp_weight_min": float(temp_weight.min()),
        "temp_weight_max": float(temp_weight.max()),
    }


async def main_async(args: argparse.Namespace) -> int:
    db = Database()
    await db.connect()
    assert not db._read_only, "freeze must be off for graph build"

    print(f"db_path:    {db.db_path}")
    print(f"k_knn:      {args.k_knn}")
    print(f"n_temporal: {args.n_temporal}")
    print(f"tau_days:   {args.tau_days}")
    print()

    print("─── Loading clustering vectors ──────────────────────────")
    t0 = time.time()
    ids, vecs, age_days = await load_clustering_vectors(db)
    print(f"  loaded {len(ids)} vectors in {time.time()-t0:.2f}s")
    if len(ids) < 2:
        # Nothing to connect. Clear old edges so stale ones don't linger.
        n_deleted = 0 if args.dry_run else await clear_existing_edges(
            db, [KNN_EDGE_TYPE, TEMPORAL_EDGE_TYPE])
        await db.conn.commit()
        await db.close()
        print(f"  fewer than 2 active embers; no graph to build "
              f"(cleared {n_deleted} prior edges)")
        return 0
    # Small stores have fewer neighbors than requested.
    args.k_knn = min(args.k_knn, len(ids) - 1)
    args.n_temporal = min(args.n_temporal, len(ids) - 1)
    print(f"  vector shape: {vecs.shape}, dtype: {vecs.dtype}")
    print(f"  temporal span: {age_days.max():.1f} days")

    print("\n─── Computing kNN edges (semantic) ──────────────────────")
    t0 = time.time()
    knn_idx, knn_sim = compute_knn_edges(vecs, args.k_knn)
    print(f"  kNN computed in {time.time()-t0:.2f}s — "
          f"shape {knn_idx.shape}")
    print(f"  sim range: [{knn_sim.min():.4f}, {knn_sim.max():.4f}], "
          f"mean {knn_sim.mean():.4f}")

    print("\n─── Computing temporal edges ────────────────────────────")
    t0 = time.time()
    temp_idx, temp_w = compute_temporal_edges(
        age_days, args.n_temporal, args.tau_days
    )
    print(f"  temporal computed in {time.time()-t0:.2f}s — "
          f"shape {temp_idx.shape}")
    print(f"  weight range: [{temp_w.min():.4f}, {temp_w.max():.4f}], "
          f"mean {temp_w.mean():.4f}")

    print("\n─── Graph properties ────────────────────────────────────")
    props = graph_properties(len(ids), knn_idx, knn_sim, temp_idx, temp_w)
    for k, v in props.items():
        if isinstance(v, float):
            print(f"  {k:<28} {v:.4f}")
        else:
            print(f"  {k:<28} {v}")

    if props["knn_components"] > 1:
        print(f"\n  WARN: kNN graph has {props['knn_components']} components — "
              "consider higher --k-knn for full connectivity")

    if args.dry_run:
        print("\n(dry-run; no edges written)")
        await db.close()
        return 0

    print("\n─── Clearing existing knn/temporal edges ────────────────")
    n_deleted = await clear_existing_edges(
        db, [KNN_EDGE_TYPE, TEMPORAL_EDGE_TYPE]
    )
    print(f"  cleared {n_deleted} prior edges")

    print("\n─── Inserting kNN edges ─────────────────────────────────")
    t0 = time.time()
    n_knn = await insert_edges(db, ids, knn_idx, knn_sim, KNN_EDGE_TYPE)
    print(f"  inserted {n_knn} in {time.time()-t0:.2f}s")

    print("\n─── Inserting temporal edges ────────────────────────────")
    t0 = time.time()
    n_temp = await insert_edges(db, ids, temp_idx, temp_w, TEMPORAL_EDGE_TYPE)
    print(f"  inserted {n_temp} in {time.time()-t0:.2f}s")

    # One transaction for clear + both inserts, so a live recall never sees
    # the graph half-built.
    await db.conn.commit()

    print("\n─── Verify ─────────────────────────────────────────────")
    cur = await db.conn.execute(
        "SELECT edge_type, COUNT(*) FROM edges GROUP BY edge_type ORDER BY edge_type"
    )
    for row in await cur.fetchall():
        print(f"  {row[0]:<28} {row[1]}")

    await db.close()
    print("\nDone. Set EMBER4_READ_ONLY=1 for normal use.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--k-knn", type=int, default=20,
                        help="k for semantic kNN (default 20)")
    parser.add_argument("--n-temporal", type=int, default=5,
                        help="N nearest temporal neighbors per ember "
                             "(default 5)")
    parser.add_argument("--tau-days", type=float, default=7.0,
                        help="temporal decay scale in days (default 7)")
    parser.add_argument("--dry-run", action="store_true",
                        help="compute graph but do not write to DB")
    args = parser.parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    sys.exit(main())
