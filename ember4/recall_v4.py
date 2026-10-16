"""ember-4 graph-based recall: Personalized PageRank over per-type stochastic mix.

Design:

1. Per-type transition matrices, not raw edge-weight mix. Each edge type is
   normalized into its own row-stochastic matrix P_type, then mixed:
       P = Σ w_type · P_type
   This stops "edge count / hub degree" from masquerading as semantic
   importance.

2. Hub penalty BEFORE row-normalization: w_ij /= log(2 + in_degree_j_by_type).
   Without this, even row-stochastic P_type routes probability into hubs.

3. kNN edges symmetrized via max(w_ij, w_ji) — directed kNN amplifies
   hub/sink pathology.

4. KG subtype mass split: derived_from/elaborates/supports get more than
   generic `related`; `contrasts_with` gets zero diffusion mass (reserved
   for sidebar/diversity use later); `shadow` ignored entirely.

5. Seeds come from RRF over (dense top-K, BM25 top-K). Teleport is mass-
   weighted by RRF score, not equal-split.

6. Final score = weighted sum of rank-normalized features (PPR rank, dense
   rank, BM25 rank). Feature vector is returned per-result for logging —
   "the logs matter more than the exact initial weights."

PPR α=0.25 (0.15 predicted too diffusive on a connected k=20 graph).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import scipy.sparse as sp

from ember4.embedder import Embedder, serialize_vector
from ember4.utils import l2_to_cosine

try:
    from ember4.config import (
        RECENCY_STRENGTH, RECENCY_TAU_DAYS, RECENCY_EXEMPT_TAGS,
    )
except Exception:  # pragma: no cover — config import safety for standalone tests
    RECENCY_STRENGTH = 0.75
    RECENCY_TAU_DAYS = 21.0
    RECENCY_EXEMPT_TAGS = ("foundation", "principle")


# ─── Configuration ────────────────────────────────────────────────────────────

# Stochastic-matrix mixture weights (sum to 1.0)
EDGE_TYPE_WEIGHTS: dict[str, float] = {
    "knn_nomic_v15":   0.40,   # semantic manifold backbone
    "tag_cohort":      0.25,   # project-cohort adjacency from name patterns
                                # (kNN alone under-samples a project's
                                # version-to-version trajectory)
    "temporal_nomic":  0.05,   # burst-mate proximity (deprioritized — see below)
    "derived_from":    0.15,   # human-curated chains, high signal
    "elaborates":      0.05,
    "supports":        0.05,
    "related":         0.05,   # generic edge type, lower trust
    "contrasts_with":  0.00,   # NOT proximity; reserved for diversity later
    # 'shadow' deliberately absent — legacy decay artifact, ignored
}

PPR_ALPHA = 0.25       # 0.15 predicted too diffusive on k=20
PPR_MAX_ITER = 100
PPR_TOL = 1e-8

# Final-score feature weights: a dense-primary blend. Starting values from a
# small ablation: a PPR-primary blend let diffusion into hub memories demote
# direct matches, and conditioned activity carried most of the gain over
# dense + BM25 alone. PPR keeps a near-zero score weight, but it still widens
# the candidate set. Log the feature vector and re-tune on your own data.
FEATURE_WEIGHTS: dict[str, float] = {
    "dense":          0.40,
    "bm25":           0.30,
    "cond_activity":  0.28,
    "ppr":            0.02,
}

# Recency decay constant for conditioned activity (days). 14 days = 2 weeks;
# events older than ~6 weeks contribute < 5% of their weight.
COND_ACTIVITY_TAU_DAYS = 14.0

# Small-weight contribution for human/agent tag_update events with no
# content change. Full-weight events (edit, contradict, edge_form) get 1.0.
TAG_UPDATE_WEIGHT = 0.3

# RRF damping constant for rank-normalization. RRF is used both for seeds
# and as a final feature. 1/rank decays too aggressively — a rank-#13 target
# gets crushed by rank-#1-in-one-method-only results. 1/(60+rank) is much
# gentler: rank-#1=0.016, rank-#13=0.014, rank-#50=0.009. Standard k=60
# from the original Cormack et al. RRF paper.
RRF_K = 60

# Stop-word handling moved to query_prep.ENGLISH_STOPWORDS (proper ~170-word
# list) + corpus-specific stop words computed from ember DF. Imported lazily
# inside gather_seeds_bm25 to avoid a circular dependency at module load.


# ─── Data containers ─────────────────────────────────────────────────────────

@dataclass
class RecallResult:
    ember_id: str
    score: float
    features: dict[str, float]
    name: str = ""

@dataclass
class RecallTrace:
    """Diagnostic info captured during recall — the logs matter more than the weights."""
    n_seeds_dense: int
    n_seeds_bm25:  int
    n_candidates:  int
    ppr_iters:     int
    ppr_final_delta: float
    cond_signal_alive: bool = False  # any update_log activity reached a candidate?
    n_cond_act_nonzero: int = 0       # how many candidates had nonzero cond_act
    seed_ids:      list[str] = field(default_factory=list)
    edge_type_used: dict[str, int] = field(default_factory=dict)  # nnz per type


# ─── Graph state loading ─────────────────────────────────────────────────────

async def load_graph_state(
    db, edge_types: Optional[list[str]] = None, include_stale: bool = False,
) -> tuple[list[str], dict[str, int], dict[str, sp.csr_matrix]]:
    """Load all ember-4-relevant edges into per-type sparse adjacency matrices.

    Returns:
        ids: list of ember_ids in canonical order (sorted ember_id)
        id_to_idx: dict mapping ember_id → row index
        matrices: dict mapping edge_type → CSR matrix of weights, shape (N, N)

    Embers without any matching edges still occupy their row index — they
    just have zero rows in that type's matrix, which PPR handles via the
    α-teleport.

    Stale embers (superseded by ember_contradict or synthesis) get no row
    unless include_stale, so edges into them drop out and no mass diffuses
    through them.
    """
    if edge_types is None:
        edge_types = [t for t, w in EDGE_TYPE_WEIGHTS.items() if w > 0.0]

    cur = await db.conn.execute(
        "SELECT ember_id FROM embers "
        + ("" if include_stale else "WHERE is_stale = 0 ")
        + "ORDER BY ember_id"
    )
    ids = [r["ember_id"] for r in await cur.fetchall()]
    id_to_idx = {eid: i for i, eid in enumerate(ids)}
    n = len(ids)

    matrices: dict[str, sp.csr_matrix] = {}
    for edge_type in edge_types:
        cur = await db.conn.execute(
            "SELECT source_id, target_id, weight FROM edges "
            "WHERE edge_type = ?",
            (edge_type,),
        )
        rows, cols, data = [], [], []
        for r in await cur.fetchall():
            s = id_to_idx.get(r["source_id"])
            t = id_to_idx.get(r["target_id"])
            if s is None or t is None:
                continue
            rows.append(s)
            cols.append(t)
            data.append(float(r["weight"]))
        if rows:
            M = sp.coo_matrix(
                (data, (rows, cols)), shape=(n, n), dtype=np.float64
            ).tocsr()
        else:
            M = sp.csr_matrix((n, n), dtype=np.float64)
        matrices[edge_type] = M

    return ids, id_to_idx, matrices


# ─── Transition matrix construction ──────────────────────────────────────────

def build_transition_matrix(
    matrices: dict[str, sp.csr_matrix],
    symmetrize_knn: bool = True,
    hub_penalty: bool = True,
) -> sp.csr_matrix:
    """Per-type normalize → mix into one stochastic transition matrix.

    Steps:
      1. Symmetrize kNN with max(w_ij, w_ji) to neuter directed-edge hub bias.
      2. Apply hub penalty w_ij /= log(2 + indeg_j) BEFORE row-normalize.
      3. Row-normalize → P_type (rows sum to 1 where any out-edge exists).
      4. Mix: P = Σ EDGE_TYPE_WEIGHTS[type] · P_type.

    Dangling rows (embers with no out-edges in any type) end up zero in P.
    PPR handles those via α-teleport: random walker just jumps back to seeds.
    """
    P_total: Optional[sp.csr_matrix] = None
    for edge_type, weight in EDGE_TYPE_WEIGHTS.items():
        if weight == 0.0:
            continue
        M = matrices.get(edge_type)
        if M is None or M.nnz == 0:
            continue
        M = M.astype(np.float64, copy=True)

        # 1. Symmetrize kNN — directed kNN strengthens hubs/sinks
        if edge_type == "knn_nomic_v15" and symmetrize_knn:
            M = M.maximum(M.T)

        # 2. Hub penalty: divide by log(2 + in-degree) per column
        if hub_penalty:
            in_deg = np.asarray(M.sum(axis=0)).flatten()  # column sums
            penalty = np.log(2 + in_deg)
            inv_pen = np.where(penalty > 0, 1.0 / penalty, 0.0)
            # Right-multiply by diag(inv_pen) scales columns
            M = M @ sp.diags(inv_pen)

        # 3. Row-normalize. Clip to avoid divide-by-zero RuntimeWarning;
        # genuine-zero rows produce zero out-mass (PPR α-teleport handles them).
        row_sums = np.asarray(M.sum(axis=1)).flatten()
        inv_rs = np.zeros_like(row_sums)
        nz = row_sums > 0
        inv_rs[nz] = 1.0 / row_sums[nz]
        P_type = sp.diags(inv_rs) @ M  # type: ignore[operator]
        P_type = P_type.tocsr()

        # 4. Mix
        if P_total is None:
            P_total = weight * P_type
        else:
            P_total = (P_total + weight * P_type).tocsr()

    if P_total is None:
        # No edges yet (fresh install, or before ember4_build_graph.py has
        # run): every row is dangling, so PPR reduces to the teleport seeds.
        if not matrices:
            raise RuntimeError("no edge types loaded")
        return sp.csr_matrix(next(iter(matrices.values())).shape, dtype=np.float64)
    return P_total


# ─── Personalized PageRank ───────────────────────────────────────────────────

def personalized_pagerank(
    P: sp.csr_matrix,
    p0: np.ndarray,
    alpha: float = PPR_ALPHA,
    max_iter: int = PPR_MAX_ITER,
    tol: float = PPR_TOL,
) -> tuple[np.ndarray, int, float]:
    """PPR via power iteration.

    Update rule: r_{t+1} = α · p0 + (1 - α) · P^T r_t

    Returns (stationary distribution, n_iterations_used, final L1 delta).
    """
    assert P.shape[0] == P.shape[1] == p0.shape[0]
    r = p0.copy().astype(np.float64)
    PT = P.T.tocsr()  # precompute for repeated multiplies
    final_delta = float("inf")
    iters_used = 0
    for it in range(1, max_iter + 1):
        r_new = alpha * p0 + (1.0 - alpha) * (PT @ r)
        final_delta = float(np.abs(r_new - r).sum())
        r = r_new
        iters_used = it
        if final_delta < tol:
            break
    return r, iters_used, final_delta


# ─── Seed gathering ──────────────────────────────────────────────────────────

VEC0_MAX_K = 4096  # sqlite-vec rejects larger k in a knn query

async def gather_seeds_dense(
    db, embedder: Embedder, query: str, k: int = 20,
    exclude: frozenset = frozenset(),
) -> list[tuple[str, float]]:
    """Dense retrieval using search_query: prefix.

    vec0 can't filter, so over-fetch by len(exclude) and drop excluded ids.
    Past sqlite-vec's k cap that could underfill, so then score every
    active vector exactly instead.
    """
    qvec = embedder.embed_query(query)
    qblob = serialize_vector(qvec)
    if k + len(exclude) <= VEC0_MAX_K:
        cur = await db.conn.execute(
            "SELECT ember_id, distance FROM vec_embers_nomic_v15_retrieval "
            "WHERE embedding MATCH ? AND k = ? ORDER BY distance",
            (qblob, k + len(exclude)),
        )
        rows = [(r["ember_id"], r["distance"]) for r in await cur.fetchall()
                if r["ember_id"] not in exclude]
        return [(eid, l2_to_cosine(d)) for eid, d in rows[:k]]
    cur = await db.conn.execute(
        "SELECT ember_id, embedding FROM vec_embers_nomic_v15_retrieval")
    ids, vecs = [], []
    for r in await cur.fetchall():
        if r["ember_id"] not in exclude:
            ids.append(r["ember_id"])
            vecs.append(np.frombuffer(r["embedding"], dtype=np.float32))
    if not ids:
        return []
    sims = np.stack(vecs) @ np.asarray(qvec, dtype=np.float32)
    top = np.argsort(-sims, kind="stable")[:k]
    return [(ids[i], float(sims[i])) for i in top]


async def gather_seeds_bm25(
    db, query: str, k: int = 20, exclude: frozenset = frozenset(),
) -> list[tuple[str, float]]:
    """BM25 sparse retrieval over embers_fts.

    Uses query_prep.prepare_bm25_query to strip English + corpus-specific
    stop words (2026-05-15 enhancement). The corpus stopword list is
    cached in db config and computed once from current ember content;
    rebuild via ember4_rebuild_stopwords.py if the corpus drifts.
    """
    from ember4.query_prep import prepare_bm25_query, get_corpus_stopwords

    corpus_stops = await get_corpus_stopwords(db)
    fts_q = prepare_bm25_query(query, corpus_stopwords=corpus_stops)
    if not fts_q:
        return []
    try:
        cur = await db.conn.execute(
            "SELECT ember_id, bm25(embers_fts) AS score "
            "FROM embers_fts WHERE embers_fts MATCH ? "
            "ORDER BY bm25(embers_fts) LIMIT ?",
            (fts_q, k + len(exclude)),
        )
        rows = [r for r in await cur.fetchall() if r["ember_id"] not in exclude]
        # bm25() returns negative scores; flip so higher = better
        return [(r["ember_id"], -float(r["score"])) for r in rows[:k]]
    except Exception:
        return []


def reciprocal_rank_fusion(
    method_rankings: dict[str, list[str]], k: int = 60
) -> dict[str, float]:
    """RRF combine multiple rankings into a single score per ember.

    rrf_score[e] = Σ_methods 1 / (k + rank_in_method)
    """
    scores: dict[str, float] = {}
    for method, ranked_ids in method_rankings.items():
        for i, eid in enumerate(ranked_ids):
            scores[eid] = scores.get(eid, 0.0) + 1.0 / (k + (i + 1))
    return scores


# ─── Activity features (task 6, 2026-05-13) ──────────────────────────────────

async def compute_global_activity(
    db,
    candidate_ids: list[str],
    tau_days: float = COND_ACTIVITY_TAU_DAYS,
) -> dict[str, float]:
    """Per-candidate recency-weighted activity, with NO query conditioning.

    This is the "hub smell" baseline to ablate against. If
    global_activity beats conditioned_activity on the benchmark, it means
    global popularity is doing the work — which is the basin pathology
    repackaged. Conditioned should match or beat global; equal is OK,
    global > conditioned is a red flag.
    """
    if not candidate_ids:
        return {}
    ph = ",".join(["?"] * len(candidate_ids))
    sql = f"""
    SELECT
        ember_id,
        SUM(
            CASE
                WHEN event_type IN ('edit','contradict_old','contradict_new','edge_form')
                     AND origin IN ('user','agent')
                THEN 1.0
                WHEN event_type = 'tag_update' AND origin IN ('user','agent')
                     AND content_changed = 0
                THEN ?
                ELSE 0.0
            END
            * EXP(-(julianday('now') - julianday(timestamp)) / ?)
        ) AS activity
    FROM update_log
    WHERE ember_id IN ({ph})
    GROUP BY ember_id
    """
    params = [TAG_UPDATE_WEIGHT, tau_days] + list(candidate_ids)
    cur = await db.conn.execute(sql, params)
    rows = await cur.fetchall()
    result = {c: 0.0 for c in candidate_ids}
    for r in rows:
        result[r["ember_id"]] = float(r["activity"] or 0.0)
    return result


async def compute_conditioned_activity(
    db,
    candidate_ids: list[str],
    seed_ids: list[str],
    tau_days: float = COND_ACTIVITY_TAU_DAYS,
) -> dict[str, float]:
    """For each candidate, compute query-conditioned engagement.

    The signal is:
        Σ over embers x that are tag_cohort-bridged to BOTH:
            (a) the candidate, AND
            (b) at least one of the query's seed embers
          of   event_weight(x) × exp(-Δt(x) / tau_days)
        for events on x that pass the contribution filter:
            (event_type IN {edit, contradict_old, contradict_new, edge_form}
             AND origin IN {user, agent})
            OR (event_type = 'tag_update' AND origin IN {user, agent}
                AND content_changed = 0)   -- small TAG_UPDATE_WEIGHT

    This conditions the engagement signal on the query: an atlas query gets
    boosted by atlas-cohort activity; an ember-design query doesn't pull
    on atlas activity even if atlas is globally hot. Engagement is
    project/thread-conditioned, not global.

    With an empty update_log (cold start), returns 0.0 for every candidate.
    """
    if not candidate_ids or not seed_ids:
        return {c: 0.0 for c in candidate_ids}

    cph = ",".join(["?"] * len(candidate_ids))
    sph = ",".join(["?"] * len(seed_ids))

    # tag_cohort edges were inserted directed (source → target). The
    # cohort relation is symmetric in intent, so we union both directions.
    sql = f"""
    WITH q_neighbors AS (
        SELECT DISTINCT target_id AS x_id FROM edges
        WHERE source_id IN ({sph}) AND edge_type = 'tag_cohort'
        UNION
        SELECT DISTINCT source_id AS x_id FROM edges
        WHERE target_id IN ({sph}) AND edge_type = 'tag_cohort'
    ),
    cand_neighbors AS (
        SELECT source_id AS cand_id, target_id AS x_id
        FROM edges
        WHERE source_id IN ({cph}) AND edge_type = 'tag_cohort'
        UNION
        SELECT target_id AS cand_id, source_id AS x_id
        FROM edges
        WHERE target_id IN ({cph}) AND edge_type = 'tag_cohort'
    ),
    qualifying_events AS (
        SELECT
            ember_id,
            timestamp,
            CASE
                WHEN event_type IN ('edit','contradict_old','contradict_new','edge_form')
                     AND origin IN ('user','agent')
                THEN 1.0
                WHEN event_type = 'tag_update' AND origin IN ('user','agent')
                     AND content_changed = 0
                THEN ?
                ELSE 0.0
            END AS event_weight
        FROM update_log
    )
    SELECT
        cn.cand_id AS cand_id,
        SUM(
            qe.event_weight
            * EXP(-(julianday('now') - julianday(qe.timestamp)) / ?)
        ) AS cond_act
    FROM cand_neighbors cn
    JOIN q_neighbors    qn ON cn.x_id = qn.x_id
    JOIN qualifying_events qe ON qe.ember_id = cn.x_id
    WHERE qe.event_weight > 0
    GROUP BY cn.cand_id
    """

    params = (
        list(seed_ids) + list(seed_ids)
        + list(candidate_ids) + list(candidate_ids)
        + [TAG_UPDATE_WEIGHT, tau_days]
    )
    cur = await db.conn.execute(sql, params)
    rows = await cur.fetchall()

    result = {c: 0.0 for c in candidate_ids}
    for r in rows:
        result[r["cand_id"]] = float(r["cond_act"] or 0.0)
    return result


# ─── Recency (anti-ghost) ─────────────────────────────────────────────────────

def compute_recency_factor(
    age_days: float, strength: float, tau_days: float
) -> float:
    """Multiplicative demotion for stale embers (1.0 = no effect).

        recency_factor = 1 - strength * (1 - exp(-age/tau))

    strength=0 → 1.0 always (deep_recall / wonder reach back freely). An old
    ember bottoms out at a floor of (1-strength), so a very strong dense+bm25
    match on an old-but-specific gem still surfaces — recency re-ranks, it does
    not gate. Guards against "ghosts of defunct architectures": a
    conceptually-similar past attempt shouldn't rank as current.
    """
    if strength <= 0.0:
        return 1.0
    decay = float(np.exp(-age_days / tau_days)) if tau_days > 0 else 0.0
    return 1.0 - strength * (1.0 - decay)


async def fetch_recency_meta(
    db, candidate_ids: list[str]
) -> dict[str, tuple[float, bool]]:
    """Per-candidate (age_days, is_exempt) for the recency re-rank.

    age_days = days since last *meaningful touch* = creation or content
    edit/contradict (both bump created_at/updated_at), NOT mere access.
    last_accessed_at is deliberately excluded: recalling a ghost bumps access,
    which would let ghosts self-refresh. max(created, updated) protects durable
    principles — a reinforced/contradicted gem has a recent updated_at even with
    old creation, so it reads as fresh.

    is_exempt = ember carries a RECENCY_EXEMPT_TAGS tag (foundation/principle):
    timeless knowledge that has EARNED a permanent recall slot. Coupled
    discipline: a superseded exempt ember must be ember_contradict'd, or it
    becomes a permanent ghost.
    """
    if not candidate_ids:
        return {}
    ph = ",".join(["?"] * len(candidate_ids))
    sql = f"""
        SELECT ember_id, tags,
               (julianday('now') - MAX(julianday(created_at), julianday(updated_at)))
                   AS age_days
        FROM embers WHERE ember_id IN ({ph})
    """
    cur = await db.conn.execute(sql, list(candidate_ids))
    rows = await cur.fetchall()
    out: dict[str, tuple[float, bool]] = {c: (0.0, False) for c in candidate_ids}
    for r in rows:
        age = max(0.0, float(r["age_days"] or 0.0))
        exempt = False
        raw = r["tags"]
        if raw:
            try:
                import json
                tagset = set(json.loads(raw))
                exempt = any(t in tagset for t in RECENCY_EXEMPT_TAGS)
            except Exception:
                exempt = False
        out[r["ember_id"]] = (age, exempt)
    return out


# ─── Full recall pipeline ────────────────────────────────────────────────────

def shared_ranks(values: np.ndarray) -> np.ndarray:
    """Descending competition ranks (1 = highest); equal values share a rank.

    Arbitrary distinct ranks for ties (set or hash order) would let
    iteration order masquerade as a feature signal.
    """
    neg = -np.asarray(values, dtype=np.float64)
    return np.searchsorted(np.sort(neg), neg, side="left") + 1


async def graph_recall(
    db,
    embedder: Embedder,
    query: str,
    top_k: int = 10,
    seed_k: int = 20,
    return_trace: bool = True,
    recency_strength: float = RECENCY_STRENGTH,
    recency_tau_days: float = RECENCY_TAU_DAYS,
    include_stale: bool = False,
) -> tuple[list[RecallResult], Optional[RecallTrace]]:
    """ember-4 graph-based recall: dense + BM25 seeds → RRF teleport → PPR → blend.

    recency_strength (0..1) demotes stale embers as a final multiplicative
    re-rank (see compute_recency_factor); pass 0 to disable.

    Active recall leaves out stale embers (superseded by ember_contradict or
    synthesis) at every stage: seeds, graph, candidates. include_stale=True
    is historical recall: old versions are seeds and graph nodes again.

    Returns (results, trace). trace=None if return_trace=False.
    """
    # 1. Gather seeds from both pathways
    stale: frozenset = frozenset()
    if not include_stale:
        cur = await db.conn.execute("SELECT ember_id FROM embers WHERE is_stale = 1")
        stale = frozenset(r["ember_id"] for r in await cur.fetchall())
    dense_seeds = await gather_seeds_dense(db, embedder, query, seed_k, exclude=stale)
    bm25_seeds = await gather_seeds_bm25(db, query, seed_k, exclude=stale)

    if not dense_seeds and not bm25_seeds:
        return [], (
            RecallTrace(0, 0, 0, 0, 0.0) if return_trace else None
        )

    # 2. RRF to define teleport mass distribution
    rrf_scores = reciprocal_rank_fusion({
        "dense": [eid for eid, _ in dense_seeds],
        "bm25":  [eid for eid, _ in bm25_seeds],
    })

    # 3. Load graph + build transition matrix
    ids, id_to_idx, matrices = await load_graph_state(db, include_stale=include_stale)
    n = len(ids)
    P = build_transition_matrix(matrices)

    # 4. Build teleport distribution p0 — NOT equal mass
    p0 = np.zeros(n, dtype=np.float64)
    for eid, score in rrf_scores.items():
        idx = id_to_idx.get(eid)
        if idx is not None:
            p0[idx] = score
    s = p0.sum()
    if s <= 0:
        return [], (
            RecallTrace(len(dense_seeds), len(bm25_seeds), 0, 0, 0.0)
            if return_trace else None
        )
    p0 /= s

    # 5. PPR
    ppr_score, iters, delta = personalized_pagerank(
        P, p0, alpha=PPR_ALPHA, max_iter=PPR_MAX_ITER, tol=PPR_TOL
    )

    # 6. Build candidate set: top from PPR ∪ dense ∪ bm25
    ppr_top = np.argsort(-ppr_score)[: top_k * 3]
    cand: set[str] = set(ids[i] for i in ppr_top)
    cand.update(eid for eid, _ in dense_seeds[: top_k * 2])
    cand.update(eid for eid, _ in bm25_seeds[: top_k * 2])

    # 7. Conditioned activity per candidate (query-cohort-restricted engagement)
    cand_list = sorted(cand)
    seed_ids_for_cohort = [eid for eid, _ in dense_seeds[:10]]  # top-10 dense as cohort seeds
    cond_act_by_id = await compute_conditioned_activity(
        db, candidate_ids=cand_list, seed_ids=seed_ids_for_cohort,
    )
    # Rank candidates by cond_activity (descending); equal activity, equal rank.
    cond_rank = dict(zip(cand_list, shared_ranks(
        [cond_act_by_id.get(c, 0.0) for c in cand_list]).tolist()))

    # 8. Rank-normalize features (RRF damping; gentle decay rewards consensus)
    dense_rank = {eid: i + 1 for i, (eid, _) in enumerate(dense_seeds)}
    bm25_rank  = {eid: i + 1 for i, (eid, _) in enumerate(bm25_seeds)}
    ppr_rank = dict(zip(ids, shared_ranks(ppr_score).tolist()))

    # If update_log is empty (cold start), cond_act is uniformly 0 and the
    # cond_rank is essentially arbitrary. Detect and skip the feature to
    # avoid noise contamination.
    cond_signal_alive = any(v > 0.0 for v in cond_act_by_id.values())

    # Recency: (age_days, is_exempt) per candidate (anti-ghost).
    recency_meta = await fetch_recency_meta(db, cand_list)

    results: list[RecallResult] = []
    for eid in cand_list:
        idx = id_to_idx.get(eid)
        if idx is None:
            continue
        N_default = max(top_k * 3, seed_k * 2)
        d_r = dense_rank.get(eid, N_default + 1)
        b_r = bm25_rank.get(eid, N_default + 1)
        p_r = ppr_rank.get(eid, n)
        c_r = cond_rank.get(eid, len(cand_list))
        c_raw = cond_act_by_id.get(eid, 0.0)

        feats = {
            "ppr_score":  float(ppr_score[idx]),
            "ppr_rank":   p_r,
            "dense_rank": d_r,
            "bm25_rank":  b_r,
            "cond_rank":  c_r,
            "cond_act":   c_raw,
            # RRF damping (gentle): rewards consensus across pathways
            "ppr_f":           1.0 / (RRF_K + p_r),
            "dense_f":         1.0 / (RRF_K + d_r),
            "bm25_f":          1.0 / (RRF_K + b_r),
            "cond_activity_f": (1.0 / (RRF_K + c_r)) if cond_signal_alive else 0.0,
        }
        # Renormalize weights to drop cond_activity contribution when signal is dead
        if cond_signal_alive:
            score = (
                FEATURE_WEIGHTS["dense"]         * feats["dense_f"]
                + FEATURE_WEIGHTS["bm25"]        * feats["bm25_f"]
                + FEATURE_WEIGHTS["cond_activity"] * feats["cond_activity_f"]
                + FEATURE_WEIGHTS["ppr"]         * feats["ppr_f"]
            )
        else:
            # Cold start: redistribute cond_activity weight proportionally
            wD = FEATURE_WEIGHTS["dense"]
            wB = FEATURE_WEIGHTS["bm25"]
            wP = FEATURE_WEIGHTS["ppr"]
            wC = FEATURE_WEIGHTS["cond_activity"]
            denom = wD + wB + wP
            score = (
                (wD + wC * wD / denom) * feats["dense_f"]
                + (wB + wC * wB / denom) * feats["bm25_f"]
                + (wP + wC * wP / denom) * feats["ppr_f"]
            )

        # Recency re-rank: multiplicative demotion of stale embers. Applied
        # AFTER the consensus blend so it scales the whole score uniformly and
        # cannot flip a strong multi-pathway match below a weak recent one
        # (floor = 1-strength). cond_activity (engagement recency) and this
        # (content recency) are complementary: together they bury the ghost
        # — old content AND cold engagement — while sparing reinforced gems.
        age_days, is_exempt = recency_meta.get(eid, (0.0, False))
        rec_factor = 1.0 if is_exempt else compute_recency_factor(
            age_days, recency_strength, recency_tau_days
        )
        feats["age_days"] = age_days
        feats["recency_exempt"] = is_exempt
        feats["recency_factor"] = rec_factor
        score = score * rec_factor

        results.append(RecallResult(
            ember_id=eid, score=score, features=feats
        ))

    results.sort(key=lambda r: (-r.score, r.ember_id))
    results = results[:top_k]

    # 8. Fetch names for display
    if results:
        placeholders = ",".join("?" * len(results))
        cur = await db.conn.execute(
            f"SELECT ember_id, name FROM embers WHERE ember_id IN ({placeholders})",
            [r.ember_id for r in results],
        )
        name_by_id = {row["ember_id"]: row["name"] for row in await cur.fetchall()}
        for r in results:
            r.name = name_by_id.get(r.ember_id, "")

    trace = None
    if return_trace:
        trace = RecallTrace(
            n_seeds_dense=len(dense_seeds),
            n_seeds_bm25=len(bm25_seeds),
            n_candidates=len(cand),
            ppr_iters=iters,
            ppr_final_delta=delta,
            cond_signal_alive=cond_signal_alive,
            n_cond_act_nonzero=sum(1 for v in cond_act_by_id.values() if v > 0.0),
            seed_ids=[eid for eid, _ in dense_seeds[:5]],
            edge_type_used={t: int(m.nnz) for t, m in matrices.items()},
        )
    return results, trace
