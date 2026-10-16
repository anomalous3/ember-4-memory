"""Scoring utilities for HESTIA retrieval and Shadow-Decay framework.

All functions are pure — no DB or I/O. Parameters are passed explicitly.
"""

import math
from datetime import datetime, timezone
from typing import List, Optional, Tuple

from ember4.config import DECAY_HALF_LIVES


def l2_to_cosine(l2: float) -> float:
    """Convert sqlite-vec's L2 distance to cosine similarity.

    sqlite-vec returns plain L2 (not squared): sqrt(2) between orthogonal
    unit vectors. For L2-normalized vectors L2² = 2(1 - cos_sim), so
    cos_sim = 1 - L2²/2, in [-1, 1]. Clipping negatives is a scoring
    policy, not part of the metric. Pinned by ember4_test_metric.py.
    """
    return max(-1.0, min(1.0, 1.0 - (l2 * l2) / 2.0))


def cosine_to_l2(cos_sim: float) -> float:
    """Convert cosine similarity to L2 distance for normalized vectors."""
    cos_sim = max(-1.0, min(1.0, cos_sim))
    return math.sqrt(2.0 * (1.0 - cos_sim))


# ── Temporal Scoring (legacy fallback) ────────────────────────────────────────

def compute_temporal_score(
    importance: str,
    updated_at: datetime,
    access_count: int,
    is_stale: bool,
    query_distance: float,
    now: Optional[datetime] = None,
) -> float:
    if now is None:
        now = datetime.now(timezone.utc)
    semantic_score = max(0.0, l2_to_cosine(query_distance))
    age_days = max(0.0, (now - updated_at).total_seconds() / 86400.0)
    half_life = DECAY_HALF_LIVES.get(importance, 30.0)
    recency_weight = 0.5 ** (age_days / half_life)
    access_boost = math.log(access_count + 1) * 0.1
    staleness_factor = 0.1 if is_stale else 1.0
    return round(
        semantic_score * recency_weight * (1.0 + access_boost) * staleness_factor, 4
    )


# ── Shadow-Decay Framework ───────────────────────────────────────────────────

def compute_shadow_potential(
    cos_sim: float,
    t_target: datetime,
    t_shadower: datetime,
    delta: float = 0.3,
    epsilon: float = 0.05,
) -> float:
    """Shadow potential φ(m_shadower | m_target). Only newer can shadow older."""
    if t_shadower <= t_target:
        return 0.0
    val = (cos_sim - (1.0 - delta)) / (delta - epsilon)
    return max(0.0, min(1.0, val))


def compute_shadow_load(
    ember_cos_sims: List[float],
    ember_time: datetime,
    neighbor_times: List[datetime],
    neighbor_ids: List[str],
    delta: float = 0.3,
    epsilon: float = 0.05,
) -> Tuple[float, Optional[str]]:
    """Max shadow load Φ on an ember from its neighbors.

    Returns (max_shadow_load, dominant_shadower_id).
    """
    max_load = 0.0
    dominant = None
    for sim, n_time, n_id in zip(ember_cos_sims, neighbor_times, neighbor_ids):
        phi = compute_shadow_potential(sim, ember_time, n_time, delta, epsilon)
        if phi > max_load:
            max_load = phi
            dominant = n_id
    return max_load, dominant


# ── HESTIA Retrieval Score ────────────────────────────────────────────────────

def compute_hestia_score(
    cos_sim: float,
    shadow_load: float,
    vitality: float,
    v_max: float,
    gamma: float = 2.0,
    alpha: float = 0.1,
    utility: float = 0.5,
    utility_weight: float = 0.15,
) -> Tuple[float, dict]:
    """HESTIA score: S = cos_sim · (1-Φ)^γ · vitality_factor · utility_factor.

    Returns (score, breakdown_dict).
    """
    shadow_factor = (1.0 - shadow_load) ** gamma
    vitality_factor = (
        alpha + (1.0 - alpha) * (vitality / v_max) if v_max > 0 else alpha
    )
    utility_factor = 1.0 + utility_weight * (utility - 0.5)
    score = cos_sim * shadow_factor * vitality_factor * utility_factor

    return score, {
        "cos_sim": cos_sim,
        "shadow_factor": shadow_factor,
        "vitality_factor": vitality_factor,
        "utility_factor": utility_factor,
    }


# ── Hallucination Risk ───────────────────────────────────────────────────────

def compute_hallucination_risk(
    shadow_loads: List[float],
    stale_flags: List[bool],
    vitalities: List[float],
    v_min: float = 0.01,
) -> dict:
    """Risk = 0.4 × heavily_shadowed + 0.3 × stale_ratio + 0.3 × silent_ratio."""
    total = len(shadow_loads)
    if total == 0:
        return {
            "total": 0, "shadowed_count": 0, "stale_count": 0,
            "silent_topics": 0, "avg_shadow_load": 0.0, "risk_score": 0.0,
        }
    shadowed_count = sum(1 for l in shadow_loads if l > 0.5)
    stale_count = sum(1 for s in stale_flags if s)
    silent_topics = sum(1 for v in vitalities if v < v_min)
    avg_shadow = sum(shadow_loads) / total
    risk = (
        0.4 * shadowed_count / total
        + 0.3 * stale_count / total
        + 0.3 * (silent_topics / len(vitalities) if vitalities else 0.0)
    )
    return {
        "total": total,
        "shadowed_count": shadowed_count,
        "stale_count": stale_count,
        "silent_topics": silent_topics,
        "avg_shadow_load": round(avg_shadow, 4),
        "risk_score": round(risk, 4),
    }


# ── Knowledge Graph Edge Detection ───────────────────────────────────────────

def detect_kg_edges(
    cos_sims: List[float],
    shadow_potentials: List[float],
    neighbor_ids: List[str],
    threshold: float = 0.4,
    max_edges: int = 5,
) -> List[str]:
    """Related but NOT shadowing: cos_sim > threshold AND φ < 0.1."""
    candidates = [
        (sim, nid)
        for sim, phi, nid in zip(cos_sims, shadow_potentials, neighbor_ids)
        if sim > threshold and phi < 0.1
    ]
    candidates.sort(key=lambda x: x[0], reverse=True)
    return [c[1] for c in candidates[:max_edges]]


# ── Region Stats Update ──────────────────────────────────────────────────────

def update_region_shadow(
    old_accum: float,
    phi_value: float,
    ema_alpha: float = 0.1,
) -> float:
    """EMA update for per-cell conflict density."""
    return (1.0 - ema_alpha) * old_accum + ema_alpha * phi_value
