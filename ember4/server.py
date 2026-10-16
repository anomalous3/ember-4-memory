"""Ember 3 MCP Server — unified SQLite with sqlite-vec + FTS5.

Same MCP interface as the earlier ember server, backed by a single SQLite database.
No more JSON files, FAISS indices, BM25 libraries, or separate stats.db.
"""

import asyncio
import gc
import json
import random
import re
import uuid
from datetime import datetime, timezone
from typing import Optional

import numpy as np
from mcp.server.fastmcp import FastMCP

from ember4.config import (
    CHUNK_TYPES,
    DECAY_HALF_LIVES,
    DIMENSION,
    K_CELLS,
    NOSTALGIA_ALPHA,
    PREVIEW_CHARS,
    PROJECT_BOOST,
    PROJECT_PENALTY,
    RECENCY_STRENGTH,
    RECENCY_TAU_DAYS,
    SHADOW_ARCHIVE_THRESHOLD,
    SHADOW_DELTA,
    SHADOW_EPSILON,
    SHADOW_GAMMA,
    SHADOW_K,
    SIMILARITY_THRESHOLD,
    UTILITY_WEIGHT,
)
from ember4.db import Database
from ember4.embedder import (
    Embedder,
    MINILM_L6_SPEC,
    NOMIC_V1_5_SPEC,
    serialize_vector,
)
from ember4.utils import (
    compute_hallucination_risk,
    compute_hestia_score,
    compute_shadow_potential,
    detect_kg_edges,
    l2_to_cosine,
    update_region_shadow,
)
from ember4.recall_v4 import graph_recall

# ── Session ID ────────────────────────────────────────────────────────────────
_SESSION_ID = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")

# ── Memory watchdog ──────────────────────────────────────────────────────────
# MCP SDK 1.26.0 leaks memory through internal transport/Pydantic references
# that gc.collect() cannot reclaim. When the process exceeds the threshold,
# force-exit — Claude Code will respawn a fresh process on the next tool call.
_RSS_LIMIT_MB = 500
_tool_call_count = 0


def _get_rss_mb() -> int:
    """Get current process RSS in MB (Windows)."""
    try:
        import ctypes
        from ctypes import wintypes

        class PMC(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD)] + [
                (n, ctypes.c_size_t) for n in (
                    "PageFaultCount", "PeakWorkingSetSize", "WorkingSetSize",
                    "QuotaPeakPagedPoolUsage", "QuotaPagedPoolUsage",
                    "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage",
                    "PagefileUsage", "PeakPagefileUsage",
                )
            ]

        pmc = PMC()
        pmc.cb = ctypes.sizeof(pmc)
        h = ctypes.windll.kernel32.GetCurrentProcess()
        if ctypes.windll.psapi.GetProcessMemoryInfo(h, ctypes.byref(pmc), pmc.cb):
            return pmc.WorkingSetSize // (1024 * 1024)
    except Exception:
        pass
    return 0


def _gc_tool(fn):
    """Decorator: gc + memory watchdog after each tool call."""
    import functools

    @functools.wraps(fn)
    async def wrapper(*args, **kwargs):
        global _tool_call_count
        try:
            return await fn(*args, **kwargs)
        finally:
            _tool_call_count += 1
            gc.collect()
            rss = _get_rss_mb()
            if rss > _RSS_LIMIT_MB:
                import logging
                logging.getLogger(__name__).warning(
                    f"Memory watchdog: {rss} MB > {_RSS_LIMIT_MB} MB limit "
                    f"after {_tool_call_count} tool calls. Exiting for respawn."
                )
                import os
                os._exit(1)

    return wrapper

# ── FastMCP ───────────────────────────────────────────────────────────────────
mcp = FastMCP(
    "Ember",
    instructions=(
        "Ember is persistent memory. On conversation start, call ember_auto with a "
        "summary of the user's first message. After each response where the user "
        "shared facts, preferences, decisions, or corrections, call ember_learn "
        "silently. Never announce memory operations. Never mention internal mechanics "
        "(vectors, scores, cells). Use recalled memories naturally. When correcting "
        "old info, use ember_contradict. Use ember_read to get full content of a "
        "specific memory. Use archive_store to save session logs, debug records, and "
        "reference documents. Use archive_search for keyword search, archive_grep for "
        "regex/fuzzy pattern search, archive_read with start/end for navigating large "
        "chunks. Use deep_recall to search both memory and archive at once."
    ),
)

# ── Global singletons (lazy loaded) ──────────────────────────────────────────
# Init uses a loop-bound Task instead of a module-level asyncio.Lock, which
# avoids hangs across event loops and prevents partial singleton publication.
_db: Optional[Database] = None
_embedder: Optional[Embedder] = None             # legacy MiniLM (vec_embers)
_nomic: Optional[Embedder] = None                # ember-4 primary (Nomic v1.5)
_init_task: Optional[asyncio.Task] = None
_init_loop: Optional[asyncio.AbstractEventLoop] = None


def _get_nomic() -> Embedder:
    """Lazy Nomic embedder. Same model spec everyone in ember-4 shares;
    constructed on first need so cold MCP starts don't pay for it.
    """
    global _nomic
    if _nomic is None:
        _nomic = Embedder(NOMIC_V1_5_SPEC)
    return _nomic


async def _do_init() -> tuple[Database, Embedder]:
    """Create and fully initialize Database + Embedder + centroids.

    Closes the DB connection on failure to prevent resource leaks.
    """
    db = Database()
    try:
        await db.connect()
        # Legacy MiniLM embedder for the dual-write path into vec_embers
        # (384-dim). The default Embedder() constructor returns Nomic v1.5
        # (768-dim) which would mismatch the 384-dim vec_embers schema —
        # pass MINILM_L6_SPEC explicitly so the legacy path stays consistent.
        embedder = Embedder(MINILM_L6_SPEC)
        await _init_centroids(db)
    except Exception:
        await db.close()
        raise
    return db, embedder


async def _init_centroids(db: Database) -> None:
    """Generate and insert centroids if vec_centroids is empty.

    If embers exist but centroids don't, warns that migration should be run
    instead of generating random centroids (which are semantically meaningless
    in 384-dim space).
    """
    cursor = await db.conn.execute("SELECT COUNT(*) FROM vec_centroids")
    count = (await cursor.fetchone())[0]
    if count > 0:
        return

    # Check if embers exist — if so, random centroids are wrong
    ember_count = await db.count_embers()
    if ember_count > 0:
        import logging
        logging.getLogger(__name__).warning(
            f"Database has {ember_count} embers but no centroids. "
            "Run migrate.py to import centroids from the old system, "
            "or centroids will be random (semantically meaningless). "
            "Generating random centroids as fallback."
        )

    rng = np.random.default_rng(seed=42)
    centroids = rng.standard_normal((K_CELLS, DIMENSION)).astype(np.float32)
    norms = np.linalg.norm(centroids, axis=1, keepdims=True)
    centroids /= np.clip(norms, 1e-12, None)

    for i, vec in enumerate(centroids):
        await db.conn.execute(
            "INSERT INTO vec_centroids(cell_id, embedding) VALUES (?, ?)",
            (i, serialize_vector(vec)),
        )
    await db.conn.commit()


async def _ensure_init():
    """Lazy initialization — loop-bound, no partial publish, safe across contexts."""
    global _db, _embedder, _init_task, _init_loop

    # Fast path: fully initialized
    if _db is not None and _embedder is not None:
        return

    loop = asyncio.get_running_loop()

    if _init_task is None:
        # First caller creates the init task on this loop
        _init_loop = loop
        _init_task = loop.create_task(_do_init())
    elif _init_loop is not loop:
        raise RuntimeError("Ember server init is bound to a different event loop")

    try:
        db, embedder = await _init_task
    except Exception:
        # Clear sentinels so next attempt can retry
        _init_task = None
        _init_loop = None
        raise

    # Only publish after full success
    _db, _embedder = db, embedder


# ── Helpers ───────────────────────────────────────────────────────────────────

def _embed(text: str) -> tuple[np.ndarray, bytes]:
    """Embed text, return (vector_array, serialized_bytes)."""
    vec = _embedder.embed(text)
    return vec, serialize_vector(vec)


def _make_preview(text: str, max_chars: int = PREVIEW_CHARS) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0] + "..."


def _parse_tags(tags_str: str) -> list[str]:
    return [t.strip() for t in tags_str.split(",") if t.strip()] if tags_str else []


def _cell_diverse_select(scored: list, top_k: int) -> list:
    """Select top_k results with cell diversity (interleaving)."""
    if top_k < 4 or len(scored) <= top_k:
        return scored[:top_k]

    core = scored[:top_k - 1]
    core_cells = {item[0]["cell_id"] for item in core}
    top_score = scored[0][1] if scored else 0

    for item in scored[top_k - 1:]:
        ember, score, breakdown = item
        if ember["cell_id"] not in core_cells and score >= top_score * 0.5:
            return core + [item]

    return scored[:top_k]


async def _fetch_and_rerank(
    query_text: str, top_k: int = 5, fetch_multiplier: int = 10,
    active_project: str = "",
) -> list:
    """Fetch-and-Rerank pipeline using HESTIA scoring with project scoping.

    Returns list of (ember_dict, score, breakdown_dict).
    """
    now = datetime.now(timezone.utc)
    vec, embedding = _embed(query_text)

    total = await _db.ember_vector_count()
    fetch_k = min(top_k * fetch_multiplier, total) if total > 0 else 0
    if fetch_k == 0:
        return []

    results = await _db.search_embers(embedding, k=fetch_k)
    if not results:
        return []

    # Per-cell vitality from region_stats (computed offline by maintain.py)
    all_region_stats = await _db.get_all_region_stats()
    cell_vitalities = {cid: rs["vitality_score"] for cid, rs in all_region_stats.items()}
    v_max = max(cell_vitalities.values(), default=0.001)
    v_max = max(v_max, 0.001)

    scored = []
    for ember_id, dist in results:
        ember = await _db.get_ember(ember_id)
        if not ember:
            continue

        cos_sim = max(0.0, l2_to_cosine(dist))
        cell_vitality = cell_vitalities.get(ember["cell_id"], 0.0)
        score, breakdown = compute_hestia_score(
            cos_sim, ember["shadow_load"], cell_vitality, v_max,
            SHADOW_GAMMA, NOSTALGIA_ALPHA,
            utility=ember["utility_score"],
            utility_weight=UTILITY_WEIGHT,
        )

        # Project scoping
        project_factor = 1.0
        if active_project:
            project_tag = f"project:{active_project}"
            ember_tags = set(ember["tags"]) if ember["tags"] else set()
            if project_tag in ember_tags:
                project_factor = 1.0 + PROJECT_BOOST
                score = cos_sim * breakdown["shadow_factor"]
                breakdown["vitality_factor"] = 1.0
                breakdown["dormancy_protected"] = True
            elif {"foundation", "wonder", "meta"} & ember_tags:
                project_factor = 1.0
            else:
                project_factor = max(NOSTALGIA_ALPHA, 1.0 - PROJECT_PENALTY)
            score = score * project_factor
            breakdown["project_factor"] = project_factor

        scored.append((ember, score, breakdown))

    scored.sort(key=lambda x: x[1], reverse=True)
    selected = _cell_diverse_select(scored, top_k)

    # Update access stats for surfaced embers (atomic SQL increment)
    for ember, _, _ in selected:
        await _db.increment_access(ember["ember_id"])
        await _db.log_recall(_SESSION_ID, ember["ember_id"], "surfaced")

    return selected


async def _resolve_recency_strength(
    active_project: str, override: Optional[float]
) -> float:
    """Per-project recency strength. Explicit override wins; else a per-project
    config value (db key 'recency_strength:<project>'); else the global default.
    Lets fast projects keep heavy recency bias while slower ones relax it without
    a code change."""
    if override is not None and override >= 0.0:
        return override
    if active_project:
        try:
            v = await _db.get_config(f"recency_strength:{active_project}")
            if v is not None:
                return float(v)
        except Exception:
            pass
    return RECENCY_STRENGTH


async def _graph_rerank(
    query_text: str, top_k: int = 5, active_project: str = "",
    recency: Optional[float] = None, include_stale: bool = False,
) -> list:
    """Live recall via the ember-4 graph engine (dense+BM25+cohort+PPR+recency).

    Returns the same (ember_dict, score, breakdown) contract as
    _fetch_and_rerank so display code is unchanged. recency: None → per-project
    resolved default; 0 → reach back freely (deep_recall / wonder); >0 → explicit.
    """
    embedder = _get_nomic()
    strength = await _resolve_recency_strength(active_project, recency)
    # Over-fetch so project re-weighting can reorder before truncation.
    fetch_k = max(top_k * 3, 15)
    results, _trace = await graph_recall(
        _db, embedder, query_text, top_k=fetch_k, seed_k=20,
        recency_strength=strength, recency_tau_days=RECENCY_TAU_DAYS,
        include_stale=include_stale,
    )
    if not results:
        return []

    scored = []
    for r in results:
        ember = await _db.get_ember(r.ember_id)
        if not ember:
            continue
        score = r.score
        breakdown = dict(r.features)
        # Project scoping — parity with the legacy path's active_project contract.
        # (foundation/wonder/meta are project-penalty-exempt; this is separate
        # from the recency exemption, which is foundation/principle only.)
        project_factor = 1.0
        if active_project:
            project_tag = f"project:{active_project}"
            tags = set(ember["tags"]) if ember["tags"] else set()
            if project_tag in tags:
                project_factor = 1.0 + PROJECT_BOOST
            elif {"foundation", "wonder", "meta"} & tags:
                project_factor = 1.0
            else:
                project_factor = max(NOSTALGIA_ALPHA, 1.0 - PROJECT_PENALTY)
            score *= project_factor
        breakdown["project_factor"] = project_factor
        scored.append((ember, score, breakdown))

    scored.sort(key=lambda x: x[1], reverse=True)
    selected = scored[:top_k]

    for ember, _, _ in selected:
        await _db.increment_access(ember["ember_id"])
        await _db.log_recall(_SESSION_ID, ember["ember_id"], "surfaced")

    return selected


async def _archive_decayed_ember(ember: dict) -> None:
    """Archive a fully-shadowed ember to the document store."""
    tags = list(ember["tags"]) + ["decayed", "auto-archived"]
    project = ""
    for tag in ember["tags"]:
        if tag.startswith("project:"):
            project = tag[len("project:"):]
            break

    chunk_id = await _db.archive_next_chunk_id(
        project or "memory-archive", "general"
    )
    try:
        await _db.archive_store(
            chunk_id=chunk_id,
            content=ember["content"],
            summary=f"[decayed] {ember['name']}",
            tags=tags,
            project=project or "memory-archive",
            domain="decayed-embers",
        )
    except Exception:
        pass


async def _post_insert_v4(
    ember_id: str, name: str, content: str
) -> tuple[np.ndarray, np.ndarray]:
    """ember-4 post-insert: embed with Nomic prefixes, populate vec tables.

    Called after a new ember is INSERTed into the embers table. Generates
    both task-prefix vectors (clustering + retrieval), inserts them into
    the parallel sqlite-vec tables, and returns the vectors so callers
    can reuse them (e.g., to seed _relate_on_insert).

    Why two embeds: Nomic v1.5 has prefix-aware vector spaces. The
    clustering: vector lives near other "concept-adjacent" embers (used
    for the graph topology). The search_document: vector lives near
    things the query is *asking for* (used for retrieval). Same text,
    different fibers.
    """
    nomic = _get_nomic()
    text = f"{name}\n{content}"
    vec_cluster = nomic.embed_cluster(text)
    vec_retrieval = nomic.embed_document(text)
    cluster_blob = serialize_vector(vec_cluster)
    retrieval_blob = serialize_vector(vec_retrieval)

    await _db.conn.execute(
        "INSERT OR REPLACE INTO vec_embers_nomic_v15_clustering "
        "(ember_id, embedding) VALUES (?, ?)",
        (ember_id, cluster_blob),
    )
    await _db.conn.execute(
        "INSERT OR REPLACE INTO vec_embers_nomic_v15_retrieval "
        "(ember_id, embedding) VALUES (?, ?)",
        (ember_id, retrieval_blob),
    )
    await _db.conn.commit()
    return vec_cluster, vec_retrieval


async def _relate_on_insert(
    ember_id: str,
    vec_cluster: np.ndarray,
    k: int = 20,
    cohort_cosine_threshold: float = 0.755,
    contradict_cosine_threshold: float = 0.955,
) -> None:
    """ember-4 replacement for _shadow_on_insert.

    Design:
        * Same fiber + high cosine → REINFORCE via 'related' edge.
            No shadow_load. No auto-archive.
        * Different fiber + very high cosine → flag with
            'contradict_candidate' edge. Human (or agent) can decide
            later whether to invoke ember_contradict explicitly. No
            auto-archive — destructive ops are deliberate.
        * Mid-range cosine → 'related' edge regardless of fiber.

    Operates on Nomic clustering vectors (task-prefix space) so the
    "fiber" relation matches what the kNN+tag_cohort graph uses for
    PPR. Does NOT touch shadow_load, shadow_accum, or region_stats —
    those columns survive in the schema as frozen ember-3 artifacts
    but are dead from ember-4's perspective (see AUDIT_cell_id_*.md).
    """
    cluster_blob = serialize_vector(vec_cluster)

    # kNN in clustering space (NOT vec_embers / MiniLM)
    rows = await _db.conn.execute(
        "SELECT ember_id, distance FROM vec_embers_nomic_v15_clustering "
        "WHERE embedding MATCH ? AND k = ?",
        (cluster_blob, k + 1),  # +1 because we'll filter out self
    )
    neighbors = [(r["ember_id"], r["distance"]) for r in await rows.fetchall()]

    # tag_cohort co-membership tells us "same fiber"
    cur = await _db.conn.execute(
        """SELECT DISTINCT target_id AS x FROM edges
           WHERE source_id = ? AND edge_type = 'tag_cohort'
           UNION
           SELECT DISTINCT source_id AS x FROM edges
           WHERE target_id = ? AND edge_type = 'tag_cohort'""",
        (ember_id, ember_id),
    )
    cohort_members = {r["x"] for r in await cur.fetchall()}

    now_iso = datetime.now(timezone.utc).isoformat()
    edges_to_add: list[tuple[str, str, str, float]] = []

    for nid, dist in neighbors:
        if nid == ember_id:
            continue
        cos_sim = l2_to_cosine(dist)
        same_fiber = nid in cohort_members

        if cos_sim >= contradict_cosine_threshold and not same_fiber:
            # Very high cosine but different cohort → potential
            # paraphrase-or-contradiction. Flag, do not auto-resolve.
            edges_to_add.append((
                ember_id, nid, "contradict_candidate", float(cos_sim)
            ))
        elif cos_sim >= cohort_cosine_threshold:
            # Reinforce via 'related' (cohort or not). Cohort embers
            # already have a tag_cohort edge; this layer is dense-sim
            # confirmation. Same edge type either way — the graph
            # mixer in recall_v4 treats them uniformly.
            edges_to_add.append((
                ember_id, nid, "related", float(cos_sim)
            ))

    if edges_to_add:
        await _db.conn.executemany(
            "INSERT OR REPLACE INTO edges "
            "(source_id, target_id, edge_type, weight, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            [(s, t, et, w, now_iso) for s, t, et, w in edges_to_add],
        )
        await _db.conn.commit()


# Backward-compat insert hook. Existing call sites stay; under the hood
# this now does the ember-4 thing instead of writing shadow_load:
#
#   1. Embed name + content with Nomic clustering AND retrieval prefixes
#      → populate vec_embers_nomic_v15_clustering and ..._retrieval so
#        the new ember is findable in the v4 pipeline immediately.
#   2. Add 'related' / 'contradict_candidate' edges via _relate_on_insert.
#   3. NEVER write shadow_load. NEVER auto-archive. NEVER touch region_stats.
#
# Errors are swallowed (logged to stderr) so a failure in the Nomic embed
# or new edges can't break the underlying ember create. Principle:
# don't ablate a signal until its replacement is wired — and conversely,
# don't make the new path so fragile it breaks the create flow.
async def _shadow_on_insert(ember_id: str, ember_dict: dict, vec: np.ndarray) -> None:
    """ember-4 post-insert hook (kept under the legacy name for caller
    compatibility). Routes through the new v4 helpers; no shadow writes.
    """
    name = ember_dict.get("name", "") if ember_dict else ""
    content = ember_dict.get("content", "") if ember_dict else ""
    try:
        vec_cluster, _vec_retrieval = await _post_insert_v4(
            ember_id, name, content
        )
        await _relate_on_insert(ember_id, vec_cluster)
    except Exception as e:
        import sys as _sys
        print(
            f"[ember-4] post-insert hook failed for {ember_id[:8]}: {e}",
            file=_sys.stderr,
        )


# ═══════════════════════════════════════════════════════════════════════════════
#  CORE MEMORY OPERATIONS
# ═══════════════════════════════════════════════════════════════════════════════


@mcp.tool()
@_gc_tool
async def ember_store(
    name: str,
    content: str,
    tags: str = "",
    importance: str = "context",
    source_path: str = "",
) -> str:
    """Store a memory ember with importance level.

    Args:
        name: Short descriptive name
        content: The content to remember
        tags: Comma-separated tags
        importance: One of: fact, decision, preference, context, learning
        source_path: Optional source file path
    """
    await _ensure_init()

    tag_list = _parse_tags(tags)
    if importance not in ("fact", "decision", "preference", "context", "learning"):
        importance = "context"

    vec, embedding = _embed(content)
    cell_id = await _db.assign_cell(embedding)
    eid = str(uuid.uuid4())

    await _db.save_ember(
        ember_id=eid, name=name, content=content, tags=tag_list,
        cell_id=cell_id, importance=importance, source="manual",
        embedding=embedding, source_path=source_path,
    )

    ember_dict = await _db.get_ember(eid)
    await _shadow_on_insert(eid, ember_dict, vec)
    # Cognitive engagement: deliberate user/agent create. Contributes to importance.
    await _db.log_update(
        ember_id=eid, event_type="edit", origin="agent",
        content_changed=True, session_id=_SESSION_ID,
        details=f"ember_store: {importance}",
    )

    half_life = DECAY_HALF_LIVES.get(importance, 30.0)
    return (
        f"Stored ember '{name}' (ID: {eid}) in Cell {cell_id}. "
        f"Importance: {importance} (half-life: {int(half_life)}d)"
    )


@mcp.tool()
@_gc_tool
async def ember_recall(
    query: str, top_k: int = 5, active_project: str = "",
    recency: float = -1.0, include_stale: bool = False,
) -> str:
    """Retrieve memory embers semantically via the ember-4 graph engine
    (dense + BM25 + tag-cohort + recency re-rank).

    Args:
        query: What to search for
        top_k: Number of results (default 5)
        active_project: Boosts embers tagged "project:<name>"
        recency: Staleness demotion strength. -1 (default) = per-project resolved
            default (favors recent work — last session is usually most relevant).
            0 = reach back freely (old-but-general principles, historical lookups;
            deep_recall does this). >0 = explicit override. foundation/principle-
            tagged embers are exempt regardless.
        include_stale: Historical recall. True also searches memories that
            ember_contradict or synthesis superseded (shown [STALE]); by
            default they are left out.
    """
    await _ensure_init()
    now = datetime.now(timezone.utc)
    scored = await _graph_rerank(
        query, top_k=top_k, active_project=active_project,
        recency=(None if recency < 0 else recency),
        include_stale=include_stale,
    )

    if not scored:
        return "No embers found."

    lines = []
    for ember, score, breakdown in scored:
        age_days = breakdown.get("age_days")
        if age_days is None:
            created = datetime.fromisoformat(ember["created_at"].replace("Z", "+00:00"))
            age_days = (now - created).total_seconds() / 86400.0
        freshness = "fresh" if age_days < 7 else f"{int(age_days)}d ago"
        exempt_mark = " [foundation]" if breakdown.get("recency_exempt") else ""
        stale_mark = " [STALE]" if ember["is_stale"] else ""
        shadow_mark = f" [shadow:{ember['shadow_load']:.1f}]" if ember["shadow_load"] > 0.1 else ""
        source_note = f"\n  source: {ember['source_path']}" if ember.get("source_path") else ""
        preview = _make_preview(ember["content"])
        lines.append(
            f"\U0001f525 {ember['name']} [id: {ember['ember_id']}] "
            f"(score: {score:.3f}, {freshness}{exempt_mark}{stale_mark}{shadow_mark})\n"
            f"  {preview}{source_note}"
        )

    lines.append("\n\u2192 Use ember_read(id) for full content of any memory.")
    return "\n\n".join(lines)


@mcp.tool()
@_gc_tool
async def ember_learn(conversation_context: str, source_path: str = "") -> str:
    """Auto-capture key information from conversation. Call silently after every
    substantive user message.

    Args:
        conversation_context: "TYPE: description" where TYPE is fact/decision/preference/learning
        source_path: Optional source file path
    """
    await _ensure_init()

    importance = "context"
    content = conversation_context

    for itype in ("fact", "decision", "preference", "learning", "context"):
        if conversation_context.lower().startswith(f"{itype}:"):
            importance = itype
            content = conversation_context[len(itype) + 1:].strip()
            break

    if not content.strip():
        return "Nothing to capture: empty content."

    name = content[:60].strip()
    if len(content) > 60:
        name = name.rsplit(" ", 1)[0] + "..."

    vec, embedding = _embed(content)
    cell_id = await _db.assign_cell(embedding)

    # Near-duplicate / evolve check. Similarity nominates; it doesn't
    # authorize. Changed numbers, negations and qualifiers all sit close in
    # embedding space (MiniLM puts "28.5 C" vs "26.5 C" above cosine 0.995),
    # and containment isn't entailment either ("It is false that X" contains
    # X). So a near match reinforces only when the text is the same up to
    # whitespace; case and punctuation can matter (identifiers, code).
    # Anything else is stored as its own ember with an 'evolves' edge to the
    # one it resembles; resolve with ember_contradict.
    EVOLVE_THRESHOLD = 0.3      # L2 distance: cosine > 0.955

    def _norm(s: str) -> str:
        return " ".join(s.split())

    evolves_from = None  # (ember, cosine) of the closest active near-match
    existing = await _db.search_embers(embedding, k=3)
    for ex_id, dist in existing:
        if dist < EVOLVE_THRESHOLD:
            ex_ember = await _db.get_ember(ex_id)
            if ex_ember and not ex_ember["is_stale"]:
                if _norm(content) == _norm(ex_ember["content"]):
                    await _db.increment_access(ex_id)
                    return f"Reinforced existing ember: '{ex_ember['name']}'"
                if evolves_from is None:
                    evolves_from = (ex_ember, l2_to_cosine(dist))

    tags = ["auto-captured", importance]
    eid = str(uuid.uuid4())
    await _db.save_ember(
        ember_id=eid, name=name, content=content, tags=tags,
        cell_id=cell_id, importance=importance, source="auto",
        embedding=embedding, source_path=source_path,
    )

    ember_dict = await _db.get_ember(eid)
    await _shadow_on_insert(eid, ember_dict, vec)
    # Auto-captured (passive context extraction). Logged for audit but does
    # NOT contribute to the importance signal (origin='auto_system').
    await _db.log_update(
        ember_id=eid, event_type="edit", origin="auto_system",
        content_changed=True, session_id=_SESSION_ID,
        details="ember_learn: auto-captured from conversation context",
    )

    if evolves_from:
        ex_ember, cos_sim = evolves_from
        await _db.save_edge(eid, ex_ember["ember_id"], "evolves", float(cos_sim))
        return (
            f"Captured {importance}: '{name}' (evolves '{ex_ember['name']}'"
            f" [id: {ex_ember['ember_id']}], similarity {cos_sim:.3f}; both kept)"
        )
    return f"Captured {importance}: '{name}'"


@mcp.tool()
@_gc_tool
async def ember_wonder(question: str, tags: str = "", context: str = "") -> str:
    """Store an open question or hypothesis as a wonder-ember.

    Args:
        question: The open question or hypothesis
        tags: Comma-separated tags beyond automatic 'wonder'
        context: Optional context about what prompted this wonder
    """
    await _ensure_init()

    content = question.strip()
    if context:
        content += f"\n\nContext: {context.strip()}"

    tag_list = ["wonder"]
    tag_list.extend(_parse_tags(tags))

    name = f"[wonder] {question[:55].strip()}"
    if len(question) > 55:
        name = name.rsplit(" ", 1)[0] + "..."

    vec, embedding = _embed(content)
    cell_id = await _db.assign_cell(embedding)

    # Near-duplicate check (cosine > 0.875 = L2 < 0.5)
    existing = await _db.search_embers(embedding, k=5)
    for ex_id, dist in existing:
        cos = l2_to_cosine(dist)
        if cos > 0.875:
            ex_ember = await _db.get_ember(ex_id)
            if ex_ember and not ex_ember["is_stale"]:
                await _db.increment_access(ex_id)
                return f"Reinforced existing wonder: '{ex_ember['name']}' (cosine: {cos:.3f})"

    eid = str(uuid.uuid4())
    await _db.save_ember(
        ember_id=eid, name=name, content=content, tags=tag_list,
        cell_id=cell_id, importance="learning", source="wonder",
        embedding=embedding,
    )

    ember_dict = await _db.get_ember(eid)
    await _shadow_on_insert(eid, ember_dict, vec)
    # Agent reflective action: wonder is a cognitive engagement event.
    await _db.log_update(
        ember_id=eid, event_type="edit", origin="agent",
        content_changed=True, session_id=_SESSION_ID,
        details="ember_wonder",
    )
    return f"Wonder stored: '{name}'"


@mcp.tool()
@_gc_tool
async def ember_contradict(ember_id: str, new_content: str, reason: str = "") -> str:
    """Mark an existing memory as stale and store an updated version.

    Args:
        ember_id: The ID of the ember to mark stale
        new_content: The corrected/updated information
        reason: Why the old information is stale
    """
    await _ensure_init()

    old_ember = await _db.get_ember(ember_id)
    if not old_ember:
        return f"Ember {ember_id} not found."

    await _archive_decayed_ember(old_ember)

    vec, embedding = _embed(new_content)
    cell_id = await _db.assign_cell(embedding)
    new_eid = str(uuid.uuid4())

    # Store new version FIRST (so FK reference from old->new is valid)
    await _db.save_ember(
        ember_id=new_eid, name=old_ember["name"], content=new_content,
        tags=old_ember["tags"], cell_id=cell_id,
        importance=old_ember["importance"], source="manual",
        embedding=embedding, supersedes_id=ember_id,
    )

    # Mark old as stale (now safe — new_eid exists for shadowed_by FK)
    await _db.update_ember(
        ember_id, content_changed=False,
        is_stale=True,
        stale_reason=reason or "Superseded by newer information",
        shadow_load=1.0,
        shadowed_by=new_eid,
        shadow_updated_at=datetime.now(timezone.utc).isoformat(),
    )

    await _db.save_edge(ember_id, new_eid, "supersedes", 1.0)

    # Clean up edges from the stale ember — prevents wrong dream bridges
    # and KG edges from distorting future retrievals (stale edges
    # accumulate in the graph if not cleaned on contradict)
    old_edges = await _db.get_edges(ember_id)
    for edge in old_edges:
        # Keep the supersedes edge we just created; remove everything else
        if edge["edge_type"] == "supersedes" and edge["target_id"] == new_eid:
            continue
        # Transfer 'related' edges to the new ember so lineage is preserved
        if edge["edge_type"] == "related":
            other_id = edge["target_id"] if edge["source_id"] == ember_id else edge["source_id"]
            await _db.save_edge(new_eid, other_id, "related", edge["weight"])

    # Remove all old edges except the supersedes link
    await _db.conn.execute(
        "DELETE FROM edges WHERE (source_id = ? OR target_id = ?) AND NOT (source_id = ? AND target_id = ? AND edge_type = 'supersedes')",
        (ember_id, ember_id, ember_id, new_eid),
    )
    await _db.conn.commit()

    new_dict = await _db.get_ember(new_eid)
    await _shadow_on_insert(new_eid, new_dict, vec)
    # Contradict: log both sides for ember-4's update-activity signal.
    # The OLD ember's last engagement event was the contradiction
    # itself (worth recording); the NEW ember is a fresh substantive
    # write. Both count toward importance under the contribution
    # filter (origin='agent', meaningful events).
    await _db.log_update(
        ember_id=ember_id, event_type="contradict_old", origin="agent",
        content_changed=False, session_id=_SESSION_ID,
        details=f"contradicted by {new_eid}: {reason[:80]}",
    )
    await _db.log_update(
        ember_id=new_eid, event_type="contradict_new", origin="agent",
        content_changed=True, session_id=_SESSION_ID,
        details=f"contradicts {ember_id}: {reason[:80]}",
    )

    return (
        f"Updated memory: '{old_ember['name']}'. "
        f"Old version fully shadowed. New version: {new_eid}"
    )


@mcp.tool()
@_gc_tool
async def ember_read(ember_id: str) -> str:
    """Read the full content of a specific ember by ID.

    Args:
        ember_id: The ID shown in search results as [id: ...]
    """
    await _ensure_init()
    ember = await _db.get_ember(ember_id)
    if not ember:
        return f"Ember {ember_id} not found."

    await _db.increment_access(ember_id)
    await _db.log_recall(_SESSION_ID, ember_id, "read")

    tags_str = ", ".join(ember["tags"]) if ember["tags"] else "none"
    source = f"\nSource: {ember['source_path']}" if ember.get("source_path") else ""
    return (
        f"\U0001f525 {ember['name']} ({ember['importance']})\n\n"
        f"{ember['content']}\n\n"
        f"Tags: {tags_str}{source}"
    )


@mcp.tool()
@_gc_tool
async def ember_list(tag: str = "", limit: int = 20, offset: int = 0) -> str:
    """List stored memory embers with pagination. Returns metadata only.

    Args:
        tag: Optional tag filter
        limit: Max results per page (default 20)
        offset: Skip this many results
    """
    await _ensure_init()
    all_embers = await _db.list_embers(tag=tag)

    if not all_embers:
        return "No embers stored." if not tag else f"No embers with tag '{tag}'."

    total = len(all_embers)
    page = all_embers[offset:offset + limit]
    now = datetime.now(timezone.utc)

    lines = []
    for e in page:
        created = datetime.fromisoformat(e["created_at"].replace("Z", "+00:00"))
        age_days = (now - created).total_seconds() / 86400.0
        freshness = "today" if age_days < 1 else f"{int(age_days)}d ago"
        stale = " [STALE]" if e["is_stale"] else ""
        shadow = f" [shadow:{e['shadow_load']:.1f}]" if e["shadow_load"] > 0.1 else ""
        lines.append(
            f"\u2022 {e['name']} ({e['importance']}) [{freshness}]{stale}{shadow} "
            f"[id: {e['ember_id']}]"
        )

    start = offset + 1
    end = min(offset + limit, total)
    header = f"Showing {start}-{end} of {total} embers"
    if end < total:
        header += f" (use offset={end} for more)"

    return header + ":\n" + "\n".join(lines)


@mcp.tool()
@_gc_tool
async def ember_delete(ember_id: str) -> str:
    """Delete a memory ember by its ID."""
    await _ensure_init()
    deleted = await _db.delete_ember(ember_id)
    if not deleted:
        return f"Ember {ember_id} not found."
    return f"Ember {ember_id} deleted."


@mcp.tool()
@_gc_tool
async def ember_auto(conversation_context: str, active_project: str = "") -> str:
    """Automatically retrieve relevant memory embers based on conversation context.
    Call at the start of every conversation.

    Args:
        conversation_context: Summary of the conversation topic
        active_project: Boosts embers tagged "project:<name>"
    """
    await _ensure_init()
    # Session-start load: use the graph engine with default (per-project) recency
    # so the last session's work surfaces first.
    scored = await _graph_rerank(
        conversation_context, top_k=5, active_project=active_project
    )

    if not scored:
        return ""

    lines = []
    for ember, score, breakdown in scored:
        stale_note = " (outdated)" if ember["is_stale"] else ""
        preview = _make_preview(ember["content"])
        lines.append(
            f"\U0001f525 {ember['name']} [id: {ember['ember_id']}]{stale_note}: {preview}"
        )

    # Session continuity: recent sessions
    try:
        session_types = ("session", "session-export", "compact-export")

        def _dedup_sorted(chunks, lim):
            seen = set()
            out = []
            for c in sorted(chunks, key=lambda c: c.get("created_at", ""), reverse=True):
                if c["chunk_id"] not in seen:
                    seen.add(c["chunk_id"])
                    out.append(c)
                if len(out) >= lim:
                    break
            return out

        def _fmt_chunk(c):
            ts = c.get("created_at", "")[:16].replace("T", " ")
            title = c.get("summary", c["chunk_id"])
            agent = c.get("from_agent", "")
            agent_tag = f" [{agent}]" if agent else ""
            return f"  \u2022 [{ts}]{agent_tag} {title} ({c.get('chunk_type', '')})"

        project_chunks = []
        all_chunks = []
        for stype in session_types:
            if active_project:
                project_chunks.extend(
                    await _db.archive_list(chunk_type=stype, project=active_project, limit=5)
                )
            all_chunks.extend(await _db.archive_list(chunk_type=stype, limit=5))

        if active_project and project_chunks:
            in_project = _dedup_sorted(project_chunks, 2)
            in_project_ids = {c["chunk_id"] for c in in_project}
            others = _dedup_sorted(
                [c for c in all_chunks if c["chunk_id"] not in in_project_ids], 2
            )
            lines.append(f"\n\U0001f4cb Recent sessions ({active_project}):")
            for c in in_project:
                lines.append(_fmt_chunk(c))
            if others:
                lines.append("  Other recent:")
                for c in others:
                    lines.append(_fmt_chunk(c))
        else:
            unique = _dedup_sorted(all_chunks, 3)
            if unique:
                lines.append("\n\U0001f4cb Recent sessions:")
                for c in unique:
                    lines.append(_fmt_chunk(c))

        lines.append("  \u2192 Use archive_read(id) for session details.")
    except Exception:
        pass

    lines.append("\n\u2192 Use ember_read(id) for full content of any memory.")
    return "\n\n".join(lines)


@mcp.tool()
@_gc_tool
async def ember_update_tags(
    ember_id: str, add_tags: str = "", remove_tags: str = "", set_tags: str = "",
) -> str:
    """Update tags on an existing ember without changing content.

    Args:
        ember_id: UUID of the ember
        add_tags: Comma-separated tags to add
        remove_tags: Comma-separated tags to remove
        set_tags: Replace ALL existing tags with this list
    """
    await _ensure_init()
    ember = await _db.get_ember(ember_id)
    if not ember:
        return f"Ember {ember_id} not found."

    old_tags = list(ember["tags"])

    if set_tags:
        new_tags = _parse_tags(set_tags)
    else:
        current = set(ember["tags"])
        if add_tags:
            current.update(_parse_tags(add_tags))
        if remove_tags:
            current.difference_update(_parse_tags(remove_tags))
        new_tags = sorted(current)

    await _db.update_ember(ember_id, content_changed=True, tags=new_tags)
    # Tag-only update: content body unchanged. Small-weight contribution
    # in the importance filter (only when origin is user/agent).
    await _db.log_update(
        ember_id=ember_id, event_type="tag_update", origin="agent",
        content_changed=False, session_id=_SESSION_ID,
        details=f"tags: {len(old_tags)}→{len(new_tags)}",
    )
    return (
        f"Updated tags on '{ember['name']}' [{ember_id}].\n"
        f"Before: {old_tags}\nAfter:  {new_tags}"
    )


@mcp.tool()
@_gc_tool
async def ember_save_session(
    summary: str, decisions: str = "", next_steps: str = "", source_path: str = "",
) -> str:
    """Save key takeaways from the current session.

    Args:
        summary: Brief summary of the session's key work
        decisions: Decisions made during the session
        next_steps: Open items and next actions
        source_path: Optional path to handoff file
    """
    await _ensure_init()
    saved = []

    async def _store(name, content, extra_tags, importance):
        vec, embedding = _embed(content)
        cell_id = await _db.assign_cell(embedding)
        eid = str(uuid.uuid4())
        await _db.save_ember(
            ember_id=eid, name=name, content=content,
            tags=["session"] + extra_tags, cell_id=cell_id,
            importance=importance, source="session",
            embedding=embedding, source_path=source_path,
        )
        ember_dict = await _db.get_ember(eid)
        await _shadow_on_insert(eid, ember_dict, vec)
        await _db.log_update(
            ember_id=eid, event_type="edit", origin="agent",
            content_changed=True, session_id=_SESSION_ID,
            details="ember_save_session",
        )

    if summary:
        await _store("Session Summary", summary, ["summary"], "context")
        saved.append("summary")
    if decisions:
        await _store("Session Decisions", decisions, ["decisions"], "decision")
        saved.append("decisions")
    if next_steps:
        await _store("Next Steps", next_steps, ["next_steps"], "learning")
        saved.append("next steps")

    return f"Session saved: {', '.join(saved)}. Available in your next conversation."


# ═══════════════════════════════════════════════════════════════════════════════
#  SHADOW-DECAY ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════


@mcp.tool()
@_gc_tool
async def ember_inspect(cell_id: int = -1) -> str:
    """Inspect Voronoi cell health. Shows ember distribution and conflict density."""
    await _ensure_init()

    if cell_id >= 0:
        stats = (await _db.get_all_region_stats()).get(cell_id)
        if not stats:
            return f"Cell {cell_id}: no data"
        return (
            f"Cell {cell_id}: vitality={stats['vitality_score']:.3f}, "
            f"conflict_density={stats['shadow_accum']:.3f}, "
            f"last_updated={stats['last_updated']}"
        )

    all_embers = await _db.list_embers()
    cell_counts: dict[int, int] = {}
    for e in all_embers:
        cell_counts[e["cell_id"]] = cell_counts.get(e["cell_id"], 0) + 1

    total = len(all_embers)
    all_stats = await _db.get_all_region_stats()
    lines = [f"Voronoi Cell Map ({K_CELLS} cells, {total} embers):"]
    for i in range(K_CELLS):
        count = cell_counts.get(i, 0)
        bar = "\u2588" * min(count, 20)
        stats = all_stats.get(i)
        conflict = f" conflict:{stats['shadow_accum']:.2f}" if stats else ""
        lines.append(f"  Cell {i:2d}: {bar} {count}{conflict}")

    return "\n".join(lines)


@mcp.tool()
@_gc_tool
async def ember_recompute_centroids() -> str:
    """Recompute Voronoi centroids via k-means on actual ember vectors.

    Run this after migration, after significant knowledge growth, or when
    ember_inspect shows uneven cell distribution. Replaces random/stale
    centroids with data-derived ones and reassigns all embers to new cells.
    """
    await _ensure_init()
    result = await _db.recompute_centroids(k_cells=K_CELLS)

    if result["status"] == "skipped":
        return result["message"]

    sizes = result["cell_sizes"]
    lines = [
        f"Centroids recomputed via k-means ({result['iterations']} iterations)",
        f"  {result['embers']} embers across {result['cells']} cells",
        "",
        "Cell distribution:",
    ]
    for i in range(result["cells"]):
        count = sizes.get(i, 0)
        bar = "\u2588" * min(count, 30)
        lines.append(f"  Cell {i:2d}: {bar} {count}")

    return "\n".join(lines)


@mcp.tool()
@_gc_tool
async def ember_drift_check() -> str:
    """Analyze knowledge region health using Shadow-Decay conflict density."""
    await _ensure_init()

    drifting = []
    silent = []
    healthy = 0
    all_stats = await _db.get_all_region_stats()

    for cell_id in range(K_CELLS):
        stats = all_stats.get(cell_id)
        if not stats:
            silent.append(f"  Cell {cell_id}: no data (uninitialized)")
            continue
        if stats["shadow_accum"] > 0.3:
            drifting.append(
                f"  Cell {cell_id}: conflict_density={stats['shadow_accum']:.3f} (HIGH)"
            )
        elif stats["vitality_score"] < 0.01:
            silent.append(
                f"  Cell {cell_id}: vitality={stats['vitality_score']:.4f} (SILENT)"
            )
        else:
            healthy += 1

    lines = [
        f"Knowledge Region Health ({K_CELLS} cells)",
        "=" * 50,
        f"Healthy: {healthy}  |  Drifting: {len(drifting)}  |  Silent: {len(silent)}",
        "",
    ]
    if drifting:
        lines.append("Drifting regions (high conflict density):")
        lines.extend(drifting)
        lines.append("")
    if silent:
        lines.append("Silent regions (low vitality):")
        lines.extend(silent)
        lines.append("")
    if not drifting and not silent:
        lines.append("All regions healthy.")

    return "\n".join(lines)


@mcp.tool()
@_gc_tool
async def ember_graph_search(
    query: str, depth: int = 2, top_k: int = 5, active_project: str = "",
) -> str:
    """Vector search -> entry node -> BFS via knowledge graph -> correlated context.

    Args:
        query: What to search for
        depth: How many hops (default 2)
        top_k: Max results (default 5)
        active_project: Project scoping for entry point
    """
    await _ensure_init()

    entry_results = await _graph_rerank(query, top_k=1, active_project=active_project)
    if not entry_results:
        return "No embers found."

    entry_ember, entry_score, _ = entry_results[0]
    entry_id = entry_ember["ember_id"]

    connected_ids = await _db.traverse_kg(entry_id, depth=depth)

    graph_embers = []
    for eid in connected_ids:
        ember = await _db.get_ember(eid)
        if ember:
            await _db.increment_access(eid)
            await _db.log_recall(_SESSION_ID, eid, "graph")
            graph_embers.append(ember)

    lines = [
        f"Graph search for: '{query}'",
        f"Entry: \U0001f525 {entry_ember['name']} [id: {entry_id}] (score: {entry_score:.2f})",
        f"  {_make_preview(entry_ember['content'])}",
        "",
        f"Connected memories ({len(graph_embers)} found via {depth}-hop traversal):",
    ]

    for ember in graph_embers[:top_k]:
        shadow_info = f" [shadow:{ember['shadow_load']:.1f}]" if ember["shadow_load"] > 0.1 else ""
        stale_info = " [STALE]" if ember["is_stale"] else ""
        preview = _make_preview(ember["content"])
        lines.append(
            f"  \U0001f525 {ember['name']} [id: {ember['ember_id']}]{stale_info}{shadow_info}: {preview}"
        )

    if not graph_embers:
        lines.append("  No graph-connected memories found.")
    lines.append("\n\u2192 Use ember_read(id) for full content of any memory.")
    return "\n".join(lines)


@mcp.tool()
@_gc_tool
async def ember_health() -> str:
    """Compute hallucination risk across all embers with trend."""
    await _ensure_init()

    embers = await _db.list_embers()
    if not embers:
        return "No embers in storage."

    shadow_loads = [e["shadow_load"] for e in embers]
    stale_flags = [e["is_stale"] for e in embers]

    all_stats = await _db.get_all_region_stats()
    vitalities = [
        all_stats.get(i, {}).get("vitality_score", 0.0) for i in range(K_CELLS)
    ]

    risk_data = compute_hallucination_risk(shadow_loads, stale_flags, vitalities)
    await _db.log_metric("hallucination_risk", risk_data["risk_score"], risk_data)

    history = await _db.get_metric_history("hallucination_risk", limit=5)
    trend_values = [f"{h['value']:.3f}" for h in history]
    trend_str = " \u2192 ".join(trend_values) if trend_values else "no history"

    return (
        f"Health: risk={risk_data['risk_score']:.3f} (0=ok, 1=bad) | trend: {trend_str}\n"
        f"Total: {risk_data['total']} | Shadowed(\u03a6>0.5): {risk_data['shadowed_count']} | "
        f"Stale: {risk_data['stale_count']} | Silent: {risk_data['silent_topics']} | "
        f"Avg \u03a6: {risk_data['avg_shadow_load']:.3f}"
    )


@mcp.tool()
@_gc_tool
async def ember_explain(ember_id: str) -> str:
    """Return HESTIA score breakdown for a specific ember.

    Args:
        ember_id: The ID of the ember to explain
    """
    await _ensure_init()

    ember = await _db.get_ember(ember_id)
    if not ember:
        return f"Ember {ember_id} not found."

    all_stats = await _db.get_all_region_stats()
    cell_stats = all_stats.get(ember["cell_id"], {})
    vitality = cell_stats.get("vitality_score", 0.0)
    v_max = max((s.get("vitality_score", 0.0) for s in all_stats.values()), default=0.001)
    v_max = max(v_max, 0.001)

    score, breakdown = compute_hestia_score(
        1.0, ember["shadow_load"], vitality, v_max,
        SHADOW_GAMMA, NOSTALGIA_ALPHA,
    )

    edges = await _db.get_edges(ember_id)
    edge_lines = []
    for e in edges:
        other = e["target_id"] if e["source_id"] == ember_id else e["source_id"]
        edge_lines.append(f"  {e['edge_type']}: {other[:8]}... (weight={e['weight']:.2f})")

    lines = [
        f"Ember Explanation: {ember['name']}",
        f"ID: {ember_id}",
        "=" * 50,
        "",
        "HESTIA Factors (at perfect query match):",
        f"  Final Score: {score:.4f}",
        f"  Cosine Sim: 1.0 (self)",
        f"  Shadow Factor: {breakdown['shadow_factor']:.4f}  (shadow_load={ember['shadow_load']:.3f})",
        f"  Vitality Factor: {breakdown['vitality_factor']:.4f}  (vitality={vitality:.3f})",
        "",
        f"Shadow Load: {ember['shadow_load']:.3f}",
        f"Shadowed By: {ember.get('shadowed_by') or 'None'}",
        f"Stale: {ember['is_stale']} ({ember.get('stale_reason') or 'N/A'})",
        "",
        f"Edges ({len(edges)}):",
    ]
    lines.extend(edge_lines if edge_lines else ["  None"])
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
#  IMPORT
# ═══════════════════════════════════════════════════════════════════════════════


@mcp.tool()
@_gc_tool
async def ember_import_markdown(
    markdown_text: str, tags: str = "imported", project: str = "",
) -> str:
    """Import a Claude web conversation export as memory embers.

    Args:
        markdown_text: Full markdown from a Claude conversation export
        tags: Comma-separated tags (default: "imported")
        project: Optional project name
    """
    await _ensure_init()

    tag_list = _parse_tags(tags) + ["import"]
    if project:
        tag_list.append(f"project:{project}")

    turns = []
    current_role = None
    current_content = []

    for line in markdown_text.splitlines():
        if re.match(r"^##\s+Human(?:\s+\([^)]*\))?:\s*$", line):
            if current_role and current_content:
                turns.append((current_role, "\n".join(current_content).strip()))
            current_role = "human"
            current_content = []
        elif re.match(r"^##\s+Claude:\s*$", line):
            if current_role and current_content:
                turns.append((current_role, "\n".join(current_content).strip()))
            current_role = "claude"
            current_content = []
        elif re.match(r"^---+\s*$", line) or line.startswith("# "):
            pass
        elif current_role is not None:
            current_content.append(line)

    if current_role and current_content:
        turns.append((current_role, "\n".join(current_content).strip()))

    if not turns:
        return "No conversation turns found."

    stored = 0
    skipped = 0
    for role, content in turns:
        if len(content) < 80:
            skipped += 1
            continue

        name_preview = content[:50].strip().replace("\n", " ")
        if len(content) > 50:
            name_preview = name_preview.rsplit(" ", 1)[0] + "..."
        name = f"[{role.title()}] {name_preview}"

        vec, embedding = _embed(content)
        cell_id = await _db.assign_cell(embedding)
        eid = str(uuid.uuid4())
        await _db.save_ember(
            ember_id=eid, name=name, content=content,
            tags=tag_list + [f"role:{role}"], cell_id=cell_id,
            importance="context", source="auto", embedding=embedding,
        )
        ember_dict = await _db.get_ember(eid)
        await _shadow_on_insert(eid, ember_dict, vec)
        # Import is auto-system (bulk markdown ingest); doesn't contribute
        # to engagement signal but is logged for audit.
        await _db.log_update(
            ember_id=eid, event_type="edit", origin="auto_system",
            content_changed=True, session_id=_SESSION_ID,
            details="ember_import_markdown",
        )
        stored += 1

    return f"Imported {stored} turns as memories. {skipped} skipped (too short). Tags: {tag_list}"


# ═══════════════════════════════════════════════════════════════════════════════
#  ARCHIVE (Document Store)
# ═══════════════════════════════════════════════════════════════════════════════


@mcp.tool()
@_gc_tool
async def archive_store(
    content: str,
    summary: str = "",
    tags: str = "",
    project: str = "general",
    domain: str = "",
    chunk_type: str = "general",
    from_agent: str = "",
    for_agent: str = "",
    category: str = "",
    reply_to: str = "",
) -> str:
    """Save a document to the searchable archive.

    Args:
        content: Full document text
        summary: One-line description (auto-generated if empty)
        tags: Comma-separated tags
        project: Project name (default: "general")
        domain: Aspect/subdomain
        chunk_type: session, debug, snapshot, compact-export, reference, general
        from_agent: Which agent created this
        for_agent: Intended recipient agent
        category: Freeform category label
        reply_to: Chunk ID this responds to
    """
    await _ensure_init()
    tag_list = _parse_tags(tags)
    chunk_id = await _db.archive_next_chunk_id(project, chunk_type)

    # Embed summary + tags for semantic search on what chunks are "about"
    embed_text = (summary or content[:200]) + " " + " ".join(tag_list)
    _, embedding = _embed(embed_text)

    result = await _db.archive_store(
        chunk_id=chunk_id, content=content, summary=summary, tags=tag_list,
        project=project, domain=domain, chunk_type=chunk_type,
        from_agent=from_agent, for_agent=for_agent, category=category,
        reply_to=reply_to, embedding=embedding,
    )

    if result["status"] == "duplicate":
        return f"Already archived (ID: {result['existing_id']}): {result['summary']}"
    if result["status"] == "error":
        return f"Error: {result['message']}"
    return f"Archived as {result['id']} ({chunk_type}): {result['summary']}"


@mcp.tool()
@_gc_tool
async def archive_search(
    query: str,
    project: str = "",
    domain: str = "",
    chunk_type: str = "",
    date_from: str = "",
    date_to: str = "",
    limit: int = 5,
) -> str:
    """Search archived documents by keyword (FTS5/BM25).

    Args:
        query: Keywords to search for
        project: Filter by project
        domain: Filter by domain
        chunk_type: Filter by type
        date_from: YYYY-MM-DD
        date_to: YYYY-MM-DD
        limit: Max results (default 5)
    """
    await _ensure_init()
    results = await _db.archive_search_fts(
        query, limit=limit, project=project, domain=domain,
        chunk_type=chunk_type, date_from=date_from, date_to=date_to,
    )

    if not results:
        return "No matching documents found."

    lines = [f"Archive search: '{query}' \u2014 {len(results)} result(s)"]
    for r in results:
        loc = f" [{r['project']}/{r['domain']}]" if r.get("domain") else f" [{r['project']}]"
        ct = f" ({r['chunk_type']})" if r.get("chunk_type") and r["chunk_type"] != "general" else ""
        score = r.get("score", 0)
        lines.append(f"\n\u2022 {r['chunk_id']}{loc}{ct} (score: {score:.2f})")
        lines.append(f"  {r['summary']}")
    lines.append("\n\u2192 Use archive_read(id) for full document.")
    return "\n".join(lines)


@mcp.tool()
@_gc_tool
async def archive_grep(
    pattern: str,
    project: str = "",
    domain: str = "",
    chunk_type: str = "",
    chunk_id: str = "",
    date_from: str = "",
    date_to: str = "",
    fuzzy: bool = False,
    fuzzy_threshold: int = 80,
    limit: int = 20,
) -> str:
    """Search archive contents with regex or fuzzy matching.

    Args:
        pattern: Regex pattern (or text if fuzzy=True)
        project: Filter by project
        domain: Filter by domain
        chunk_type: Filter by chunk type
        chunk_id: Search within this specific chunk
        date_from: YYYY-MM-DD
        date_to: YYYY-MM-DD
        fuzzy: Use fuzzy matching (default False)
        fuzzy_threshold: Fuzzy threshold 0-100 (default 80)
        limit: Max matching chunks (default 20)
    """
    await _ensure_init()
    result = await _db.archive_grep(
        pattern, project=project, domain=domain, chunk_type=chunk_type,
        chunk_id=chunk_id, date_from=date_from, date_to=date_to,
        fuzzy=fuzzy, fuzzy_threshold=fuzzy_threshold, limit=limit,
    )

    if result["status"] == "error":
        return f"Search error: {result['message']}"

    matches = result.get("matches", [])
    if not matches:
        mode = "fuzzy" if result.get("fuzzy") else "regex"
        return f"No {mode} matches for '{pattern}'."

    lines = [f"Found '{pattern}' in {len(matches)} chunk(s):"]
    for m in matches:
        ct = f" ({m['chunk_type']})" if m.get("chunk_type") and m["chunk_type"] != "general" else ""
        lines.append(f"\n\u2022 {m['id']}{ct} \u2014 {m['match_count']} match(es)")
        lines.append(f"  {m['summary']}")
        for match in m.get("matches", [])[:3]:
            lines.append(f"  Line {match['line_number']}:")
            lines.append(f"  {match['context']}")
    lines.append("\n\u2192 Use archive_read(id) for full document.")
    return "\n".join(lines)


@mcp.tool()
@_gc_tool
async def archive_read(chunk_id: str, start: int = 0, end: int = 0) -> str:
    """Read an archived document by ID, with optional line-range navigation.

    Args:
        chunk_id: The archive document ID (exact or prefix)
        start: Start line (1-indexed, 0 = beginning)
        end: End line (0 = to end)
    """
    await _ensure_init()
    result = await _db.archive_read(chunk_id, start=start, end=end)
    if result is None:
        return f"Document {chunk_id} not found."

    meta = result.get("meta") or {}
    content = result.get("content") or ""
    tags = ", ".join(meta.get("tags", [])) or "none"
    proj = meta.get("project", "")
    dom = meta.get("domain", "")
    ct = meta.get("chunk_type", "")
    location = f"{proj}/{dom}" if dom else proj
    type_str = f" | Type: {ct}" if ct and ct != "general" else ""
    return (
        f"Archive: {meta.get('summary', chunk_id)}\n"
        f"ID: {meta.get('chunk_id', chunk_id)} | Location: {location}{type_str} | Tags: {tags}\n\n"
        f"{content.strip()}"
    )


@mcp.tool()
@_gc_tool
async def archive_list(
    project: str = "",
    domain: str = "",
    tag: str = "",
    chunk_type: str = "",
    from_agent: str = "",
    limit: int = 20,
    offset: int = 0,
) -> str:
    """List archived documents, newest first.

    Args:
        project: Filter by project
        domain: Filter by domain
        tag: Filter by tag
        chunk_type: Filter by type
        from_agent: Filter by creating agent
        limit: Max results (default 20)
        offset: Skip N results
    """
    await _ensure_init()
    chunks = await _db.archive_list(
        project=project, domain=domain, tag=tag, chunk_type=chunk_type,
        from_agent=from_agent, limit=limit, offset=offset,
    )

    if not chunks:
        filters = " | ".join(f for f in [project, domain, tag, chunk_type] if f)
        return f"No archived documents{' matching: ' + filters if filters else ''}."

    lines = [f"Archived documents ({len(chunks)} shown):"]
    for c in chunks:
        loc = f" [{c.get('project', '')}/{c.get('domain', '')}]" if c.get("domain") else f" [{c.get('project', '')}]"
        ct = f" ({c.get('chunk_type', '')})" if c.get("chunk_type") and c["chunk_type"] != "general" else ""
        lines.append(f"\u2022 {c['chunk_id']}{loc}{ct}: {c.get('summary', '(no summary)')}")
    lines.append("\n\u2192 Use archive_read(id) for full content.")
    return "\n".join(lines)


@mcp.tool()
@_gc_tool
async def archive_update(
    chunk_id: str,
    content: str = "",
    add_tags: str = "",
    remove_tags: str = "",
    summary: str = "",
    domain: str = "",
    status: str = "",
) -> str:
    """Update an archived document's content, tags, or metadata.

    Args:
        chunk_id: Document ID to update
        content: New content (replaces existing)
        add_tags: Comma-separated tags to add
        remove_tags: Comma-separated tags to remove
        summary: New summary
        domain: Change domain
        status: Change status
    """
    await _ensure_init()
    add = _parse_tags(add_tags) if add_tags else None
    remove = _parse_tags(remove_tags) if remove_tags else None
    result = await _db.archive_update(
        chunk_id, content=content, add_tags=add, remove_tags=remove,
        summary=summary, domain=domain, status=status,
    )
    if result["status"] == "error":
        return f"Update failed: {result['message']}"
    return f"Updated {chunk_id}."


@mcp.tool()
@_gc_tool
async def archive_delete(chunk_id: str) -> str:
    """Delete an archived document by ID. Permanent."""
    await _ensure_init()
    deleted = await _db.archive_delete(chunk_id)
    if not deleted:
        return f"Document {chunk_id} not found."
    return f"Deleted {chunk_id}."


@mcp.tool()
@_gc_tool
async def archive_retention(days: int = 0, execute: bool = False) -> str:
    """Preview or run retention lifecycle on the archive.

    Args:
        days: Override retention period (0 = use defaults per chunk_type)
        execute: Actually run retention (default: preview only)
    """
    await _ensure_init()
    result = await _db.archive_retention_run(days=days, dry_run=not execute)

    if result.get("dry_run"):
        to_archive = result.get("to_archive", [])
        protected = result.get("protected", [])
        total = result.get("total_chunks", 0)
        lines = [f"Retention preview ({total} total chunks):"]
        if to_archive:
            lines.append(f"\nWould archive {len(to_archive)} chunk(s):")
            for item in to_archive[:10]:
                lines.append(f"  \u2022 {item['id']} ({item['chunk_type']}, {item['age_days']}d old)")
        else:
            lines.append("\nNothing to archive.")
        if protected:
            lines.append(f"\nProtected ({len(protected)} frequently accessed):")
            for item in protected[:5]:
                lines.append(f"  \u2022 {item['id']} ({item['access_count']} accesses)")
        if to_archive:
            lines.append("\n\u2192 Run with execute=True to proceed.")
        return "\n".join(lines)
    else:
        archived = result.get("archived", [])
        protected = result.get("protected", [])
        return f"Retention complete: {len(archived)} archived, {len(protected)} protected."


# ═══════════════════════════════════════════════════════════════════════════════
#  CROSS-MEMORY SEARCH
# ═══════════════════════════════════════════════════════════════════════════════


@mcp.tool()
@_gc_tool
async def deep_recall(query: str, active_project: str = "", limit: int = 5) -> str:
    """Search both semantic memory (Ember) and the document archive simultaneously.

    Args:
        query: What to search for
        active_project: Project name for scoping
        limit: Max results from each system (default 5)
    """
    await _ensure_init()

    lines = ["=== DEEP RECALL ===", f"Query: '{query}'", ""]

    # 1. Ember semantic search — recency=0: deep_recall reaches back freely
    #    (old-but-general principles and specific historical lookups are the point).
    scored = await _graph_rerank(
        query, top_k=limit, active_project=active_project, recency=0.0
    )
    lines.append("--- Ember (semantic memory) ---")
    if scored:
        for ember, score, _ in scored:
            stale = " [STALE]" if ember["is_stale"] else ""
            preview = _make_preview(ember["content"])
            lines.append(f"\u2022 {ember['name']} [id: {ember['ember_id']}] (score: {score:.2f}){stale}")
            lines.append(f"  {preview}")
        lines.append("\u2192 Use ember_read(id) for full content.")
    else:
        lines.append("(no semantic memories found)")

    lines.append("")

    # 2. Archive semantic search (via vec_archive)
    vec, embedding = _embed(query)
    archive_vec_results = await _db.search_archive(embedding, k=limit)
    if archive_vec_results:
        lines.append("--- Archive (semantic on summaries) ---")
        for chunk_id, dist in archive_vec_results:
            cos = l2_to_cosine(dist)
            read_result = await _db.archive_read(chunk_id)
            if read_result and read_result.get("meta", {}).get("status") != "archived":
                meta = read_result.get("meta", {})
                lines.append(
                    f"\u2022 {chunk_id} [{meta.get('project', '')}] (cos: {cos:.2f})"
                )
                lines.append(f"  {meta.get('summary', '')}")
        lines.append("")

    # 3. Archive FTS search
    lines.append("--- Archive (keyword search) ---")
    fts_results = await _db.archive_search_fts(query, limit=limit, project=active_project)
    if fts_results:
        for r in fts_results:
            loc = f"{r.get('project', '')}"
            lines.append(f"\u2022 {r['chunk_id']} [{loc}] (score: {r.get('score', 0):.2f})")
            lines.append(f"  {r['summary']}")
        lines.append("\u2192 Use archive_read(id) for full document.")
    else:
        lines.append("(no archive documents found)")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
#  DREAM CYCLE
# ═══════════════════════════════════════════════════════════════════════════════

_COPOUT_PATTERNS = [
    r"^good question", r"^let me think", r"^i('ll| will) get back",
    r"^that'?s (an )?interesting", r"^i('m| am) not sure",
    r"^hmm,? let me", r"^this is a (complex|nuanced|great)",
    r"^(certainly|absolutely|of course)[!,.]",
]


def _is_copout(text: str) -> bool:
    lower = text.strip().lower()
    return any(re.match(p, lower) for p in _COPOUT_PATTERNS)


def _dream_depth(tags: list[str]) -> int:
    for tag in tags:
        if tag.startswith("dream-depth:"):
            try:
                return int(tag.split(":")[1])
            except (ValueError, IndexError):
                pass
    if "dream" in tags:
        return 0
    return -1


@mcp.tool()
@_gc_tool
async def ember_dream_scan(
    days_back: int = 7, max_targets: int = 3, max_pairs: int = 5,
) -> str:
    """Analyze memory graph for dream cycle consolidation.

    Args:
        days_back: Days of recent embers to include (default 7)
        max_targets: Max bridge targets (default 3)
        max_pairs: Max similar pairs (default 5)
    """
    await _ensure_init()
    now = datetime.now(timezone.utc)

    all_embers = await _db.list_embers()
    if not all_embers:
        return "No embers in storage."

    active = [e for e in all_embers if not e["is_stale"]]
    if not active:
        return "All embers are stale."

    # Edge counts
    edge_counts: dict[str, int] = {}
    for e in active:
        edges = await _db.get_edges(e["ember_id"])
        edge_counts[e["ember_id"]] = len(edges)

    # Bridge targets (0-2 edges)
    targets = []
    max_ec = max(edge_counts.values()) if edge_counts else 0
    for e in active:
        ec = edge_counts.get(e["ember_id"], 0)
        if ec <= 2:
            targets.append((e, ec, max_ec - ec))
    targets.sort(key=lambda x: x[2], reverse=True)
    targets = targets[:max_targets]

    cell_embers: dict[int, list] = {}
    for e in active:
        cell_embers.setdefault(e["cell_id"], []).append(e)

    bridge_lines = []
    for target, ec, _ in targets:
        bridge_lines.append(
            f"\n  TARGET: {target['name']} [id: {target['ember_id']}] "
            f"(edges: {ec}, cell: {target['cell_id']})"
        )
        bridge_lines.append(f"  Content: {target['content'][:200]}")

        candidates = []
        for cid, embs in cell_embers.items():
            if cid == target["cell_id"]:
                continue
            for e in embs:
                candidates.append((e, edge_counts.get(e["ember_id"], 0)))
        candidates.sort(key=lambda x: x[1], reverse=True)

        bridge_lines.append("  Candidates (most-connected from other cells):")
        for cand, cand_ec in candidates[:3]:
            bridge_lines.append(
                f"    \u2022 {cand['name']} [id: {cand['ember_id']}] "
                f"(edges: {cand_ec}, cell: {cand['cell_id']}): {cand['content'][:200]}"
            )

        context_cells = [c for c in cell_embers if c != target["cell_id"]]
        random.shuffle(context_cells)
        bridge_lines.append("  Context (one per cell, stratified):")
        for cid in context_cells[:6]:
            sample = random.choice(cell_embers[cid])
            bridge_lines.append(f"    \u2022 [cell {cid}] {sample['name']}: {sample['content'][:200]}")

    # Recent embers
    cutoff = now.timestamp() - (days_back * 86400)
    recent = []
    for e in active:
        created = datetime.fromisoformat(e["created_at"].replace("Z", "+00:00"))
        if created.timestamp() > cutoff:
            recent.append(e)
    recent.sort(key=lambda e: e["created_at"], reverse=True)
    recent = recent[:30]

    recent_lines = []
    for e in recent:
        created = datetime.fromisoformat(e["created_at"].replace("Z", "+00:00"))
        age_h = (now - created).total_seconds() / 3600
        age_str = f"{int(age_h)}h ago" if age_h < 48 else f"{int(age_h / 24)}d ago"
        recent_lines.append(
            f"  \u2022 {e['name']} [id: {e['ember_id']}] ({age_str}, {e['importance']}): "
            f"{e['content'][:150]}"
        )

    # Similar pairs (batch embed + matmul)
    pair_lines = []
    pairs = []
    if len(active) >= 2:
        texts = [e["content"] for e in active]
        vectors = _embedder.batch_embed(texts)
        sim_matrix = vectors @ vectors.T

        for i in range(len(active)):
            for j in range(i + 1, len(active)):
                cos = float(sim_matrix[i, j])
                if 0.50 < cos < 0.80:
                    pairs.append((active[i], active[j], cos))
        pairs.sort(key=lambda x: x[2], reverse=True)
        pairs = pairs[:max_pairs]

        for a, b, cos in pairs:
            pair_lines.append(f"  Pair (cosine: {cos:.3f}):")
            pair_lines.append(f"    A: {a['name']} [id: {a['ember_id']}]: {a['content'][:200]}")
            pair_lines.append(f"    B: {b['name']} [id: {b['ember_id']}]: {b['content'][:200]}")

    # Dense cells
    cell_counts: dict[int, int] = {}
    for e in active:
        cell_counts[e["cell_id"]] = cell_counts.get(e["cell_id"], 0) + 1
    dense_cells = sorted(cell_counts.items(), key=lambda x: x[1], reverse=True)[:3]
    dense_lines = []
    for cid, count in dense_cells:
        dense_lines.append(f"\n  Cell {cid} ({count} embers):")
        for e in cell_embers.get(cid, [])[:8]:
            dense_lines.append(f"    \u2022 {e['name']}: {e['content'][:150]}")

    # Assemble
    output = [
        f"=== DREAM SCAN ({len(active)} active embers) ===",
        "",
        f"--- Bridge Targets ({len(targets)} isolated/weak) ---",
    ]
    output.extend(bridge_lines or ["  No isolated embers found."])
    output.append("")
    output.append(f"--- Recent Embers (last {days_back} days: {len(recent)}) ---")
    output.extend(recent_lines or ["  No recent embers."])
    output.append("")
    output.append(f"--- Similar Pairs ({len(pairs)}) ---")
    output.extend(pair_lines or ["  No similar pairs in 0.50-0.80 range."])
    output.append("")
    output.append(f"--- Dense Cells (top {len(dense_cells)}) ---")
    output.extend(dense_lines or ["  No dense cells."])

    return "\n".join(output)


@mcp.tool()
@_gc_tool
async def ember_dream_save(
    content: str,
    source_ember_id: str,
    bridge_to_ids: str = "",
    tags: str = "",
) -> str:
    """Store a dream bridge connecting isolated memories.

    Args:
        content: The dream text (associative paragraph)
        source_ember_id: The ember this dream is derived from
        bridge_to_ids: Comma-separated IDs of embers this bridges to
        tags: Additional comma-separated tags
    """
    await _ensure_init()

    if _is_copout(content):
        return "Rejected: content looks like a hedge. Write a genuine associative thought."

    source = await _db.get_ember(source_ember_id)
    if not source:
        return f"Source ember {source_ember_id} not found."

    await _db.increment_access(source_ember_id)

    source_depth = _dream_depth(source["tags"])
    new_depth = source_depth + 1

    tag_list = ["dream", "bridge", f"dream-depth:{new_depth}"]
    tag_list.extend(_parse_tags(tags))

    bridge_ids = [b.strip() for b in bridge_to_ids.split(",") if b.strip()] if bridge_to_ids else []
    for bid in bridge_ids:
        target = await _db.get_ember(bid)
        if not target:
            return f"Bridge target {bid} not found."
        await _db.increment_access(bid)

    vec, embedding = _embed(content)

    # Dedup against existing dreams
    existing_dreams = await _db.search_embers(embedding, k=10)
    for ex_id, dist in existing_dreams:
        cos = l2_to_cosine(dist)
        if cos > 0.82:
            ex = await _db.get_ember(ex_id)
            if ex and "dream" in ex["tags"]:
                return (
                    f"Too similar to existing dream '{ex['name']}' "
                    f"[id: {ex_id}] (cosine: {cos:.3f}). Write something more distinct."
                )

    cell_id = await _db.assign_cell(embedding)
    name = f"[dream] {content[:55].strip()}"
    if len(content) > 55:
        name = name.rsplit(" ", 1)[0] + "..."

    eid = str(uuid.uuid4())
    await _db.save_ember(
        ember_id=eid, name=name, content=content, tags=tag_list,
        cell_id=cell_id, importance="learning", source="dream",
        embedding=embedding,
    )

    # Edges
    await _db.save_edge(eid, source_ember_id, "derived_from", 1.0)
    for bid in bridge_ids:
        await _db.save_edge(eid, bid, "derived_from", 0.7)

    # Related edges from KNN
    related_count = 0
    search_results = await _db.search_embers(embedding, k=10)
    for rid, dist in search_results:
        if related_count >= 5:
            break
        if rid == eid:
            continue
        cos = l2_to_cosine(dist)
        if cos >= 0.28:
            await _db.save_edge(eid, rid, "related", 0.0)
            related_count += 1

    edge_summary = f"derived_from: {1 + len(bridge_ids)}, related: {related_count}"
    return (
        f"Dream stored: '{name}' [id: {eid}] in Cell {cell_id}. "
        f"Depth: {new_depth}. Edges: {edge_summary}"
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  SYNTHESIS CYCLE TOOLS
# ═══════════════════════════════════════════════════════════════════════════════


@mcp.tool()
@_gc_tool
async def ember_synthesis_scan(
    days_back: int = 14,
    max_recent: int = 30,
    max_pairs: int = 8,
    decay_threshold: float = 0.08,
) -> str:
    """Structured scan for the synthesis phase of the sleep cycle.

    Returns five sections:
    - recent_embers: newly created embers for theme analysis
    - suspicious_pairs: high-cosine pairs (>0.80) — duplicate/contradiction candidates
    - dense_clusters: top 3 densest cells for wonder generation
    - foundation_audit: high-importance embers with dormancy and promotion stats
    - decayed_embers: low-utility active embers candidates for archival

    Args:
        days_back: Recency window for recent_embers (default 14)
        max_recent: Cap on recent_embers returned (default 30)
        max_pairs: Cap on suspicious_pairs returned (default 8)
        decay_threshold: utility_score below this → decayed candidate (default 0.08)
    """
    await _ensure_init()
    now = datetime.now(timezone.utc)

    all_embers = await _db.list_embers()
    if not all_embers:
        return "No embers in storage."

    active = [e for e in all_embers if not e["is_stale"]]
    if not active:
        return "All embers are stale."

    # ── Recent Embers ─────────────────────────────────────────────────────────
    cutoff = now.timestamp() - (days_back * 86400)
    recent = []
    for e in active:
        created = datetime.fromisoformat(e["created_at"].replace("Z", "+00:00"))
        if created.timestamp() > cutoff:
            recent.append(e)
    recent.sort(key=lambda e: e["created_at"], reverse=True)
    recent = recent[:max_recent]

    recent_lines = []
    for e in recent:
        created = datetime.fromisoformat(e["created_at"].replace("Z", "+00:00"))
        age_h = (now - created).total_seconds() / 3600
        age_str = f"{int(age_h)}h ago" if age_h < 48 else f"{int(age_h / 24)}d ago"
        recent_lines.append(
            f"  \u2022 {e['name']} [id: {e['ember_id']}] "
            f"({age_str}, {e['importance']}, access:{e['access_count']}): "
            f"{e['content'][:180]}"
        )

    # ── Suspicious Pairs (cosine > 0.80) ─────────────────────────────────────
    pair_lines: list[str] = []
    pairs: list = []
    if len(active) >= 2:
        texts = [e["content"] for e in active]
        vectors = _embedder.batch_embed(texts)
        sim_matrix = vectors @ vectors.T

        for i in range(len(active)):
            for j in range(i + 1, len(active)):
                cos = float(sim_matrix[i, j])
                if cos > 0.80:
                    pairs.append((active[i], active[j], cos))
        pairs.sort(key=lambda x: x[2], reverse=True)
        pairs = pairs[:max_pairs]

        for a, b, cos in pairs:
            pair_lines.append(f"  Pair (cosine: {cos:.3f}):")
            pair_lines.append(
                f"    A: {a['name']} [id: {a['ember_id']}] "
                f"(access:{a['access_count']}, {a['importance']}): "
                f"{a['content'][:220]}"
            )
            pair_lines.append(
                f"    B: {b['name']} [id: {b['ember_id']}] "
                f"(access:{b['access_count']}, {b['importance']}): "
                f"{b['content'][:220]}"
            )

    # ── Dense Clusters ────────────────────────────────────────────────────────
    cell_embers: dict[int, list] = {}
    for e in active:
        cell_embers.setdefault(e["cell_id"], []).append(e)
    cell_counts = {cid: len(embs) for cid, embs in cell_embers.items()}
    dense_cells = sorted(cell_counts.items(), key=lambda x: x[1], reverse=True)[:3]

    cluster_lines: list[str] = []
    for cid, count in dense_cells:
        cluster_lines.append(f"\n  Cluster cell={cid} ({count} embers):")
        for e in cell_embers.get(cid, [])[:10]:
            cluster_lines.append(
                f"    \u2022 {e['name']} [id: {e['ember_id']}]: {e['content'][:160]}"
            )

    # ── Foundation Audit ─────────────────────────────────────────────────────
    HIGH_IMPORTANCE = {"fact", "decision"}
    foundations = [e for e in active if e["importance"] in HIGH_IMPORTANCE]
    foundations.sort(key=lambda e: e["access_count"])
    dormant = [e for e in foundations if e["access_count"] == 0]
    promotion_candidates = [
        e for e in active
        if e["access_count"] >= 5 and e["importance"] not in HIGH_IMPORTANCE
    ]
    promotion_candidates.sort(key=lambda e: e["access_count"], reverse=True)

    foundation_lines: list[str] = []
    foundation_lines.append(f"  Total high-importance embers: {len(foundations)}")
    foundation_lines.append(f"  Dormant (never accessed): {len(dormant)}")
    if dormant:
        for e in dormant[:5]:
            created = datetime.fromisoformat(e["created_at"].replace("Z", "+00:00"))
            age_d = int((now - created).total_seconds() / 86400)
            foundation_lines.append(
                f"    \u2022 [DORMANT] {e['name']} [id: {e['ember_id']}] "
                f"(age:{age_d}d, {e['importance']}): {e['content'][:160]}"
            )
    if promotion_candidates:
        foundation_lines.append("  Promotion candidates (high access, lower importance):")
        for e in promotion_candidates[:5]:
            foundation_lines.append(
                f"    \u2022 {e['name']} [id: {e['ember_id']}] "
                f"(access:{e['access_count']}, {e['importance']})"
            )

    # ── Decayed Embers ────────────────────────────────────────────────────────
    decayed = [
        e for e in active
        if e.get("utility_score", 0.5) < decay_threshold
    ]
    decayed.sort(key=lambda e: e.get("utility_score", 0.5))
    decayed = decayed[:5]

    decay_lines: list[str] = []
    for e in decayed:
        created = datetime.fromisoformat(e["created_at"].replace("Z", "+00:00"))
        age_d = int((now - created).total_seconds() / 86400)
        decay_lines.append(
            f"  \u2022 {e['name']} [id: {e['ember_id']}] "
            f"(utility:{e.get('utility_score', 0):.3f}, age:{age_d}d, {e['importance']}): "
            f"{e['content'][:160]}"
        )

    # ── Assemble ──────────────────────────────────────────────────────────────
    output = [
        f"=== SYNTHESIS SCAN ({len(active)} active embers) ===",
        "",
        f"--- recent_embers (last {days_back} days: {len(recent)}) ---",
    ]
    output.extend(recent_lines or ["  No recent embers."])
    output.append("")
    output.append(f"--- suspicious_pairs ({len(pairs)} pairs with cosine > 0.80) ---")
    output.extend(pair_lines or ["  No highly-similar pairs found."])
    output.append("")
    output.append(f"--- dense_clusters (top {len(dense_cells)}) ---")
    output.extend(cluster_lines or ["  No clusters."])
    output.append("")
    output.append("--- foundation_audit ---")
    output.extend(foundation_lines)
    output.append("")
    output.append(
        f"--- decayed_embers ({len(decayed)} below utility {decay_threshold}) ---"
    )
    output.extend(decay_lines or ["  No decayed embers."])

    return "\n".join(output)


@mcp.tool()
@_gc_tool
async def ember_synthesis_save(
    save_type: str,
    content: str = "",
    name: str = "",
    source_ember_ids: str = "",
    stale_ember_id: str = "",
) -> str:
    """Store synthesis cycle outputs: themes, resolutions, consolidations, wonders.

    Args:
        save_type: One of: "theme", "resolution", "consolidation", "wonder"
        content: The synthesized content (required for all types)
        name: Name for the new ember (required for theme; auto-derived for others)
        source_ember_ids: Comma-separated IDs of contributing embers
        stale_ember_id: ID of the ember to retire (resolution/consolidation)
    """
    await _ensure_init()

    VALID_TYPES = {"theme", "resolution", "consolidation", "wonder"}
    if save_type not in VALID_TYPES:
        return f"Invalid save_type '{save_type}'. Must be one of: {', '.join(sorted(VALID_TYPES))}"

    source_ids = [
        sid.strip() for sid in source_ember_ids.split(",") if sid.strip()
    ] if source_ember_ids else []

    # ── Theme ─────────────────────────────────────────────────────────────────
    if save_type == "theme":
        if not content.strip():
            return "content is required for save_type='theme'."
        if not name.strip():
            return "name is required for save_type='theme'."

        vec, embedding = _embed(content)

        # Dedup: reject themes too similar to existing embers
        existing = await _db.search_embers(embedding, k=5)
        for ex_id, dist in existing:
            cos = l2_to_cosine(dist)
            if cos > 0.875:
                ex = await _db.get_ember(ex_id)
                if ex and not ex["is_stale"]:
                    await _db.increment_access(ex_id)
                    return (
                        f"Too similar to existing ember '{ex['name']}' "
                        f"[id: {ex_id}] (cosine: {cos:.3f}). Reinforced instead."
                    )

        cell_id = await _db.assign_cell(embedding)
        eid = str(uuid.uuid4())
        await _db.save_ember(
            ember_id=eid, name=name, content=content,
            tags=["synthesis-theme", "synthesis"],
            cell_id=cell_id, importance="learning", source="synthesis",
            embedding=embedding,
        )
        for sid in source_ids:
            if await _db.get_ember(sid):
                await _db.save_edge(eid, sid, "derived_from", 0.8)
        ember_dict = await _db.get_ember(eid)
        await _shadow_on_insert(eid, ember_dict, vec)
        await _db.log_update(
            ember_id=eid, event_type="edit", origin="agent",
            content_changed=True, session_id=_SESSION_ID,
            details="synthesis: theme",
        )
        return f"Theme stored: '{name}' [id: {eid}]"

    # ── Resolution ────────────────────────────────────────────────────────────
    elif save_type == "resolution":
        if not stale_ember_id:
            return "stale_ember_id is required for save_type='resolution'."
        if not content.strip():
            return "content is required for save_type='resolution'."

        stale_ember = await _db.get_ember(stale_ember_id)
        if not stale_ember:
            return f"Ember {stale_ember_id} not found."

        vec, embedding = _embed(content)
        cell_id = await _db.assign_cell(embedding)
        eid = str(uuid.uuid4())
        res_name = name.strip() or f"[resolution] {stale_ember['name'][:60].strip()}"

        await _db.save_ember(
            ember_id=eid, name=res_name, content=content,
            tags=["synthesis-resolution", "synthesis"],
            cell_id=cell_id, importance=stale_ember["importance"],
            source="synthesis", embedding=embedding,
            supersedes_id=stale_ember_id,
        )
        await _db.update_ember(
            stale_ember_id, content_changed=False,
            is_stale=True,
            stale_reason="Resolved by synthesis cycle",
            shadowed_by=eid,
            shadow_load=1.0,
            shadow_updated_at=datetime.now(timezone.utc).isoformat(),
        )
        await _db.save_edge(stale_ember_id, eid, "supersedes", 1.0)
        for sid in source_ids:
            if sid != stale_ember_id and await _db.get_ember(sid):
                await _db.save_edge(eid, sid, "derived_from", 0.7)
        ember_dict = await _db.get_ember(eid)
        await _shadow_on_insert(eid, ember_dict, vec)
        # Resolution is a contradict-like operation: old retired, new created.
        await _db.log_update(
            ember_id=stale_ember_id, event_type="contradict_old", origin="agent",
            content_changed=False, session_id=_SESSION_ID,
            details=f"synthesis resolution; superseded by {eid}",
        )
        await _db.log_update(
            ember_id=eid, event_type="contradict_new", origin="agent",
            content_changed=True, session_id=_SESSION_ID,
            details=f"synthesis resolution of {stale_ember_id}",
        )
        return (
            f"Resolution stored: '{res_name}' [id: {eid}]. "
            f"Retired: '{stale_ember['name']}' [{stale_ember_id}]"
        )

    # ── Consolidation ─────────────────────────────────────────────────────────
    elif save_type == "consolidation":
        if not stale_ember_id:
            return "stale_ember_id is required for save_type='consolidation'."
        if not content.strip():
            return "content is required for save_type='consolidation'."
        if len(source_ids) < 2:
            return "source_ember_ids must contain both IDs (weaker,stronger) for consolidation."

        weaker_id = stale_ember_id
        stronger_id = next((sid for sid in source_ids if sid != weaker_id), None)
        if not stronger_id:
            return "Could not identify stronger ember from source_ember_ids."

        weaker = await _db.get_ember(weaker_id)
        stronger = await _db.get_ember(stronger_id)
        if not weaker:
            return f"Weaker ember {weaker_id} not found."
        if not stronger:
            return f"Stronger ember {stronger_id} not found."

        # Update stronger ember content in-place
        vec, embedding = _embed(content)
        cell_id = await _db.assign_cell(embedding)
        await _db.update_ember(
            stronger_id, content_changed=True,
            content=content,
            cell_id=cell_id,
        )
        await _db.update_ember_vector(stronger_id, embedding)

        # Retire weaker, pointing to stronger
        await _db.update_ember(
            weaker_id, content_changed=False,
            is_stale=True,
            stale_reason=f"Consolidated into {stronger_id} by synthesis cycle",
            shadowed_by=stronger_id,
            shadow_load=1.0,
            shadow_updated_at=datetime.now(timezone.utc).isoformat(),
        )
        await _db.save_edge(weaker_id, stronger_id, "supersedes", 1.0)

        stronger_dict = await _db.get_ember(stronger_id)
        await _shadow_on_insert(stronger_id, stronger_dict, vec)
        # Consolidation: weaker retired, stronger updated (treat as edit on
        # stronger + contradict_old on weaker).
        await _db.log_update(
            ember_id=weaker_id, event_type="contradict_old", origin="agent",
            content_changed=False, session_id=_SESSION_ID,
            details=f"synthesis consolidated into {stronger_id}",
        )
        await _db.log_update(
            ember_id=stronger_id, event_type="edit", origin="agent",
            content_changed=True, session_id=_SESSION_ID,
            details=f"synthesis consolidation absorbed {weaker_id}",
        )
        return (
            f"Consolidated: '{weaker['name']}' [{weaker_id}] merged into "
            f"'{stronger['name']}' [{stronger_id}]"
        )

    # ── Wonder ────────────────────────────────────────────────────────────────
    elif save_type == "wonder":
        if not content.strip():
            return "content is required for save_type='wonder'."

        wonder_name = name.strip() if name.strip() else f"[wonder] {content[:55].strip()}"
        if not name.strip() and len(content) > 55:
            wonder_name = wonder_name.rsplit(" ", 1)[0] + "..."

        vec, embedding = _embed(content)
        cell_id = await _db.assign_cell(embedding)

        existing = await _db.search_embers(embedding, k=5)
        for ex_id, dist in existing:
            cos = l2_to_cosine(dist)
            if cos > 0.875:
                ex_ember = await _db.get_ember(ex_id)
                if ex_ember and not ex_ember["is_stale"]:
                    await _db.increment_access(ex_id)
                    return f"Reinforced existing wonder: '{ex_ember['name']}' (cosine: {cos:.3f})"

        eid = str(uuid.uuid4())
        await _db.save_ember(
            ember_id=eid, name=wonder_name, content=content,
            tags=["synthesis-wonder", "synthesis", "wonder"],
            cell_id=cell_id, importance="learning", source="synthesis",
            embedding=embedding,
        )
        for sid in source_ids:
            if await _db.get_ember(sid):
                await _db.save_edge(eid, sid, "derived_from", 0.6)
        ember_dict = await _db.get_ember(eid)
        await _shadow_on_insert(eid, ember_dict, vec)
        await _db.log_update(
            ember_id=eid, event_type="edit", origin="agent",
            content_changed=True, session_id=_SESSION_ID,
            details="synthesis: wonder",
        )
        return f"Wonder stored: '{wonder_name}' [id: {eid}]"

    return f"Unhandled save_type: {save_type}"  # unreachable


# ═══════════════════════════════════════════════════════════════════════════════
#  PROMPTS
# ═══════════════════════════════════════════════════════════════════════════════


@mcp.prompt()
def start_session() -> str:
    return (
        "Check my persistent memory for any relevant context.\n\n"
        "Steps:\n"
        "1. Call ember_auto with a summary of what the user is asking about\n"
        "2. If relevant memories are found, incorporate them naturally\n"
        "3. If there are recent 'Next Steps' embers, mention what was planned\n"
        "4. Respond to the user with full context"
    )


@mcp.prompt()
def end_session() -> str:
    return (
        "Before we end, let's save the important parts.\n\n"
        "Steps:\n"
        "1. Summarize key work (2-3 sentences)\n"
        "2. List decisions made\n"
        "3. Note next steps or open items\n"
        "4. Call ember_save_session with all three\n"
        "5. Confirm what was saved"
    )


@mcp.prompt()
def remember() -> str:
    return (
        "The user wants to save something to persistent memory.\n\n"
        "Steps:\n"
        "1. Identify what to remember (preference, fact, rule, context)\n"
        "2. Choose a clear, searchable name\n"
        "3. Add relevant tags\n"
        "4. Choose importance: fact, decision, preference, context, or learning\n"
        "5. Call ember_store\n"
        "6. Confirm what was saved"
    )


@mcp.prompt()
def dream_cycle() -> str:
    return (
        "You are dreaming. Memory consolidation cycle.\n\n"
        "Phase 1: Bridge Building\n"
        "1. ember_dream_scan to see memory state\n"
        "2. For each bridge target, write associative paragraphs connecting memories\n"
        "3. Save via ember_dream_save\n\n"
        "Phase 2: Theme Emergence\n"
        "4. What threads are converging in recent embers?\n"
        "5. Store themes via ember_store with tags: 'synthesis-theme'\n\n"
        "Phase 3: Contradiction Check\n"
        "6. Review similar pairs — duplicates, contradictions, or complementary?\n"
        "7. Resolve via ember_contradict or ember_delete\n\n"
        "Phase 4: Wonder Generation\n"
        "8. What questions do dense cells leave unanswered?\n"
        "9. Store 1-2 questions via ember_wonder with tag 'dream-generated'\n\n"
        "Phase 5: Report\n"
        "10. ember_health for overall check\n"
        "11. Summarize: bridges, themes, contradictions, wonders"
    )
