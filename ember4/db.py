"""Unified SQLite database layer for Ember 3.

One file. One format. One query language.
Replaces: StorageManager (JSON files), ArchiveStore (YAML/markdown files),
VectorEngine index management (FAISS/numpy), stats.db.
"""

import hashlib
import json
import logging
import re
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import aiosqlite
import numpy as np

from ember4.config import (
    CHUNK_TYPES,
    DEFAULT_RETENTION_DAYS,
    MAX_CONTENT_SIZE,
    SUBCONSCIOUS_RETENTION_DAYS,
    get_db_path,
)

logger = logging.getLogger(__name__)

# Path traversal guard for chunk IDs
CHUNK_ID_RE = re.compile(r"^[\w.-]+$")

# ── Schema SQL ────────────────────────────────────────────────────────────────
# Embedded from schema.sql for package self-containment.
# Distance metric: sqlite-vec returns L2 distance (not squared).
# For L2-normalized vectors: cos_sim = 1.0 - distance**2 / 2.0 (utils.l2_to_cosine)

SCHEMA_SQL = """
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS embers (
    ember_id        TEXT PRIMARY KEY,
    name            TEXT NOT NULL,
    content         TEXT NOT NULL,
    tags            TEXT NOT NULL DEFAULT '[]',
    cell_id         INTEGER NOT NULL DEFAULT -1,
    importance      TEXT NOT NULL DEFAULT 'context',
    source          TEXT NOT NULL DEFAULT 'manual',
    source_path     TEXT,
    agent           TEXT,
    supersedes_id       TEXT REFERENCES embers(ember_id) ON DELETE SET NULL,
    superseded_by_id    TEXT REFERENCES embers(ember_id) ON DELETE SET NULL,
    is_stale            INTEGER NOT NULL DEFAULT 0,
    stale_reason        TEXT,
    last_accessed_at    TEXT,
    access_count        INTEGER NOT NULL DEFAULT 0,
    utility_score       REAL NOT NULL DEFAULT 0.5,
    shadow_load         REAL NOT NULL DEFAULT 0.0,
    shadowed_by         TEXT REFERENCES embers(ember_id) ON DELETE SET NULL,
    shadow_updated_at   TEXT,
    session_id      TEXT,
    created_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    updated_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

CREATE INDEX IF NOT EXISTS idx_embers_cell ON embers(cell_id);
CREATE INDEX IF NOT EXISTS idx_embers_importance ON embers(importance);
CREATE INDEX IF NOT EXISTS idx_embers_stale ON embers(is_stale);
CREATE INDEX IF NOT EXISTS idx_embers_created ON embers(created_at);
CREATE INDEX IF NOT EXISTS idx_embers_accessed ON embers(last_accessed_at);

CREATE VIRTUAL TABLE IF NOT EXISTS vec_embers USING vec0(
    ember_id TEXT PRIMARY KEY,
    embedding float[384]
);

CREATE VIRTUAL TABLE IF NOT EXISTS vec_archive USING vec0(
    chunk_id TEXT PRIMARY KEY,
    embedding float[384]
);

CREATE VIRTUAL TABLE IF NOT EXISTS vec_centroids USING vec0(
    cell_id INTEGER PRIMARY KEY,
    embedding float[384]
);

CREATE TABLE IF NOT EXISTS edges (
    source_id   TEXT NOT NULL,
    target_id   TEXT NOT NULL,
    edge_type   TEXT NOT NULL,
    weight      REAL NOT NULL DEFAULT 0.0,
    created_at  TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    PRIMARY KEY (source_id, target_id, edge_type)
) WITHOUT ROWID;

CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source_id);
CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target_id);
CREATE INDEX IF NOT EXISTS idx_edges_type ON edges(edge_type);

CREATE TABLE IF NOT EXISTS region_stats (
    cell_id         INTEGER PRIMARY KEY,
    vitality_score  REAL NOT NULL DEFAULT 0.0,
    shadow_accum    REAL NOT NULL DEFAULT 0.0,
    last_updated    TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

CREATE TABLE IF NOT EXISTS archive (
    chunk_id        TEXT PRIMARY KEY,
    summary         TEXT NOT NULL,
    content         TEXT NOT NULL,
    tags            TEXT NOT NULL DEFAULT '[]',
    project         TEXT NOT NULL DEFAULT '',
    domain          TEXT NOT NULL DEFAULT '',
    chunk_type      TEXT NOT NULL DEFAULT 'general',
    from_agent      TEXT NOT NULL DEFAULT '',
    for_agent       TEXT NOT NULL DEFAULT '',
    status          TEXT NOT NULL DEFAULT 'active',
    category        TEXT NOT NULL DEFAULT '',
    entity          TEXT NOT NULL DEFAULT '',
    reply_to        TEXT NOT NULL DEFAULT '',
    tokens_estimate INTEGER NOT NULL DEFAULT 0,
    content_hash    TEXT NOT NULL DEFAULT '',
    access_count    INTEGER NOT NULL DEFAULT 0,
    last_accessed_at TEXT,
    archived_at     TEXT,
    created_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

CREATE INDEX IF NOT EXISTS idx_archive_type ON archive(chunk_type);
CREATE INDEX IF NOT EXISTS idx_archive_project ON archive(project);
CREATE INDEX IF NOT EXISTS idx_archive_created ON archive(created_at);
CREATE INDEX IF NOT EXISTS idx_archive_status ON archive(status);

CREATE VIRTUAL TABLE IF NOT EXISTS archive_fts USING fts5(
    chunk_id UNINDEXED,
    summary,
    content,
    tags,
    project,
    content='archive',
    content_rowid='rowid',
    tokenize='porter unicode61'
);

CREATE TRIGGER IF NOT EXISTS archive_ai AFTER INSERT ON archive BEGIN
    INSERT INTO archive_fts(rowid, chunk_id, summary, content, tags, project)
    VALUES (new.rowid, new.chunk_id, new.summary, new.content, new.tags, new.project);
END;

CREATE TRIGGER IF NOT EXISTS archive_ad AFTER DELETE ON archive BEGIN
    INSERT INTO archive_fts(archive_fts, rowid, chunk_id, summary, content, tags, project)
    VALUES ('delete', old.rowid, old.chunk_id, old.summary, old.content, old.tags, old.project);
END;

CREATE TRIGGER IF NOT EXISTS archive_au AFTER UPDATE ON archive BEGIN
    INSERT INTO archive_fts(archive_fts, rowid, chunk_id, summary, content, tags, project)
    VALUES ('delete', old.rowid, old.chunk_id, old.summary, old.content, old.tags, old.project);
    INSERT INTO archive_fts(rowid, chunk_id, summary, content, tags, project)
    VALUES (new.rowid, new.chunk_id, new.summary, new.content, new.tags, new.project);
END;

CREATE TABLE IF NOT EXISTS recall_log (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id      TEXT NOT NULL,
    ember_id        TEXT NOT NULL,
    event_type      TEXT NOT NULL,
    timestamp       TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

CREATE INDEX IF NOT EXISTS idx_recall_session ON recall_log(session_id);
CREATE INDEX IF NOT EXISTS idx_recall_ember ON recall_log(ember_id);

CREATE TABLE IF NOT EXISTS metrics_log (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp       TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    metric_type     TEXT NOT NULL,
    value           REAL,
    details         TEXT
);

CREATE TABLE IF NOT EXISTS loading_recipes (
    recipe_id    TEXT PRIMARY KEY,
    name         TEXT NOT NULL,
    description  TEXT NOT NULL DEFAULT '',
    tag_filters  TEXT NOT NULL DEFAULT '[]',
    topic_queries TEXT NOT NULL DEFAULT '[]',
    pinned_ids   TEXT NOT NULL DEFAULT '[]',
    exclude_tags TEXT NOT NULL DEFAULT '[]',
    procedural   TEXT NOT NULL DEFAULT '',
    max_embers   INTEGER NOT NULL DEFAULT 10,
    created_at   TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

CREATE TABLE IF NOT EXISTS config (
    key     TEXT PRIMARY KEY,
    value   TEXT NOT NULL
);

INSERT OR IGNORE INTO config(key, value) VALUES ('schema_version', '"3.0.0"');
INSERT OR IGNORE INTO config(key, value) VALUES ('dimension', '384');
INSERT OR IGNORE INTO config(key, value) VALUES ('k_cells', '16');

-- ─── Ember 4 additions (parallel to ember-3 tables; never overwrite them) ────
--
-- Naming convention: vec_<target>_<model>_<role>
--   target ∈ {embers, archive, fibers}
--   model  ∈ {nomic_v15, ...}
--   role   ∈ {clustering, retrieval}
--
-- Nomic-Embed-Text-v1.5 uses task-prefix discipline: queries get the
-- 'search_query:' prefix, documents 'search_document:', cluster-assignment
-- vectors 'clustering:'. Each prefix produces a different vector from the
-- same model — clustering proximity and retrieval match are not the same
-- space. Two vectors per ember: one for fiber assignment, one for recall.

CREATE VIRTUAL TABLE IF NOT EXISTS vec_embers_nomic_v15_clustering USING vec0(
    ember_id TEXT PRIMARY KEY,
    embedding float[768]
);

CREATE VIRTUAL TABLE IF NOT EXISTS vec_embers_nomic_v15_retrieval USING vec0(
    ember_id TEXT PRIMARY KEY,
    embedding float[768]
);

-- Archive summaries: retrieval vector only (no clustering — archive is
-- recall-by-summary, no fiber traversal). Embeds the `summary` field, which
-- is short enough for Nomic's 8K context regardless of full `content` size.
CREATE VIRTUAL TABLE IF NOT EXISTS vec_archive_nomic_v15_retrieval USING vec0(
    chunk_id TEXT PRIMARY KEY,
    embedding float[768]
);

-- Fiber centroids (populated by recompute_fibers_v4 in task 5).
-- For MPPCA we may need a side table for subspace bases; centroids first.
CREATE VIRTUAL TABLE IF NOT EXISTS vec_fibers_nomic_v15 USING vec0(
    fiber_id INTEGER PRIMARY KEY,
    embedding float[768]
);

-- Fiber assignment side table — parallels cell_id without overwriting it.
-- Multiple models can coexist (model_id in PK), so future embedder swaps
-- don't destroy prior assignments. fiber_position is reserved for later
-- (carrier coordinate + fiber-axis coordinate when MPPCA is used).
CREATE TABLE IF NOT EXISTS fiber_assignments (
    ember_id        TEXT NOT NULL,
    model_id        TEXT NOT NULL DEFAULT 'nomic-v1.5',
    fiber_id        INTEGER NOT NULL,
    fiber_position  REAL NOT NULL DEFAULT 0.0,
    confidence      REAL NOT NULL DEFAULT 1.0,
    assigned_at     TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
    PRIMARY KEY (ember_id, model_id),
    FOREIGN KEY (ember_id) REFERENCES embers(ember_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_fiber_assignments_fiber
    ON fiber_assignments(model_id, fiber_id);

-- Update-activity log — distinct from recall_log. Captures meaningful
-- engagement events (edits, contradicts, edge formation). Bulk operations
-- and pure recall do NOT contribute to importance scoring even when they
-- log here. Two columns enforce the distinction:
--
--   event_type ∈ {edit, contradict_old, contradict_new, edge_form,
--                 tag_update, auto_enrich}
--   origin     ∈ {user, agent, auto_system, migration}
--
-- The IMPORTANCE FILTER (computed at recall time, not in the table):
--   contributes_to_importance ←
--     (event_type IN ('edit','contradict_old','contradict_new','edge_form')
--      AND origin IN ('user','agent'))
--     OR
--     (event_type = 'tag_update' AND origin IN ('user','agent')
--      AND content_changed = 0)  -- small-weight: human-curated tag
--
-- This stops auto-system tag-on-recall, Haiku batch enrichment, and bulk
-- migrations from feedback-looping into engagement scoring — exactly the
-- basin pathology ember-3 suffered from.
--
-- content_changed = 1 when the actual content field was modified by the
-- event; 0 when only metadata (tags, utility, source_path) changed. Used to
-- distinguish a real edit from a metadata-only update with the same
-- event_type label.
CREATE TABLE IF NOT EXISTS update_log (
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

CREATE INDEX IF NOT EXISTS idx_update_log_ember ON update_log(ember_id);
CREATE INDEX IF NOT EXISTS idx_update_log_event ON update_log(event_type);
CREATE INDEX IF NOT EXISTS idx_update_log_origin ON update_log(origin);
CREATE INDEX IF NOT EXISTS idx_update_log_time ON update_log(timestamp);

-- Compound index for the common "scoped events for ember X in recency
-- window" query at recall time.
CREATE INDEX IF NOT EXISTS idx_update_log_ember_time
    ON update_log(ember_id, timestamp);

-- ─── FTS5 over embers (BM25 lexical retrieval) ──────────────────────────────
-- Parallel to archive_fts. Complements Nomic dense retrieval: dense handles
-- vocabulary-divergent paraphrases, BM25 handles vocabulary-shared cases
-- (e.g. "fiber bundles" ↔ "long skinny manifolds" share "manifold"). Porter
-- stemming + unicode tokenization matches the archive_fts settings so we
-- don't have two FTS regimes to reason about.

CREATE VIRTUAL TABLE IF NOT EXISTS embers_fts USING fts5(
    ember_id UNINDEXED,
    name,
    content,
    tags,
    content='embers',
    content_rowid='rowid',
    tokenize='porter unicode61'
);

CREATE TRIGGER IF NOT EXISTS embers_ai AFTER INSERT ON embers BEGIN
    INSERT INTO embers_fts(rowid, ember_id, name, content, tags)
    VALUES (new.rowid, new.ember_id, new.name, new.content, new.tags);
END;

CREATE TRIGGER IF NOT EXISTS embers_ad AFTER DELETE ON embers BEGIN
    INSERT INTO embers_fts(embers_fts, rowid, ember_id, name, content, tags)
    VALUES ('delete', old.rowid, old.ember_id, old.name, old.content, old.tags);
END;

CREATE TRIGGER IF NOT EXISTS embers_au AFTER UPDATE ON embers BEGIN
    INSERT INTO embers_fts(embers_fts, rowid, ember_id, name, content, tags)
    VALUES ('delete', old.rowid, old.ember_id, old.name, old.content, old.tags);
    INSERT INTO embers_fts(rowid, ember_id, name, content, tags)
    VALUES (new.rowid, new.ember_id, new.name, new.content, new.tags);
END;

INSERT OR IGNORE INTO config(key, value) VALUES ('ember4_schema_version', '"4.0.0-dev"');
INSERT OR IGNORE INTO config(key, value) VALUES ('ember4_embedder', '"nomic-v1.5"');
INSERT OR IGNORE INTO config(key, value) VALUES ('ember4_dimension', '768');
INSERT OR IGNORE INTO config(key, value) VALUES ('ember4_fts', '"embers_fts,archive_fts"');
"""


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def _parse_date(date_str: str) -> Optional[datetime]:
    if not date_str:
        return None
    try:
        if "T" in date_str:
            return datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        return datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except (ValueError, TypeError):
        return None


def _fts5_escape(query: str) -> str:
    """Escape user query for FTS5 MATCH. Wraps terms in quotes for safety."""
    words = query.split()
    if not words:
        return ""
    return " ".join(f'"{w}"' for w in words if w.strip())


class Database:
    """Unified SQLite database with sqlite-vec + FTS5."""

    def __init__(self, db_path: Path | str | None = None):
        self.db_path = Path(db_path) if db_path else get_db_path()
        self._conn: Optional[aiosqlite.Connection] = None

    async def connect(self) -> None:
        """Open connection, load sqlite-vec, initialize schema.

        Honors the EMBER4_READ_ONLY freeze flag from config: when set, opens
        the database in SQLite read-only URI mode and skips schema init.
        """
        from ember4.config import READ_ONLY
        self._read_only = READ_ONLY
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        if self._read_only:
            # file:URI with mode=ro guarantees no writes hit disk, even WAL.
            #
            # NOTE: we deliberately do NOT use immutable=1 here. immutable=1
            # tells SQLite the file won't change, which lets it cache schema
            # aggressively — including across our own schema migrations done
            # in a separate process. We hit this on 2026-05-13 when an
            # update_log migration succeeded but read-only connects still
            # saw the pre-migration columns. mode=ro alone gives us the
            # no-writes guarantee without the stale-schema risk. The cost is
            # negligible (SQLite re-checks freshness on each transaction).
            uri = f"file:{self.db_path}?mode=ro"
            self._conn = await aiosqlite.connect(uri, uri=True)
        else:
            self._conn = await aiosqlite.connect(str(self.db_path))
        self._conn.row_factory = aiosqlite.Row

        # Load sqlite-vec extension via aiosqlite's async API
        await self._conn.enable_load_extension(True)
        import sqlite_vec
        await self._conn.load_extension(sqlite_vec.loadable_path())

        if not self._read_only:
            # Initialize schema (idempotent via IF NOT EXISTS).
            # Skipped in read-only mode — the frozen DB already has the schema.
            await self._conn.executescript(SCHEMA_SQL)
            await self._conn.commit()

    async def close(self) -> None:
        if self._conn:
            await self._conn.close()
            self._conn = None

    @property
    def conn(self) -> aiosqlite.Connection:
        if self._conn is None:
            raise RuntimeError("Database not connected. Call connect() first.")
        return self._conn

    # ══════════════════════════════════════════════════════════════════════════
    #  EMBER CRUD
    # ══════════════════════════════════════════════════════════════════════════

    async def save_ember(
        self,
        ember_id: str,
        name: str,
        content: str,
        tags: list[str],
        cell_id: int,
        importance: str,
        source: str,
        embedding: bytes,
        source_path: str = "",
        agent: str = "",
        session_id: str = "",
        supersedes_id: str = "",
        utility_score: float = 0.5,
    ) -> None:
        """Insert ember row + vector in one transaction (rollback-safe)."""
        now = _now_iso()
        try:
            await self.conn.execute(
                """INSERT INTO embers (
                    ember_id, name, content, tags, cell_id, importance, source,
                    source_path, agent, session_id, supersedes_id,
                    utility_score, created_at, updated_at, last_accessed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    ember_id, name, content, json.dumps(tags), cell_id, importance,
                    source, source_path or None, agent or None, session_id or None,
                    supersedes_id or None, utility_score, now, now, now,
                ),
            )
            await self.conn.execute(
                "INSERT INTO vec_embers(ember_id, embedding) VALUES (?, ?)",
                (ember_id, embedding),
            )
            await self.conn.commit()
        except Exception:
            await self.conn.rollback()
            raise

    async def get_ember(self, ember_id: str) -> Optional[dict]:
        """Return full ember row as dict, or None."""
        cursor = await self.conn.execute(
            "SELECT * FROM embers WHERE ember_id = ?", (ember_id,)
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        d = dict(row)
        d["tags"] = json.loads(d["tags"]) if isinstance(d["tags"], str) else d["tags"]
        d["is_stale"] = bool(d["is_stale"])
        return d

    # Allowed columns for dynamic update (whitelist against injection)
    _EMBER_UPDATE_FIELDS = frozenset({
        "name", "content", "tags", "cell_id", "importance", "source",
        "source_path", "agent", "session_id", "supersedes_id", "superseded_by_id",
        "is_stale", "stale_reason", "last_accessed_at", "access_count",
        "utility_score", "shadow_load", "shadowed_by", "shadow_updated_at",
        "updated_at",
    })

    async def update_ember(self, ember_id: str, content_changed: bool = False, **fields) -> bool:
        """Update specific fields on an ember. Bumps updated_at if content_changed."""
        if not fields and not content_changed:
            return False

        # Whitelist check
        bad = set(fields) - self._EMBER_UPDATE_FIELDS
        if bad:
            raise ValueError(f"Disallowed update fields: {bad}")

        if content_changed:
            fields["updated_at"] = _now_iso()

        # Serialize tags if present
        if "tags" in fields and isinstance(fields["tags"], list):
            fields["tags"] = json.dumps(fields["tags"])

        # Convert bools to int for SQLite
        if "is_stale" in fields:
            fields["is_stale"] = int(fields["is_stale"])

        set_clause = ", ".join(f"{k} = ?" for k in fields)
        values = list(fields.values()) + [ember_id]

        result = await self.conn.execute(
            f"UPDATE embers SET {set_clause} WHERE ember_id = ?", values
        )
        await self.conn.commit()
        return result.rowcount > 0

    async def increment_access(self, ember_id: str) -> None:
        """Atomically bump access_count and last_accessed_at in SQL."""
        await self.conn.execute(
            "UPDATE embers SET access_count = access_count + 1, last_accessed_at = ? WHERE ember_id = ?",
            (_now_iso(), ember_id),
        )
        await self.conn.commit()

    async def delete_ember(self, ember_id: str) -> bool:
        """Delete ember + vector + edges."""
        await self.conn.execute(
            "DELETE FROM vec_embers WHERE ember_id = ?", (ember_id,)
        )
        await self.conn.execute(
            "DELETE FROM edges WHERE source_id = ? OR target_id = ?",
            (ember_id, ember_id),
        )
        result = await self.conn.execute(
            "DELETE FROM embers WHERE ember_id = ?", (ember_id,)
        )
        await self.conn.commit()
        return result.rowcount > 0

    async def list_embers(
        self,
        tag: str = "",
        importance: str = "",
        is_stale: Optional[bool] = None,
        limit: int = 0,
    ) -> list[dict]:
        """List embers with optional filters."""
        sql = "SELECT * FROM embers WHERE 1=1"
        params: list = []

        if tag:
            sql += " AND tags LIKE ?"
            params.append(f'%"{tag}"%')
        if importance:
            sql += " AND importance = ?"
            params.append(importance)
        if is_stale is not None:
            sql += " AND is_stale = ?"
            params.append(int(is_stale))

        sql += " ORDER BY created_at DESC"
        if limit > 0:
            sql += " LIMIT ?"
            params.append(limit)

        cursor = await self.conn.execute(sql, params)
        rows = await cursor.fetchall()
        result = []
        for row in rows:
            d = dict(row)
            d["tags"] = json.loads(d["tags"]) if isinstance(d["tags"], str) else d["tags"]
            d["is_stale"] = bool(d["is_stale"])
            result.append(d)
        return result

    async def count_embers(self) -> int:
        cursor = await self.conn.execute("SELECT COUNT(*) FROM embers")
        row = await cursor.fetchone()
        return row[0]

    # ══════════════════════════════════════════════════════════════════════════
    #  VECTOR SEARCH
    # ══════════════════════════════════════════════════════════════════════════

    async def search_embers(
        self, embedding: bytes, k: int = 10
    ) -> list[tuple[str, float]]:
        """KNN on vec_embers. Returns [(ember_id, l2_distance), ...]."""
        # Clamp k to actual count
        cursor = await self.conn.execute("SELECT COUNT(*) FROM vec_embers")
        total = (await cursor.fetchone())[0]
        if total == 0:
            return []
        k = min(k, total)

        cursor = await self.conn.execute(
            """SELECT ember_id, distance
               FROM vec_embers
               WHERE embedding MATCH ? AND k = ?
               ORDER BY distance""",
            (embedding, k),
        )
        return [(row[0], row[1]) for row in await cursor.fetchall()]

    async def search_archive(
        self, embedding: bytes, k: int = 10
    ) -> list[tuple[str, float]]:
        """KNN on vec_archive. Returns [(chunk_id, l2_distance), ...]."""
        cursor = await self.conn.execute("SELECT COUNT(*) FROM vec_archive")
        total = (await cursor.fetchone())[0]
        if total == 0:
            return []
        k = min(k, total)

        cursor = await self.conn.execute(
            """SELECT chunk_id, distance
               FROM vec_archive
               WHERE embedding MATCH ? AND k = ?
               ORDER BY distance""",
            (embedding, k),
        )
        return [(row[0], row[1]) for row in await cursor.fetchall()]

    async def assign_cell(self, embedding: bytes) -> int:
        """Find nearest centroid cell_id."""
        cursor = await self.conn.execute("SELECT COUNT(*) FROM vec_centroids")
        total = (await cursor.fetchone())[0]
        if total == 0:
            return -1

        cursor = await self.conn.execute(
            """SELECT cell_id, distance
               FROM vec_centroids
               WHERE embedding MATCH ? AND k = 1
               ORDER BY distance""",
            (embedding,),
        )
        row = await cursor.fetchone()
        return row[0] if row else -1

    async def ember_vector_count(self) -> int:
        cursor = await self.conn.execute("SELECT COUNT(*) FROM vec_embers")
        return (await cursor.fetchone())[0]

    async def update_ember_vector(self, ember_id: str, embedding: bytes) -> None:
        """Replace the vector for an ember (used when content changes)."""
        await self.conn.execute(
            "DELETE FROM vec_embers WHERE ember_id = ?", (ember_id,)
        )
        await self.conn.execute(
            "INSERT INTO vec_embers(ember_id, embedding) VALUES (?, ?)",
            (ember_id, embedding),
        )
        await self.conn.commit()

    async def recompute_centroids(self, k_cells: int = 16, max_iter: int = 20) -> dict:
        """Recompute Voronoi centroids via k-means on actual ember vectors.

        Reads all vectors from vec_embers, runs k-means clustering,
        replaces vec_centroids with data-derived centroids.
        Returns stats about the clustering.
        """
        import numpy as np

        # Read all vectors from vec_embers
        cursor = await self.conn.execute("SELECT ember_id, embedding FROM vec_embers")
        rows = await cursor.fetchall()
        if len(rows) < k_cells:
            return {
                "status": "skipped",
                "message": f"Need at least {k_cells} embers for {k_cells} centroids, have {len(rows)}",
            }

        vectors = np.array(
            [np.frombuffer(row[1], dtype=np.float32) for row in rows]
        )  # (N, dim)

        # K-means clustering
        rng = np.random.default_rng(seed=42)
        # Initialize with random sample of actual vectors (k-means++)
        indices = rng.choice(len(vectors), size=k_cells, replace=False)
        centroids = vectors[indices].copy()

        for iteration in range(max_iter):
            # Assign each vector to nearest centroid
            # L2 distance: ||v - c||^2 = ||v||^2 + ||c||^2 - 2*v·c
            # For L2-normalized vectors: = 2 - 2*v·c
            dots = vectors @ centroids.T  # (N, k_cells)
            assignments = np.argmax(dots, axis=1)  # nearest = highest dot product

            # Recompute centroids as mean of assigned vectors
            new_centroids = np.zeros_like(centroids)
            for i in range(k_cells):
                mask = assignments == i
                if mask.any():
                    new_centroids[i] = vectors[mask].mean(axis=0)
                else:
                    # Empty cluster — reinitialize from random vector
                    new_centroids[i] = vectors[rng.integers(len(vectors))]

            # L2 normalize
            norms = np.linalg.norm(new_centroids, axis=1, keepdims=True)
            new_centroids /= np.clip(norms, 1e-12, None)

            # Check convergence
            shift = np.max(np.abs(new_centroids - centroids))
            centroids = new_centroids
            if shift < 1e-6:
                break

        # Replace vec_centroids
        await self.conn.execute("DELETE FROM vec_centroids")
        for i, vec in enumerate(centroids):
            await self.conn.execute(
                "INSERT INTO vec_centroids(cell_id, embedding) VALUES (?, ?)",
                (i, vec.astype(np.float32).tobytes()),
            )

        # Update cell assignments on all embers
        ember_ids = [row[0] for row in rows]
        assignments_final = np.argmax(vectors @ centroids.T, axis=1)
        for eid, cell_id in zip(ember_ids, assignments_final):
            await self.conn.execute(
                "UPDATE embers SET cell_id = ? WHERE ember_id = ?",
                (int(cell_id), eid),
            )

        await self.conn.commit()

        # Compute cluster sizes
        cell_sizes = {}
        for c in assignments_final:
            cell_sizes[int(c)] = cell_sizes.get(int(c), 0) + 1

        return {
            "status": "ok",
            "embers": len(rows),
            "cells": k_cells,
            "iterations": iteration + 1,
            "cell_sizes": cell_sizes,
        }

    # ══════════════════════════════════════════════════════════════════════════
    #  ARCHIVE CRUD
    # ══════════════════════════════════════════════════════════════════════════

    async def archive_store(
        self,
        chunk_id: str,
        summary: str,
        content: str,
        tags: list[str],
        project: str = "",
        domain: str = "",
        chunk_type: str = "general",
        from_agent: str = "",
        for_agent: str = "",
        entity: str = "",
        category: str = "",
        reply_to: str = "",
        status: str = "active",
        embedding: Optional[bytes] = None,
    ) -> dict:
        """Store archive chunk + optional vector."""
        if len(content) > MAX_CONTENT_SIZE:
            return {"status": "error", "message": f"Content too large ({len(content)} chars)"}

        if chunk_type not in CHUNK_TYPES:
            chunk_type = "general"

        content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()

        # Dedup check
        cursor = await self.conn.execute(
            "SELECT chunk_id, summary FROM archive WHERE content_hash = ? LIMIT 1",
            (content_hash,),
        )
        existing = await cursor.fetchone()
        if existing:
            return {
                "status": "duplicate",
                "existing_id": existing[0],
                "summary": existing[1],
            }

        if not summary:
            first_line = next(
                (l.strip() for l in content.splitlines() if l.strip()), ""
            )
            summary = (first_line[:97] + "...") if len(first_line) > 100 else first_line or "(no summary)"

        now = _now_iso()
        tokens_estimate = len(content) // 4

        try:
            await self.conn.execute(
                """INSERT INTO archive (
                    chunk_id, summary, content, tags, project, domain, chunk_type,
                    from_agent, for_agent, entity, category, reply_to, status,
                    tokens_estimate, content_hash, access_count, last_accessed_at, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?)""",
                (
                    chunk_id, summary, content, json.dumps(tags), project, domain,
                    chunk_type, from_agent, for_agent, entity, category, reply_to,
                    status, tokens_estimate, content_hash, now, now,
                ),
            )

            # Insert vector if provided
            if embedding is not None:
                await self.conn.execute(
                    "INSERT INTO vec_archive(chunk_id, embedding) VALUES (?, ?)",
                    (chunk_id, embedding),
                )

            await self.conn.commit()
        except Exception:
            await self.conn.rollback()
            raise
        return {"status": "ok", "id": chunk_id, "summary": summary}

    async def archive_read(
        self, chunk_id: str, start: int = 0, end: int = 0
    ) -> Optional[dict]:
        """Read archive chunk, optionally a line range."""
        # Try exact match, then prefix
        cursor = await self.conn.execute(
            "SELECT * FROM archive WHERE chunk_id = ?", (chunk_id,)
        )
        row = await cursor.fetchone()

        if row is None:
            if not CHUNK_ID_RE.match(chunk_id):
                return None
            cursor = await self.conn.execute(
                "SELECT * FROM archive WHERE chunk_id LIKE ? ORDER BY chunk_id",
                (chunk_id + "%",),
            )
            rows = await cursor.fetchall()
            if len(rows) == 1:
                row = rows[0]
                chunk_id = row["chunk_id"]
            elif len(rows) > 1:
                return {
                    "meta": {},
                    "content": f"Ambiguous prefix '{chunk_id}', matches: "
                    + str([r["chunk_id"] for r in rows]),
                }
            else:
                return None

        meta = dict(row)
        meta["tags"] = json.loads(meta["tags"]) if isinstance(meta["tags"], str) else meta["tags"]
        content = meta.pop("content")

        # Line range selection
        if start > 0 or end > 0:
            lines = content.splitlines()
            total = len(lines)
            start = max(1, start)
            end = min(end, total) if end > 0 else total
            selected = lines[start - 1:end]
            content = f"[Lines {start}-{end} of {total}]\n" + "\n".join(selected)

        # Update access stats
        await self.conn.execute(
            """UPDATE archive SET access_count = access_count + 1,
               last_accessed_at = ? WHERE chunk_id = ?""",
            (_now_iso(), meta["chunk_id"]),
        )
        await self.conn.commit()

        return {"meta": meta, "content": content}

    async def archive_search_fts(
        self,
        query: str,
        limit: int = 5,
        project: str = "",
        domain: str = "",
        chunk_type: str = "",
        date_from: str = "",
        date_to: str = "",
    ) -> list[dict]:
        """FTS5 search on archive. Returns ranked results.

        Uses query_prep.prepare_bm25_query (2026-05-15) for proper stop-word
        handling + OR semantics with BM25 ranking. Previously used
        _fts5_escape which had two problems: no stop words filtered (so
        common-word queries like "is there a fiber bundle" matched almost
        everything), and AND semantics (so any term miss zeroed the result
        set). The new behavior: broader recall, ranked by BM25 weighted as
        before (summary 10x, content 1x, tags 5x, project 5x).
        """
        from ember4.query_prep import prepare_bm25_query, get_corpus_stopwords
        corpus_stops = await get_corpus_stopwords(self)
        safe_query = prepare_bm25_query(query, corpus_stopwords=corpus_stops)
        if not safe_query:
            return []

        from_dt = _parse_date(date_from)
        to_dt = _parse_date(date_to)

        try:
            # Use bm25 weighted scoring: summary 10x, content 1x, tags 5x, project 5x
            cursor = await self.conn.execute(
                """SELECT a.chunk_id, a.summary, a.project, a.domain,
                          a.chunk_type, a.tags, a.created_at,
                          bm25(archive_fts, 0, 10.0, 1.0, 5.0, 5.0) as score
                   FROM archive_fts f
                   JOIN archive a ON f.chunk_id = a.chunk_id
                   WHERE archive_fts MATCH ? AND a.status != 'archived'
                   ORDER BY score
                   LIMIT ?""",
                (safe_query, limit * 3),
            )
            rows = await cursor.fetchall()
        except Exception:
            return []

        results = []
        for row in rows:
            d = dict(row)
            d["tags"] = json.loads(d["tags"]) if isinstance(d["tags"], str) else d["tags"]

            if project and d.get("project") != project:
                continue
            if domain and d.get("domain") != domain:
                continue
            if chunk_type and d.get("chunk_type") != chunk_type:
                continue

            if from_dt or to_dt:
                created = _parse_date(d.get("created_at", ""))
                if created:
                    if from_dt and created < from_dt:
                        continue
                    if to_dt and created > to_dt:
                        continue

            results.append(d)
            if len(results) >= limit:
                break

        return results

    async def archive_grep(
        self,
        pattern: str,
        project: str = "",
        domain: str = "",
        chunk_type: str = "",
        chunk_id: str = "",
        date_from: str = "",
        date_to: str = "",
        limit: int = 20,
        context_lines: int = 2,
        fuzzy: bool = False,
        fuzzy_threshold: int = 80,
    ) -> dict:
        """Regex or fuzzy search on archive content.

        Optimization: uses FTS5 as a pre-filter when the pattern contains
        searchable words, loading content only for candidate chunks.
        """
        from_dt = _parse_date(date_from)
        to_dt = _parse_date(date_to)

        # --- FTS5 pre-filter: extract plain words from pattern ---
        # If the pattern has searchable words, narrow candidates via FTS5 first.
        # Uses OR semantics so regex alternation (foo|bar) doesn't miss chunks.
        # Skips pre-filter for patterns with no extractable words.
        fts_candidate_ids: set[str] | None = None
        if not fuzzy and not chunk_id:
            words = re.findall(r"[a-zA-Z0-9]{3,}", pattern)
            if words:
                fts_query = " OR ".join(f'"{w}"' for w in words)
                if fts_query:
                    try:
                        cursor = await self.conn.execute(
                            "SELECT chunk_id FROM archive_fts WHERE archive_fts MATCH ? LIMIT 200",
                            (fts_query,),
                        )
                        fts_candidate_ids = {row[0] for row in await cursor.fetchall()}
                    except Exception:
                        pass  # Fall back to full scan

        # Build metadata filter — only select content for candidates
        sql = "SELECT chunk_id, summary, content, project, chunk_type, created_at FROM archive WHERE status != 'archived'"
        params: list = []

        if chunk_id:
            sql += " AND (chunk_id = ? OR chunk_id LIKE ?)"
            params.extend([chunk_id, chunk_id + "%"])
        elif fts_candidate_ids is not None:
            if not fts_candidate_ids:
                return {"status": "ok", "pattern": pattern, "match_count": 0, "matches": []}
            placeholders = ",".join("?" for _ in fts_candidate_ids)
            sql += f" AND chunk_id IN ({placeholders})"
            params.extend(fts_candidate_ids)
        if project:
            sql += " AND project = ?"
            params.append(project)
        if domain:
            sql += " AND domain = ?"
            params.append(domain)
        if chunk_type:
            sql += " AND chunk_type = ?"
            params.append(chunk_type)
        if from_dt:
            sql += " AND created_at >= ?"
            params.append(from_dt.strftime("%Y-%m-%dT%H:%M:%S"))
        if to_dt:
            sql += " AND created_at <= ?"
            params.append(to_dt.strftime("%Y-%m-%dT%H:%M:%S") + "Z")

        sql += " ORDER BY created_at DESC"

        cursor = await self.conn.execute(sql, params)

        if fuzzy:
            rows = await cursor.fetchall()
            return self._grep_fuzzy(
                rows, pattern, limit, context_lines, fuzzy_threshold
            )

        try:
            rx = re.compile(pattern, re.IGNORECASE | re.MULTILINE)
        except re.error as e:
            return {"status": "error", "message": f"Invalid regex: {e}", "matches": []}

        # Stream rows one at a time instead of fetchall()
        matches = []
        async for row in cursor:
            if len(matches) >= limit:
                break
            content = row["content"]
            lines = content.splitlines()
            match_lines = []
            for i, line in enumerate(lines):
                if rx.search(line):
                    ctx_start = max(0, i - context_lines)
                    ctx_end = min(len(lines), i + context_lines + 1)
                    context = "\n".join(
                        f"{'>' if j == i else ' '} {j+1}: {lines[j]}"
                        for j in range(ctx_start, ctx_end)
                    )
                    match_lines.append({"line_number": i + 1, "context": context})

            if match_lines:
                matches.append({
                    "id": row["chunk_id"],
                    "summary": row["summary"],
                    "project": row["project"],
                    "chunk_type": row["chunk_type"],
                    "match_count": len(match_lines),
                    "matches": match_lines[:10],
                })

        return {"status": "ok", "pattern": pattern, "match_count": len(matches), "matches": matches}

    @staticmethod
    def _grep_fuzzy(rows, pattern, limit, context_lines, threshold) -> dict:
        """Fuzzy text matching using thefuzz."""
        try:
            from thefuzz import fuzz
        except ImportError:
            return {"status": "error", "message": "thefuzz not installed", "matches": []}

        pattern_lower = pattern.lower()
        matches = []

        for row in rows:
            if len(matches) >= limit:
                break
            content = row["content"]
            lines = content.splitlines()
            match_lines = []
            for i, line in enumerate(lines):
                score = fuzz.partial_ratio(pattern_lower, line.lower())
                if score >= threshold:
                    ctx_start = max(0, i - context_lines)
                    ctx_end = min(len(lines), i + context_lines + 1)
                    context = "\n".join(
                        f"{'>' if j == i else ' '} {j+1}: {lines[j]}"
                        for j in range(ctx_start, ctx_end)
                    )
                    match_lines.append({
                        "line_number": i + 1,
                        "fuzzy_score": score,
                        "context": context,
                    })

            if match_lines:
                match_lines.sort(key=lambda m: m["fuzzy_score"], reverse=True)
                matches.append({
                    "id": row["chunk_id"],
                    "summary": row["summary"],
                    "project": row["project"],
                    "chunk_type": row["chunk_type"],
                    "match_count": len(match_lines),
                    "matches": match_lines[:10],
                })

        return {"status": "ok", "pattern": pattern, "fuzzy": True,
                "match_count": len(matches), "matches": matches}

    async def archive_list(
        self,
        project: str = "",
        domain: str = "",
        tag: str = "",
        chunk_type: str = "",
        from_agent: str = "",
        limit: int = 20,
        offset: int = 0,
    ) -> list[dict]:
        """List archive chunks with filters."""
        sql = "SELECT chunk_id, summary, tags, project, domain, chunk_type, from_agent, created_at, tokens_estimate, access_count FROM archive WHERE status != 'archived'"
        params: list = []

        if project:
            sql += " AND project = ?"
            params.append(project)
        if domain:
            sql += " AND domain = ?"
            params.append(domain)
        if tag:
            sql += " AND tags LIKE ?"
            params.append(f'%"{tag}"%')
        if chunk_type:
            sql += " AND chunk_type = ?"
            params.append(chunk_type)
        if from_agent:
            sql += " AND from_agent = ?"
            params.append(from_agent)

        sql += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        cursor = await self.conn.execute(sql, params)
        rows = await cursor.fetchall()
        result = []
        for row in rows:
            d = dict(row)
            d["tags"] = json.loads(d["tags"]) if isinstance(d["tags"], str) else d["tags"]
            result.append(d)
        return result

    async def archive_update(
        self,
        chunk_id: str,
        content: str = "",
        add_tags: Optional[list[str]] = None,
        remove_tags: Optional[list[str]] = None,
        summary: str = "",
        domain: str = "",
        status: str = "",
    ) -> dict:
        """Update archive chunk fields."""
        cursor = await self.conn.execute(
            "SELECT * FROM archive WHERE chunk_id = ?", (chunk_id,)
        )
        row = await cursor.fetchone()
        if row is None:
            return {"status": "error", "message": f"Chunk {chunk_id} not found"}

        updates: dict = {}

        if content:
            updates["content"] = content
            updates["content_hash"] = hashlib.sha256(content.encode()).hexdigest()
            updates["tokens_estimate"] = len(content) // 4
            if not summary:
                first_line = next(
                    (l.strip() for l in content.splitlines() if l.strip()), ""
                )
                summary = (first_line[:97] + "...") if len(first_line) > 100 else first_line

        if summary:
            updates["summary"] = summary
        if domain:
            updates["domain"] = domain
        if status:
            updates["status"] = status

        # Handle tag modifications
        current_tags = json.loads(row["tags"]) if isinstance(row["tags"], str) else row["tags"]
        tag_set = set(current_tags)
        if add_tags:
            tag_set.update(add_tags)
        if remove_tags:
            tag_set.difference_update(remove_tags)
        updates["tags"] = json.dumps(sorted(tag_set))

        if updates:
            set_clause = ", ".join(f"{k} = ?" for k in updates)
            values = list(updates.values()) + [chunk_id]
            await self.conn.execute(
                f"UPDATE archive SET {set_clause} WHERE chunk_id = ?", values
            )
            await self.conn.commit()

        return {"status": "ok", "id": chunk_id}

    async def archive_delete(self, chunk_id: str) -> bool:
        """Delete archive chunk + vector."""
        await self.conn.execute(
            "DELETE FROM vec_archive WHERE chunk_id = ?", (chunk_id,)
        )
        result = await self.conn.execute(
            "DELETE FROM archive WHERE chunk_id = ?", (chunk_id,)
        )
        await self.conn.commit()
        return result.rowcount > 0

    async def archive_next_chunk_id(
        self, project: str, chunk_type: str = "general"
    ) -> str:
        """Generate next sequential chunk ID for today."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        slug = re.sub(r"[^\w]", "_", project.lower())[:30].strip("_") or "general"
        type_slug = re.sub(r"[^\w]", "_", chunk_type.lower())[:30].strip("_") or "general"

        cursor = await self.conn.execute(
            "SELECT COUNT(*) FROM archive WHERE chunk_id LIKE ? AND project = ?",
            (f"{today}%", project),
        )
        count = (await cursor.fetchone())[0]
        return f"{today}_{slug}_{type_slug}_{count + 1:03d}"

    async def archive_retention_preview(self, days: int = 0) -> dict:
        """Preview what would be archived by retention rules."""
        now = datetime.now(timezone.utc)
        cursor = await self.conn.execute(
            "SELECT chunk_id, summary, chunk_type, created_at, access_count FROM archive WHERE status = 'active'"
        )
        rows = await cursor.fetchall()

        to_archive = []
        protected = []

        for row in rows:
            created = _parse_date(row["created_at"])
            if not created:
                continue

            ct = row["chunk_type"]
            retention = days if days > 0 else (
                SUBCONSCIOUS_RETENTION_DAYS if ct == "subconscious"
                else DEFAULT_RETENTION_DAYS
            )

            age_days = (now - created).total_seconds() / 86400.0
            if age_days > retention:
                if row["access_count"] >= 3:
                    protected.append({
                        "id": row["chunk_id"],
                        "age_days": int(age_days),
                        "access_count": row["access_count"],
                        "reason": "frequently accessed",
                    })
                else:
                    to_archive.append({
                        "id": row["chunk_id"],
                        "summary": row["summary"],
                        "age_days": int(age_days),
                        "chunk_type": ct,
                    })

        total_cursor = await self.conn.execute("SELECT COUNT(*) FROM archive")
        total = (await total_cursor.fetchone())[0]

        return {"to_archive": to_archive, "protected": protected, "total_chunks": total}

    async def archive_retention_run(self, days: int = 0, dry_run: bool = True) -> dict:
        """Execute retention: soft-archive old chunks."""
        preview = await self.archive_retention_preview(days)
        if dry_run:
            return {"dry_run": True, **preview}

        archived = []
        now = _now_iso()
        for item in preview["to_archive"]:
            await self.conn.execute(
                "UPDATE archive SET status = 'archived', archived_at = ? WHERE chunk_id = ?",
                (now, item["id"]),
            )
            archived.append(item["id"])

        await self.conn.commit()
        return {"dry_run": False, "archived": archived, "protected": preview["protected"]}

    # ══════════════════════════════════════════════════════════════════════════
    #  KNOWLEDGE GRAPH
    # ══════════════════════════════════════════════════════════════════════════

    async def save_edge(
        self, source_id: str, target_id: str, edge_type: str, weight: float = 0.0
    ) -> None:
        await self.conn.execute(
            """INSERT OR REPLACE INTO edges (source_id, target_id, edge_type, weight, created_at)
               VALUES (?, ?, ?, ?, ?)""",
            (source_id, target_id, edge_type, weight, _now_iso()),
        )
        await self.conn.commit()

    async def get_edges(self, ember_id: str, edge_type: str = "") -> list[dict]:
        if edge_type:
            cursor = await self.conn.execute(
                """SELECT source_id, target_id, edge_type, weight, created_at
                   FROM edges
                   WHERE (source_id = ? OR target_id = ?) AND edge_type = ?""",
                (ember_id, ember_id, edge_type),
            )
        else:
            cursor = await self.conn.execute(
                """SELECT source_id, target_id, edge_type, weight, created_at
                   FROM edges
                   WHERE source_id = ? OR target_id = ?""",
                (ember_id, ember_id),
            )
        return [dict(row) for row in await cursor.fetchall()]

    async def get_neighbors(self, ember_id: str, edge_type: str = "") -> list[str]:
        edges = await self.get_edges(ember_id, edge_type)
        neighbors = set()
        for e in edges:
            if e["source_id"] != ember_id:
                neighbors.add(e["source_id"])
            if e["target_id"] != ember_id:
                neighbors.add(e["target_id"])
        return list(neighbors)

    async def traverse_kg(
        self, start_id: str, depth: int = 2, edge_types: Optional[list[str]] = None
    ) -> set[str]:
        """BFS traversal from start_id up to depth hops."""
        visited: set[str] = {start_id}
        queue: deque = deque([(start_id, 0)])

        while queue:
            current_id, current_depth = queue.popleft()
            if current_depth >= depth:
                continue

            if edge_types:
                neighbors: set[str] = set()
                for et in edge_types:
                    neighbors.update(await self.get_neighbors(current_id, et))
            else:
                neighbors = set(await self.get_neighbors(current_id))

            for nid in neighbors:
                if nid not in visited:
                    visited.add(nid)
                    queue.append((nid, current_depth + 1))

        visited.discard(start_id)
        return visited

    # ══════════════════════════════════════════════════════════════════════════
    #  REGION STATS
    # ══════════════════════════════════════════════════════════════════════════

    async def update_region(
        self, cell_id: int, vitality: float, shadow_accum: float
    ) -> None:
        await self.conn.execute(
            """INSERT OR REPLACE INTO region_stats (cell_id, vitality_score, shadow_accum, last_updated)
               VALUES (?, ?, ?, ?)""",
            (cell_id, vitality, shadow_accum, _now_iso()),
        )
        await self.conn.commit()

    async def get_all_region_stats(self) -> dict[int, dict]:
        cursor = await self.conn.execute(
            "SELECT cell_id, vitality_score, shadow_accum, last_updated FROM region_stats"
        )
        result = {}
        for row in await cursor.fetchall():
            result[row["cell_id"]] = dict(row)
        return result

    # ══════════════════════════════════════════════════════════════════════════
    #  METRICS LOG
    # ══════════════════════════════════════════════════════════════════════════

    async def log_metric(
        self, metric_type: str, value: float, details: Optional[dict] = None
    ) -> None:
        await self.conn.execute(
            "INSERT INTO metrics_log (timestamp, metric_type, value, details) VALUES (?, ?, ?, ?)",
            (_now_iso(), metric_type, value, json.dumps(details) if details else None),
        )
        await self.conn.commit()

    async def get_metric_history(self, metric_type: str, limit: int = 10) -> list[dict]:
        cursor = await self.conn.execute(
            """SELECT id, timestamp, metric_type, value, details
               FROM metrics_log WHERE metric_type = ?
               ORDER BY timestamp DESC LIMIT ?""",
            (metric_type, limit),
        )
        results = []
        for row in await cursor.fetchall():
            d = dict(row)
            if d["details"]:
                try:
                    d["details"] = json.loads(d["details"])
                except (json.JSONDecodeError, TypeError):
                    pass
            results.append(d)
        return results

    # ══════════════════════════════════════════════════════════════════════════
    #  RECALL LOG
    # ══════════════════════════════════════════════════════════════════════════

    async def log_recall(
        self, session_id: str, ember_id: str, event_type: str
    ) -> None:
        try:
            await self.conn.execute(
                "INSERT INTO recall_log (session_id, ember_id, event_type) VALUES (?, ?, ?)",
                (session_id, ember_id, event_type),
            )
            await self.conn.commit()
        except Exception:
            pass  # Non-critical; don't break retrieval

    # ══════════════════════════════════════════════════════════════════════════
    #  UPDATE LOG (ember-4 engagement signal)
    # ══════════════════════════════════════════════════════════════════════════

    async def log_update(
        self,
        ember_id: str,
        event_type: str,
        origin: str = "agent",
        content_changed: bool = False,
        session_id: str = "",
        details: str = "",
    ) -> None:
        """Record an engagement event for ember-4's importance-signal pipeline.

        event_type ∈ {'edit', 'contradict_old', 'contradict_new', 'edge_form',
                      'tag_update', 'auto_enrich'}
        origin     ∈ {'user', 'agent', 'auto_system', 'migration'}

        The contribution filter (applied at recall time, not here) decides
        whether each event pumps the engagement signal. System-initiated
        writes (auto_system, migration) are recorded for audit but do NOT
        contribute to importance — preventing a feedback loop into
        engagement scoring.

        content_changed=True only when the body text actually changed; False
        for metadata-only updates (tags, utility_score). Used so a human
        adding a tag doesn't get scored the same as a substantive edit.

        Idempotent failure: like log_recall, errors are swallowed so a
        broken log table can't break a working write path.
        """
        try:
            await self.conn.execute(
                """INSERT INTO update_log
                   (ember_id, event_type, origin, content_changed,
                    session_id, details)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    ember_id,
                    event_type,
                    origin,
                    1 if content_changed else 0,
                    session_id or "",
                    details or "",
                ),
            )
            await self.conn.commit()
        except Exception as e:
            # Non-critical; don't break the write path. Surface to stderr.
            import sys as _sys
            print(f"[ember-4] log_update suppressed: {e}", file=_sys.stderr)

    # ══════════════════════════════════════════════════════════════════════════
    #  CONFIG
    # ══════════════════════════════════════════════════════════════════════════

    async def get_config(self, key: str) -> Optional[str]:
        cursor = await self.conn.execute(
            "SELECT value FROM config WHERE key = ?", (key,)
        )
        row = await cursor.fetchone()
        if row:
            return json.loads(row[0])
        return None

    async def set_config(self, key: str, value) -> None:
        await self.conn.execute(
            "INSERT OR REPLACE INTO config(key, value) VALUES (?, ?)",
            (key, json.dumps(value)),
        )
        await self.conn.commit()
