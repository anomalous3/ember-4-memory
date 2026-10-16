"""Configuration constants and path resolution for Ember 4.

Ember 4 is a development fork of Ember 3 testing:
- fiber-aware partition (replaces k-means/Voronoi)
- update-activity importance signal (replaces shadow-on-insert decay)
- temporal proximity in scoring metric

By default Ember 4 reads from ~/.ember4/ember4.db (a frozen copy of
ember3.db). Set EMBER4_DATA_DIR to override. Set EMBER4_READ_ONLY=1
during development to prevent any writes to the frozen data — shadow-on-
insert mutates older neighbors, so we cannot let development runs touch
the source data.
"""

import os
import sys
from pathlib import Path

# ── Vector / Model ────────────────────────────────────────────────────────────
DIMENSION = 384
K_CELLS = 32
HF_REPO = "sentence-transformers/all-MiniLM-L6-v2"

# ── Freeze Flag ──────────────────────────────────────────────────────────────
# When True the database is opened read-only and any write attempt raises
# EmberFrozenError. Toggle via EMBER4_READ_ONLY env var.
READ_ONLY = os.environ.get("EMBER4_READ_ONLY", "").strip() not in ("", "0", "false", "False")

# ── HESTIA Scoring ────────────────────────────────────────────────────────────
SIMILARITY_THRESHOLD = 0.4
SHADOW_DELTA = 0.3
SHADOW_EPSILON = 0.05
SHADOW_GAMMA = 2.0
NOSTALGIA_ALPHA = 0.1
SHADOW_K = 10

# ── Project Scoping ──────────────────────────────────────────────────────────
PROJECT_BOOST = 0.5
PROJECT_PENALTY = 0.7

# ── Utility Feedback ──────────────────────────────────────────────────────────
UTILITY_WEIGHT = 0.15

# ── Recency (anti-ghost) ──────────────────────────────────────────────────────
# Demote embers whose last *meaningful touch* (creation or content edit/contradict,
# NOT mere access) is old, so a conceptually-similar but architecturally-defunct
# past attempt doesn't rank as if current. recency_factor = 1 - strength*(1 - exp(-age/tau)).
# strength=0 disables it (deep_recall / wonder pass 0); the floor (1-strength) lets a
# very-strong cosine match on an old-but-specific gem still surface.
RECENCY_STRENGTH = 0.75     # default for ember_recall; heavy bias suits ~session/2-days
                            # pace. Per-project override via db config
                            # key "recency_strength:<project>". Revisit if pace slows.
RECENCY_TAU_DAYS = 21.0     # ~3 sessions; older embers useful mainly if specific/general
# Tags that EARN exemption from recency demotion (timeless principles only — NOT
# wonder/meta/dream, which are speculative). Coupled discipline: an exempt foundation
# ember that is superseded MUST be ember_contradict'd, else it becomes a permanent
# ghost (exemption removes the recency safety net). Earned going forward, not retroactive.
RECENCY_EXEMPT_TAGS = ("foundation", "principle")

# ── Shadow Archive ────────────────────────────────────────────────────────────
SHADOW_ARCHIVE_THRESHOLD = 0.95

# ── Archive ───────────────────────────────────────────────────────────────────
MAX_CONTENT_SIZE = 2 * 1024 * 1024  # 2 MB
DEFAULT_RETENTION_DAYS = 30
SUBCONSCIOUS_RETENTION_DAYS = 14

CHUNK_TYPES = {
    "session", "debug", "snapshot", "compact-export",
    "subconscious", "reference", "general", "dream-log",
}

# ── Decay Half-Lives (days) by importance ─────────────────────────────────────
DECAY_HALF_LIVES = {
    "fact": 365.0,
    "decision": 30.0,
    "preference": 60.0,
    "context": 7.0,
    "learning": 90.0,
}

# ── Preview ───────────────────────────────────────────────────────────────────
PREVIEW_CHARS = 150


# ── Path Resolution ──────────────────────────────────────────────────────────

def get_data_dir() -> Path:
    """Determine the data directory, platform-aware.

    Resolution order:
    1. EMBER4_DATA_DIR env var (ember-4 specific override)
    2. ~/.ember4 (default; isolated from ember-3's ~/.ember)
    3. %APPDATA%/ember4 on Windows
    """
    env = os.environ.get("EMBER4_DATA_DIR", "").strip()
    if env:
        return Path(env)
    if sys.platform == "win32":
        appdata = os.environ.get("APPDATA", "")
        if appdata:
            return Path(appdata) / "ember4"
    return Path.home() / ".ember4"


def get_db_path() -> Path:
    """Path to the unified SQLite database (ember-4 frozen copy)."""
    return get_data_dir() / "ember4.db"


def get_model_dir() -> Path:
    """Path to the ONNX model directory.

    Symlinked from ~/.ember/models so we share the embedder with ember-3
    (same MiniLM-L6-v2; isolates clustering/scoring changes for attribution).
    """
    return get_data_dir() / "models" / "all-MiniLM-L6-v2"


class EmberFrozenError(RuntimeError):
    """Raised when a write is attempted against a frozen ember-4 database."""
