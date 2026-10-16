"""ember-4 tag-edge augmentation: build project-cohort adjacency from name patterns.

The bare kNN graph under-samples project-cohort adjacency. Embers from the
same project can end up 2+ hops apart in clustering-vector space even though
they are working on the same problem, because the embedder picks up on
surface-format differences (LEARNING vs [theme] vs raw observations).

The fix: extract pseudo-tags from ember names (project regexes, type prefixes
like "[dream]" or "LEARNING", version strings like "v92") and build edges
between embers that share rare pseudo-tags, weighted by tag IDF.

Project patterns are yours, so they live in a JSON file rather than here:
``$EMBER4_DATA_DIR/tag_patterns.json`` (default ``~/.ember4/``), mapping tag
name to a case-insensitive regex. See ``tag_patterns.example.json``. With no
file, only type-prefix and version pseudo-tags are extracted.

Stored as edge_type='tag_cohort' in the existing edges table. Idempotent —
clears prior tag_cohort edges before re-inserting.

This is a separate edge type with its own row in EDGE_TYPE_WEIGHTS, so we
can ablate the tag-cohort contribution by zeroing its weight.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time
from collections import defaultdict
from typing import Optional

# Additive writes; turn off freeze.
os.environ["EMBER4_READ_ONLY"] = "0"

import numpy as np  # noqa: E402

from ember4.config import get_data_dir  # noqa: E402
from ember4.db import Database  # noqa: E402

EDGE_TYPE = "tag_cohort"

# Pseudo-tag extraction rules — name-pattern → tag(s)
# IDF-weighted; rare tags carry more signal than common ones.

def load_project_patterns() -> dict[str, re.Pattern]:
    """Project pseudo-tag regexes from tag_patterns.json in the data dir."""
    path = get_data_dir() / "tag_patterns.json"
    if not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {tag: re.compile(rx, re.IGNORECASE) for tag, rx in raw.items()}


PROJECT_PATTERNS = load_project_patterns()

TYPE_PREFIX_PATTERNS = {
    "type:learning":     re.compile(r"^\s*LEARNING\b"),
    "type:decision":     re.compile(r"^\s*DECISION\b"),
    "type:fact":         re.compile(r"^\s*FACT\b"),
    "type:result":       re.compile(r"^\s*RESULT\b"),
    "type:dream":        re.compile(r"^\s*\[dream\]"),
    "type:wonder":       re.compile(r"^\s*\[wonder\]"),
    "type:theme":        re.compile(r"^\s*\[theme\]"),
    "type:anti-pattern": re.compile(r"^\s*ANTI-PATTERN\b"),
    "type:major-insight": re.compile(r"^\s*MAJOR\b"),
}

VERSION_PATTERN = re.compile(r"\b[vV](\d+[a-z]?)\b")  # v49, v92, v95a, etc.


def extract_pseudo_tags(name: str, content_head: str = "") -> set[str]:
    """Extract pseudo-tags from name (+ first 200 chars of content for context)."""
    text = (name or "") + " " + (content_head or "")[:200]
    tags: set[str] = set()

    for tag, pat in PROJECT_PATTERNS.items():
        if pat.search(text):
            tags.add(tag)

    for tag, pat in TYPE_PREFIX_PATTERNS.items():
        if pat.search(name or ""):
            tags.add(tag)

    # Versions — capture iteration families (v49, v92, ...). We add both the specific version
    # ("ver:v92") and a coarse bucket ("ver:vNN") so embers from the same
    # version are tightly grouped AND any-version-of-anything has some bridge.
    for m in VERSION_PATTERN.finditer(text):
        tags.add(f"ver:v{m.group(1)}")

    return tags


async def load_real_tags_too(db: Database) -> dict[str, set[str]]:
    """Merge real ember tags into the pseudo-tag set, scoped by useful tags only.

    We exclude generic tags like auto-captured, learning (already covered by type:),
    context, dream, wonder, theme (already covered by type: prefix matches).
    Real tags that ARE useful: project:foo, anything with a colon, rare custom tags.
    """
    GENERIC_DROP = {
        "auto-captured", "learning", "context", "dream", "wonder", "theme",
        "bridge", "subconscious", "decision", "fact", "session",
        "dream-generated", "synthesis-theme", "synthesis",
    }
    out: dict[str, set[str]] = {}
    cur = await db.conn.execute("SELECT ember_id, tags FROM embers WHERE is_stale = 0")
    for r in await cur.fetchall():
        tags = json.loads(r["tags"])
        useful = {t for t in tags if t not in GENERIC_DROP and len(t) <= 60}
        if useful:
            out[r["ember_id"]] = useful
    return out


async def main_async(args: argparse.Namespace) -> int:
    db = Database()
    await db.connect()
    assert not db._read_only, "freeze must be off for graph augmentation"

    print(f"db_path: {db.db_path}")

    # ─── Step 1: extract pseudo-tags per ember ──────────────────────────
    cur = await db.conn.execute(
        "SELECT ember_id, name, content FROM embers WHERE is_stale = 0 ORDER BY ember_id"
    )
    rows = await cur.fetchall()
    n = len(rows)

    real_tags = await load_real_tags_too(db)

    ember_tags: dict[str, set[str]] = {}
    for r in rows:
        pseudo = extract_pseudo_tags(r["name"], r["content"])
        real = real_tags.get(r["ember_id"], set())
        ember_tags[r["ember_id"]] = pseudo | real

    # Tag frequency
    tag_df: dict[str, int] = defaultdict(int)
    for tags in ember_tags.values():
        for t in tags:
            tag_df[t] += 1

    # IDF: log(N / df) — rarer tags carry more signal
    log_n = float(np.log(n))
    tag_idf = {t: log_n - float(np.log(df)) for t, df in tag_df.items() if df > 0}

    n_tagged = sum(1 for v in ember_tags.values() if v)
    print(f"Embers with at least one pseudo-tag: {n_tagged}/{n}")
    print(f"Unique pseudo-tags: {len(tag_df)}")

    # Top tags by usage
    print("\nTop 15 pseudo-tags by frequency:")
    for t, df in sorted(tag_df.items(), key=lambda x: -x[1])[:15]:
        print(f"  {df:>4}  idf={tag_idf[t]:.2f}  {t}")

    # ─── Step 2: build inverted index, compute pairwise weights, top-K ──
    tag_to_embers: dict[str, list[str]] = defaultdict(list)
    for eid, tags in ember_tags.items():
        for t in tags:
            tag_to_embers[t].append(eid)

    # For each ember, score every co-occurring ember by sum-of-shared-tag-IDF.
    # Skip ultra-common tags (df > N/3) — they don't carry adjacency signal.
    DF_CAP = n // 3
    print(f"\nIgnoring tags with df > {DF_CAP} (too common to discriminate)")
    useful_tags = {t for t, df in tag_df.items() if df <= DF_CAP and df >= 2}
    print(f"Tags retained as edge sources: {len(useful_tags)} of {len(tag_df)}")

    edges_per_ember: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for tag in useful_tags:
        members = tag_to_embers[tag]
        idf = tag_idf[tag]
        # Add weight to every (a, b) pair in this tag's member set.
        # For large member sets this is O(M^2); useful_tags filter caps M.
        for i, a in enumerate(members):
            for b in members[i + 1 :]:
                edges_per_ember[a][b] += idf
                edges_per_ember[b][a] += idf

    # Keep top-K per ember
    K = args.top_k
    edge_rows: list[tuple] = []
    for eid, others in edges_per_ember.items():
        ranked = sorted(others.items(), key=lambda kv: -kv[1])[:K]
        for other_id, w in ranked:
            edge_rows.append((eid, other_id, w))

    print(f"\nTotal directed edges to insert (top-{K} per node): {len(edge_rows)}")

    # Normalize edge weights to [0, 1] for compatibility with other types
    if edge_rows:
        max_w = max(r[2] for r in edge_rows)
        edge_rows = [(s, t, w / max_w) for s, t, w in edge_rows]
        print(f"Max raw weight: {max_w:.3f}; normalized to [0, 1]")

    if args.dry_run:
        # Show example: the ember with the most tag-cohort neighbors
        TARGET = max(edges_per_ember, key=lambda e: (len(edges_per_ember[e]), e),
                     default=None)
        target_neighbors = sorted(
            edges_per_ember.get(TARGET, {}).items(), key=lambda kv: -kv[1]
        )[:10]
        print(f"\nTarget ember tag-cohort neighbors (top 10, raw IDF weight):")
        for nid, w in target_neighbors:
            cur = await db.conn.execute("SELECT name FROM embers WHERE ember_id = ?", (nid,))
            name = (await cur.fetchone())["name"]
            print(f"  w={w:.3f}  {name[:65]}")
        print(f"\nTarget pseudo-tags: {ember_tags.get(TARGET, set())}")
        print("\n(dry-run; no edges written)")
        await db.close()
        return 0

    # ─── Step 3: clear prior tag_cohort edges + insert new ──────────────
    print(f"\nClearing prior '{EDGE_TYPE}' edges...")
    await db.conn.execute("DELETE FROM edges WHERE edge_type = ?", (EDGE_TYPE,))

    print(f"Inserting {len(edge_rows)} edges...")
    t0 = time.time()
    BATCH = 500
    n_ins = 0
    for k in range(0, len(edge_rows), BATCH):
        chunk = edge_rows[k : k + BATCH]
        await db.conn.executemany(
            "INSERT OR REPLACE INTO edges "
            "(source_id, target_id, edge_type, weight) VALUES (?, ?, ?, ?)",
            [(s, t, EDGE_TYPE, w) for s, t, w in chunk],
        )
        n_ins += len(chunk)
    await db.conn.commit()
    print(f"  inserted {n_ins} in {time.time()-t0:.2f}s")

    # ─── Step 4: verify ─────────────────────────────────────────────────
    cur = await db.conn.execute(
        "SELECT edge_type, COUNT(*) FROM edges GROUP BY edge_type ORDER BY edge_type"
    )
    print("\nEdge type counts:")
    for r in await cur.fetchall():
        print(f"  {r[0]:<28} {r[1]}")

    await db.close()
    print("\nDone. Set EMBER4_READ_ONLY=1 for normal use.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--top-k", type=int, default=15,
                        help="top-K tag-cohort neighbors per ember (default 15)")
    parser.add_argument("--dry-run", action="store_true",
                        help="compute but don't write")
    args = parser.parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    sys.exit(main())
