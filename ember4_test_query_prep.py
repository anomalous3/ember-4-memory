"""Smoke test for ember4/query_prep.py — before/after on real queries.

Demonstrates:
  1. Naive FTS5 query (the old _fts5_escape behavior) vs. prepared
  2. BM25 recall on both ember-4 (embers_fts) and archive (archive_fts)
  3. Stop word filtering catches noise but preserves discriminative terms
"""

from __future__ import annotations

import asyncio
import os

os.environ["EMBER4_READ_ONLY"] = "1"

from ember4.db import Database  # noqa: E402
from ember4.query_prep import (  # noqa: E402
    ENGLISH_STOPWORDS,
    get_corpus_stopwords,
    prepare_bm25_query,
)


TEST_QUERIES = [
    "atlas cluster quality basins",
    "is there a connection between embers and fiber bundles",
    "how does k-means fail for time series with structured manifold",
    "data manifold cluster basins not smooth curve",
    'what about the "fiber bundle" interpretation of time series',
    "atlas",  # single-term, should fall back to keeping it
]


def naive_fts5_escape(query: str) -> str:
    """Replicates the OLD _fts5_escape behavior for comparison."""
    words = query.split()
    if not words:
        return ""
    return " ".join(f'"{w}"' for w in words if w.strip())


async def count_matches(db, table: str, fts_q: str) -> int:
    if not fts_q:
        return 0
    try:
        cur = await db.conn.execute(
            f"SELECT COUNT(*) FROM {table} WHERE {table} MATCH ?", (fts_q,)
        )
        return (await cur.fetchone())[0]
    except Exception as e:
        return -1


async def top_matches(db, table: str, fts_q: str, k: int = 3) -> list[tuple[str, float]]:
    if not fts_q:
        return []
    try:
        cur = await db.conn.execute(
            f"SELECT rowid, bm25({table}) AS s FROM {table} "
            f"WHERE {table} MATCH ? ORDER BY s LIMIT ?",
            (fts_q, k),
        )
        return [(r["rowid"], float(r["s"])) for r in await cur.fetchall()]
    except Exception:
        return []


async def main():
    db = Database()
    await db.connect()
    corpus_stops = await get_corpus_stopwords(db)
    print(f"corpus_stopwords ({len(corpus_stops)}): {sorted(corpus_stops)}\n")

    for q in TEST_QUERIES:
        naive = naive_fts5_escape(q)
        prepared = prepare_bm25_query(q, corpus_stopwords=corpus_stops)

        print(f"=== Query: {q!r} ===")
        print(f"  naive   FTS5: {naive}")
        print(f"  prepared FTS: {prepared}")

        # Count and show top matches for both, against embers_fts
        try:
            n_naive = await count_matches(db, "embers_fts", naive)
        except Exception:
            n_naive = -1
        n_prep = await count_matches(db, "embers_fts", prepared)
        print(f"  embers_fts matches: naive={n_naive}, prepared={n_prep}")

        # Same for archive
        n_arc_naive = await count_matches(db, "archive_fts", naive)
        n_arc_prep = await count_matches(db, "archive_fts", prepared)
        print(f"  archive_fts matches: naive={n_arc_naive}, prepared={n_arc_prep}")

        # Top 3 from prepared on embers
        top = await top_matches(db, "embers_fts", prepared, k=3)
        if top:
            print(f"  prepared top-3 on embers (BM25):")
            for rowid, score in top:
                cur = await db.conn.execute(
                    "SELECT name FROM embers WHERE rowid = ?", (rowid,)
                )
                row = await cur.fetchone()
                name = row["name"] if row else "(missing)"
                print(f"    bm25={score:.2f}  {name[:65]}")
        print()

    await db.close()


if __name__ == "__main__":
    asyncio.run(main())
