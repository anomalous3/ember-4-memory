"""Compute (or refresh) the corpus-specific stop-word list for BM25 query prep.

Counts document-frequency for every token in current embers.content; terms
appearing in more than 30% of documents are added to the corpus stopword
set (alongside the standard English stop words) and stripped from FTS5
MATCH expressions in recall_v4 + archive_search_fts.

The list is cached in the `config` table under ember4_corpus_stopwords.
This script forces recomputation. Run once after the initial cutover,
then occasionally as the corpus grows (e.g., monthly cron, or after a
big ember-import event).

Usage:
    .venv/bin/python3 ember4_rebuild_stopwords.py
    .venv/bin/python3 ember4_rebuild_stopwords.py --dry-run
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys

os.environ["EMBER4_READ_ONLY"] = "0"  # writes config table

from ember4.db import Database  # noqa: E402
from ember4.query_prep import (  # noqa: E402
    CORPUS_STOPWORD_DF_THRESHOLD,
    ENGLISH_STOPWORDS,
    get_corpus_stopwords,
    rebuild_corpus_stopwords,
)


async def main_async(args: argparse.Namespace) -> int:
    db = Database()
    await db.connect()

    print(f"db_path: {db.db_path}")
    cur = await db.conn.execute("SELECT COUNT(*) FROM embers")
    n = (await cur.fetchone())[0]
    print(f"embers: {n}")
    print(f"DF threshold: {CORPUS_STOPWORD_DF_THRESHOLD:.0%} = {int(n * CORPUS_STOPWORD_DF_THRESHOLD)} documents")

    existing = await db.get_config("ember4_corpus_stopwords")
    if existing:
        import json as _json
        try:
            old = sorted(_json.loads(existing))
            print(f"existing cache: {len(old)} terms")
        except Exception:
            old = []
    else:
        old = []
        print("no existing cache")

    if args.dry_run:
        print("\n(dry-run; will recompute but not write)")
    print("\nComputing corpus stopwords...")

    if args.dry_run:
        # Replicate compute without writing
        from collections import Counter
        from ember4.query_prep import _tokenize_no_quotes, MIN_TOKEN_LENGTH
        cur = await db.conn.execute("SELECT content FROM embers")
        rows = await cur.fetchall()
        df = Counter()
        for r in rows:
            for t in set(_tokenize_no_quotes(r["content"] or "")):
                if len(t) >= MIN_TOKEN_LENGTH:
                    df[t] += 1
        threshold = int(n * CORPUS_STOPWORD_DF_THRESHOLD)
        new = sorted({t for t, c in df.items() if c >= threshold})
    else:
        new = sorted(await rebuild_corpus_stopwords(db))

    print(f"\nresult: {len(new)} corpus-specific stop words")
    if new:
        added = sorted(set(new) - set(old))
        removed = sorted(set(old) - set(new))
        print(f"  added: {len(added)} {added[:10]}{'...' if len(added) > 10 else ''}")
        print(f"  removed: {len(removed)} {removed[:10]}{'...' if len(removed) > 10 else ''}")

    print(f"\nFull corpus stopword list ({len(new)}):")
    # Print in tidy columns
    cols = 6
    for i in range(0, len(new), cols):
        print("  " + "  ".join(f"{t:<13}" for t in new[i : i + cols]))

    print(f"\nOverlap with English stopwords: "
          f"{len(set(new) & ENGLISH_STOPWORDS)}/{len(new)}")
    print(f"  (low overlap is expected — most English stops are short and "
          f"filtered by len-3 minimum)")

    await db.close()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    sys.exit(main())
