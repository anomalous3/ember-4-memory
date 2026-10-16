"""ember-4 query preparation for BM25/FTS5 retrieval.

The naive "OR-of-all-words" FTS5 query has two failure modes that have
plagued archive_search and the v4 BM25 pathway:

  1. English stop words ("is", "the", "with", "between") match almost
     everything. BM25's IDF eventually washes them out, but the MATCH
     result set is enormous before ranking and the noise pollutes debug
     output.

  2. Corpus-specific common terms (a main project's name can appear in ~30% of embers;
     "ember" in many) similarly fail to discriminate. BM25 handles them
     too, but they shouldn't be in the MATCH expression at all.

This module:
  - Strips a proper English stop-word list (~170 words)
  - Strips corpus-specific high-DF terms (computed once, cached in DB
    config; threshold = 30% of documents)
  - Preserves quoted phrases from the input verbatim
  - Falls back gracefully: if stripping leaves an empty query, retry
    without the corpus-specific strip so the search isn't silently empty

Used by:
  - recall_v4.gather_seeds_bm25 (ember-4 dense+BM25 pipeline)
  - db.archive_search_fts (archive full-text search — same problem,
    inherited from ember-3)

Single corpus_stopword cache is shared between embers and archive
because the high-DF terms in embers tend to also be the noisy ones in
archive (project names, ember, learning, etc.).
"""

from __future__ import annotations

import json
import re
from collections import Counter
from typing import Optional


# ─── English stop words ──────────────────────────────────────────────────────
# Conservative list (~170 words) — common function words, auxiliaries,
# pronouns, conjunctions. Deliberately keeps content-bearing nouns and
# verbs that might still discriminate (e.g., "make", "find", "see").

ENGLISH_STOPWORDS: frozenset[str] = frozenset("""
a about above after again against all am an and any are aren as at
be because been before being below between both but by
can cannot could couldn
did didn do does doesn doing don down during
each either else
few for from further
had hadn has hasn have haven having he her here hers herself him
himself his how
i if in into is isn it its itself
just
let
me might more most must mustn my myself
need
no nor not now
of off on once only or other ought our ours ourselves out over own
same shall shan she should shouldn so some such
than that the their theirs them themselves then there these they
this those through thus to too
under until up upon us
very
was wasn we were weren what when whence where whether which while
who whom whose why will with within without won would wouldn
yet you your yours yourself yourselves
ll ve re s t d m
""".split())

# DF threshold for corpus-specific stopwords. A term appearing in >30%
# of documents adds essentially no information per BM25's IDF, while
# bloating the MATCH result set. We strip it from the query expression
# (but keep it in the index — other queries with different companion
# terms may still want to score on it).
CORPUS_STOPWORD_DF_THRESHOLD = 0.30

# Minimum token length to be considered. Filters out single-letter
# noise and most punctuation artifacts.
MIN_TOKEN_LENGTH = 3


# ─── Tokenization ────────────────────────────────────────────────────────────

_QUOTE_RE = re.compile(r'"([^"]+)"')
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def extract_quoted_phrases(text: str) -> list[str]:
    """Return any "..." phrases from input verbatim (stripped).

    These survive prepare_bm25_query unchanged as FTS5 phrase queries —
    `"fiber bundle"` is much more specific than `fiber OR bundle`. The
    caller can use quotes deliberately when they want phrase semantics.
    """
    return [m.strip() for m in _QUOTE_RE.findall(text) if m.strip()]


def _tokenize_no_quotes(text: str) -> list[str]:
    """Tokenize the non-quoted portion of text to lowercase alphanumeric."""
    no_quotes = _QUOTE_RE.sub(" ", text)
    return _TOKEN_RE.findall(no_quotes.lower())


# ─── Corpus stopword computation + caching ───────────────────────────────────

CONFIG_KEY = "ember4_corpus_stopwords"


async def get_corpus_stopwords(db) -> set[str]:
    """Return the cached corpus-specific stop-word set, computing on first call.

    Cached in the `config` table under key=CONFIG_KEY as a JSON list.
    Manual refresh: call rebuild_corpus_stopwords(db). Auto-refresh is
    not built in — the list is stable enough that periodic manual
    rebuilds (e.g., monthly) suffice.
    """
    cached = await db.get_config(CONFIG_KEY)
    if cached:
        try:
            return set(json.loads(cached))
        except Exception:
            pass
    return await rebuild_corpus_stopwords(db)


async def rebuild_corpus_stopwords(db) -> set[str]:
    """Compute terms appearing in >threshold of embers; cache + return.

    Counts document-frequency over the embers table only. The archive
    has more diverse vocabulary; using only embers gives us a list
    skewed toward the "project canon" (project names, ember, etc.) which is the
    noise we actually want to suppress.
    """
    cur = await db.conn.execute("SELECT content FROM embers")
    rows = await cur.fetchall()
    n_docs = len(rows)
    if n_docs == 0:
        return set()

    df: Counter[str] = Counter()
    for row in rows:
        content = row["content"] if isinstance(row, dict) or hasattr(row, "keys") else row[0]
        tokens = set(_tokenize_no_quotes(content or ""))
        for t in tokens:
            if len(t) >= MIN_TOKEN_LENGTH:
                df[t] += 1

    threshold = int(n_docs * CORPUS_STOPWORD_DF_THRESHOLD)
    corpus_stops = {t for t, c in df.items() if c >= threshold}

    # Cache
    await db.set_config(CONFIG_KEY, json.dumps(sorted(corpus_stops)))
    return corpus_stops


# ─── Public API ──────────────────────────────────────────────────────────────

def prepare_bm25_query(
    text: str,
    corpus_stopwords: Optional[set[str]] = None,
) -> str:
    """Convert free-form text → FTS5 MATCH expression.

    Output format: OR-of (quoted-phrases + individual-terms).
    Example:
        prepare_bm25_query('is there a "fiber bundle" in time series?')
        →  '"fiber bundle" OR "time" OR "series"'

    Behavior:
      * Quoted phrases preserved verbatim
      * Tokens lowercased to a-z0-9 (FTS5 unicode61 will match these)
      * English stop words stripped
      * Corpus-specific stop words stripped (if provided)
      * Fallback: if strict filtering yields empty, retry without
        corpus stopwords so single-term queries on common project
        names still produce a result set
    """
    corpus_stopwords = corpus_stopwords or set()
    phrases = extract_quoted_phrases(text)

    raw_tokens = _tokenize_no_quotes(text)
    strict = [
        t for t in raw_tokens
        if len(t) >= MIN_TOKEN_LENGTH
        and t not in ENGLISH_STOPWORDS
        and t not in corpus_stopwords
    ]

    if not strict and not phrases:
        # Fallback — let corpus-common terms back in so we don't return
        # nothing for a single-word query like a project name.
        strict = [
            t for t in raw_tokens
            if len(t) >= MIN_TOKEN_LENGTH and t not in ENGLISH_STOPWORDS
        ]

    # Dedupe while preserving order
    seen: set[str] = set()
    uniq_tokens: list[str] = []
    for t in strict:
        if t not in seen:
            seen.add(t)
            uniq_tokens.append(t)

    parts = [f'"{p}"' for p in phrases] + [f'"{t}"' for t in uniq_tokens]
    return " OR ".join(parts)
