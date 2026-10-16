"""ember-4 migration: backfill Nomic-v1.5 vectors for embers + archive summaries.

Run once, manually. Turns off the read-only freeze for the duration of the
migration (only additive writes to new tables — never touches old tables).
After migration, the freeze is re-enabled.

Pre-flight checks ensure we don't accidentally overwrite an existing
migration. The old vec_embers / vec_archive / vec_centroids tables are
read for ID enumeration only and are never modified.

Usage:
    .venv/bin/python3 ember4_migrate.py
    .venv/bin/python3 ember4_migrate.py --dry-run     # plan, don't write
    .venv/bin/python3 ember4_migrate.py --batch-size 64
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from typing import Iterable

# Migration must be allowed to write. We turn off the freeze BEFORE importing
# ember4.config so READ_ONLY is correctly False at module load time.
os.environ["EMBER4_READ_ONLY"] = "0"

import numpy as np  # noqa: E402

from ember4.db import Database  # noqa: E402
from ember4.embedder import (  # noqa: E402
    Embedder,
    NOMIC_V1_5_SPEC,
    serialize_vector,
)

MODEL = NOMIC_V1_5_SPEC
MODEL_ID = "nomic-v1.5"


def chunked(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


async def preflight(db: Database) -> dict:
    """Inspect current state — what's already migrated, what's pending."""
    stats: dict = {}

    cur = await db.conn.execute("SELECT COUNT(*) FROM embers")
    stats["embers_total"] = (await cur.fetchone())[0]

    cur = await db.conn.execute(
        "SELECT COUNT(*) FROM archive WHERE status = 'active'"
    )
    stats["archive_active"] = (await cur.fetchone())[0]

    for tbl in (
        "vec_embers_nomic_v15_clustering",
        "vec_embers_nomic_v15_retrieval",
        "vec_archive_nomic_v15_retrieval",
    ):
        cur = await db.conn.execute(f"SELECT COUNT(*) FROM {tbl}")
        stats[tbl] = (await cur.fetchone())[0]

    return stats


async def migrate_embers(
    db: Database, embedder: Embedder, batch_size: int, dry_run: bool
) -> dict:
    """Generate clustering + retrieval vectors for every ember."""
    cur = await db.conn.execute(
        "SELECT ember_id, name, content FROM embers ORDER BY created_at"
    )
    rows = await cur.fetchall()

    # Build the input text once: name + content, the same representation that
    # ember-3 uses. (Embedder will prepend the role prefix at embed time.)
    ids: list[str] = []
    texts: list[str] = []
    for r in rows:
        ids.append(r["ember_id"])
        # Name first, then content — most embers begin their content with a
        # restatement of the name, so this is mostly redundant but cheap.
        texts.append(f"{r['name']}\n{r['content']}")

    print(f"  → {len(ids)} embers; batch_size={batch_size}")
    if dry_run:
        return {"ember_clustering": len(ids), "ember_retrieval": len(ids)}

    t_total = time.time()

    # ─── Clustering vectors ──────────────────────────────────────────────
    t0 = time.time()
    n_inserted = 0
    for batch_ids, batch_texts in zip(
        chunked(ids, batch_size), chunked(texts, batch_size)
    ):
        vecs = embedder.batch_embed_cluster(batch_texts)  # (B, 768)
        await db.conn.executemany(
            "INSERT OR REPLACE INTO vec_embers_nomic_v15_clustering "
            "(ember_id, embedding) VALUES (?, ?)",
            [(eid, serialize_vector(v)) for eid, v in zip(batch_ids, vecs)],
        )
        n_inserted += len(batch_ids)
        if n_inserted % (batch_size * 4) == 0 or n_inserted == len(ids):
            rate = n_inserted / max(1e-6, time.time() - t0)
            print(
                f"    clustering: {n_inserted:>5}/{len(ids)}  "
                f"({rate:.1f}/s)"
            )
    await db.conn.commit()
    print(f"  clustering done: {n_inserted} in {time.time()-t0:.1f}s")

    # ─── Retrieval vectors ───────────────────────────────────────────────
    t0 = time.time()
    n_inserted = 0
    for batch_ids, batch_texts in zip(
        chunked(ids, batch_size), chunked(texts, batch_size)
    ):
        vecs = embedder.batch_embed_document(batch_texts)
        await db.conn.executemany(
            "INSERT OR REPLACE INTO vec_embers_nomic_v15_retrieval "
            "(ember_id, embedding) VALUES (?, ?)",
            [(eid, serialize_vector(v)) for eid, v in zip(batch_ids, vecs)],
        )
        n_inserted += len(batch_ids)
        if n_inserted % (batch_size * 4) == 0 or n_inserted == len(ids):
            rate = n_inserted / max(1e-6, time.time() - t0)
            print(
                f"    retrieval:  {n_inserted:>5}/{len(ids)}  "
                f"({rate:.1f}/s)"
            )
    await db.conn.commit()
    print(f"  retrieval done:  {n_inserted} in {time.time()-t0:.1f}s")

    return {
        "ember_clustering": len(ids),
        "ember_retrieval": len(ids),
        "total_seconds": time.time() - t_total,
    }


async def migrate_archive(
    db: Database, embedder: Embedder, batch_size: int, dry_run: bool
) -> dict:
    """Generate retrieval vectors for archive summaries.

    Embeds the `summary` field, not `content`. summary is short by design
    (typically <500 tokens); full content can exceed Nomic's 8K context for
    big session exports and is unsuitable for whole-document embedding.
    """
    cur = await db.conn.execute(
        "SELECT chunk_id, summary FROM archive WHERE status = 'active' "
        "AND summary != '' ORDER BY created_at"
    )
    rows = await cur.fetchall()

    ids = [r["chunk_id"] for r in rows]
    texts = [r["summary"] for r in rows]

    print(f"  → {len(ids)} active archive summaries; batch_size={batch_size}")
    if dry_run:
        return {"archive_retrieval": len(ids)}

    t0 = time.time()
    n_inserted = 0
    for batch_ids, batch_texts in zip(
        chunked(ids, batch_size), chunked(texts, batch_size)
    ):
        vecs = embedder.batch_embed_document(batch_texts)
        await db.conn.executemany(
            "INSERT OR REPLACE INTO vec_archive_nomic_v15_retrieval "
            "(chunk_id, embedding) VALUES (?, ?)",
            [(cid, serialize_vector(v)) for cid, v in zip(batch_ids, vecs)],
        )
        n_inserted += len(batch_ids)
        if n_inserted % (batch_size * 4) == 0 or n_inserted == len(ids):
            rate = n_inserted / max(1e-6, time.time() - t0)
            print(f"    archive:    {n_inserted:>5}/{len(ids)}  ({rate:.1f}/s)")
    await db.conn.commit()
    print(f"  archive done:    {n_inserted} in {time.time()-t0:.1f}s")
    return {"archive_retrieval": len(ids), "total_seconds": time.time() - t0}


async def main_async(args: argparse.Namespace) -> int:
    assert os.environ.get("EMBER4_READ_ONLY") == "0", "freeze must be off"
    db = Database()
    await db.connect()
    assert not db._read_only, "DB opened read-only despite EMBER4_READ_ONLY=0"

    print(f"db_path:  {db.db_path}")
    print(f"model:    {MODEL.name} ({MODEL.hf_repo})")
    print(f"onnx:     {MODEL.onnx_filename}")
    print(f"dim:      {MODEL.dimension}")
    print(f"prefixes: {MODEL.prefixes}")
    print()

    print("─── Pre-flight ─────────────────────────────────────────")
    stats = await preflight(db)
    for k, v in stats.items():
        print(f"  {k:<40} {v}")

    if any(stats[tbl] > 0 for tbl in (
        "vec_embers_nomic_v15_clustering",
        "vec_embers_nomic_v15_retrieval",
        "vec_archive_nomic_v15_retrieval",
    )) and not args.force:
        print("\nMigration tables are not empty. Re-run with --force to overwrite.")
        await db.close()
        return 1

    # Warm up the embedder before timing
    print("\n─── Embedder warmup ────────────────────────────────────")
    embedder = Embedder(MODEL)
    _ = embedder.embed_cluster("warmup")
    print("  ready")

    print("\n─── Embers ─────────────────────────────────────────────")
    e_stats = await migrate_embers(db, embedder, args.batch_size, args.dry_run)
    print(f"  {e_stats}")

    print("\n─── Archive summaries ──────────────────────────────────")
    a_stats = await migrate_archive(db, embedder, args.batch_size, args.dry_run)
    print(f"  {a_stats}")

    print("\n─── Post-flight ────────────────────────────────────────")
    stats_after = await preflight(db)
    for k, v in stats_after.items():
        delta = v - stats.get(k, 0)
        if delta:
            print(f"  {k:<40} {v}  (+{delta})")
        else:
            print(f"  {k:<40} {v}")

    await db.close()
    print("\nDone. Remember to set EMBER4_READ_ONLY=1 for normal use.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--force", action="store_true",
        help="overwrite non-empty migration tables",
    )
    args = parser.parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    sys.exit(main())
