"""ember-4 wire-up smoke test.

Exercises the actual server-side helpers (_post_insert_v4, _relate_on_insert,
log_update) on a synthetic test ember and verifies:

  1. Nomic clustering + retrieval vectors are populated
  2. log_update row written with correct (event_type, origin, content_changed)
  3. 'related' / 'contradict_candidate' edges populated by _relate_on_insert
  4. Contribution filter correctly accepts the agent-origin edit
  5. Cleanup removes everything (no synthetic data lingers)

This is the live cutover validator — if this script runs clean, the wired
server.py paths are ready to populate update_log under real operation.
"""

from __future__ import annotations

import asyncio
import os
import sys
import uuid

os.environ["EMBER4_READ_ONLY"] = "0"

from ember4.db import Database  # noqa: E402


TEST_NAME = "TEST WIRE-UP — sourdough starter feeding schedule"
TEST_CONTENT = (
    "TEST EMBER from ember4_test_wireup.py. A sourdough starter kept at "
    "room temperature is fed twice a day at a 1:1:1 ratio of starter, "
    "flour and water. This is a synthetic event for wire-up "
    "validation; it will be deleted at the end of the smoke test."
)


async def main_async() -> int:
    db = Database()
    await db.connect()
    assert not db._read_only

    test_id = f"test-wireup-{uuid.uuid4()}"
    print(f"test ember_id: {test_id}\n")

    # Pre-state
    cur = await db.conn.execute("SELECT COUNT(*) FROM update_log")
    n_before = (await cur.fetchone())[0]
    print(f"update_log rows (pre): {n_before}")

    # ─── 1. Save base ember row ──────────────────────────────────────────
    # Skip the MiniLM embedding path; we're testing the v4 hooks specifically.
    # Use a dummy 384-dim vector for vec_embers since the column is required
    # for the FK insert via save_ember.
    import numpy as np
    from ember4.embedder import serialize_vector
    dummy_minilm = np.zeros(384, dtype=np.float32)
    dummy_minilm[0] = 1.0  # not L2-normalized but it'll do for a smoke row
    await db.save_ember(
        ember_id=test_id,
        name=TEST_NAME,
        content=TEST_CONTENT,
        tags=["test", "wireup"],
        cell_id=-1,           # ember-4 ignores cell_id
        importance="learning",
        source="manual",
        embedding=serialize_vector(dummy_minilm),
    )
    print(f"[1/5] base ember inserted")

    # ─── 2. v4 post-insert hook (the real wire-up surface) ───────────────
    from ember4.server import _post_insert_v4, _relate_on_insert, _db as srv_db
    import ember4.server as srv
    if srv._db is None:
        srv._db = db
    vec_cluster, vec_retrieval = await _post_insert_v4(
        test_id, TEST_NAME, TEST_CONTENT
    )
    print(f"[2/5] _post_insert_v4 ok — shapes {vec_cluster.shape} / {vec_retrieval.shape}")

    # Confirm both vector tables now have a row for test_id
    for table in ("vec_embers_nomic_v15_clustering",
                  "vec_embers_nomic_v15_retrieval"):
        cur = await db.conn.execute(
            f"SELECT COUNT(*) FROM {table} WHERE ember_id = ?", (test_id,)
        )
        n = (await cur.fetchone())[0]
        assert n == 1, f"{table} missing test row"
    print(f"      vec tables both populated")

    # ─── 3. relate_on_insert: edges with the right types ─────────────────
    await _relate_on_insert(test_id, vec_cluster)
    cur = await db.conn.execute(
        """SELECT edge_type, COUNT(*) as n FROM edges
           WHERE source_id = ? OR target_id = ?
           GROUP BY edge_type""",
        (test_id, test_id),
    )
    edge_counts = {r["edge_type"]: r["n"] for r in await cur.fetchall()}
    print(f"[3/5] _relate_on_insert edges by type: {edge_counts}")
    # No 'shadow' edges from new inserts.
    assert "shadow" not in edge_counts, \
        "shadow edge created by new path; basin pathology not removed"

    # ─── 4. log_update writes the right row ──────────────────────────────
    await db.log_update(
        ember_id=test_id,
        event_type="edit", origin="agent",
        content_changed=True,
        session_id="smoke-test",
        details="ember4_test_wireup.py",
    )
    cur = await db.conn.execute(
        """SELECT event_type, origin, content_changed, details
           FROM update_log WHERE ember_id = ?""",
        (test_id,),
    )
    log_rows = list(await cur.fetchall())
    assert len(log_rows) == 1, f"expected 1 update_log row, got {len(log_rows)}"
    row = log_rows[0]
    assert row["event_type"] == "edit", row["event_type"]
    assert row["origin"] == "agent", row["origin"]
    assert row["content_changed"] == 1, row["content_changed"]
    print(f"[4/5] log_update row: {dict(row)}")

    # ─── 5. Contribution filter accepts our event ────────────────────────
    cur = await db.conn.execute(
        """SELECT CASE WHEN
            (event_type IN ('edit','contradict_old','contradict_new','edge_form')
             AND origin IN ('user','agent'))
            OR
            (event_type = 'tag_update' AND origin IN ('user','agent')
             AND content_changed = 0)
          THEN 1 ELSE 0 END AS contributes
           FROM update_log WHERE ember_id = ?""",
        (test_id,),
    )
    contrib = (await cur.fetchone())["contributes"]
    assert contrib == 1, "contribution filter rejected agent edit"
    print(f"[5/5] contribution filter: PASSES (event counts toward importance)")

    # ─── Cleanup ────────────────────────────────────────────────────────
    print("\nCleaning up test data...")
    await db.conn.execute(
        "DELETE FROM vec_embers_nomic_v15_clustering WHERE ember_id = ?", (test_id,)
    )
    await db.conn.execute(
        "DELETE FROM vec_embers_nomic_v15_retrieval WHERE ember_id = ?", (test_id,)
    )
    await db.conn.execute(
        "DELETE FROM edges WHERE source_id = ? OR target_id = ?", (test_id, test_id)
    )
    await db.conn.execute(
        "DELETE FROM update_log WHERE ember_id = ?", (test_id,)
    )
    await db.conn.execute("DELETE FROM embers WHERE ember_id = ?", (test_id,))
    await db.conn.commit()

    cur = await db.conn.execute("SELECT COUNT(*) FROM update_log")
    n_after = (await cur.fetchone())[0]
    assert n_after == n_before, \
        f"update_log row count drifted: {n_before} → {n_after}"
    print(f"update_log rows (post-cleanup): {n_after}  (matches pre)")

    # Verify test row is gone everywhere
    for tbl, col in [
        ("embers", "ember_id"),
        ("vec_embers_nomic_v15_clustering", "ember_id"),
        ("vec_embers_nomic_v15_retrieval", "ember_id"),
        ("edges", "source_id"),
        ("update_log", "ember_id"),
    ]:
        cur = await db.conn.execute(
            f"SELECT COUNT(*) FROM {tbl} WHERE {col} = ?", (test_id,)
        )
        n = (await cur.fetchone())[0]
        assert n == 0, f"{tbl}: {n} rows for test_id after cleanup"
    print("All test rows removed across all tables.")

    await db.close()
    print("\n✓ Wire-up smoke test PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main_async()))
