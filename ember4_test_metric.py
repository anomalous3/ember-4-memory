"""Pin the vector-distance contract that every similarity threshold relies on.

sqlite-vec returns plain L2, not squared L2. Until 2026-10-16 the conversion
helper assumed squared distance, so every cosine cutoff meant something
stricter than its comment said. The cutoffs were translated to keep their
effective behavior; this test fails if sqlite-vec's metric or the helper
changes under them.

    .venv/bin/python3 ember4_test_metric.py
"""
import math
import sqlite3
import struct
import sys

import numpy as np
import sqlite_vec

from ember4.utils import l2_to_cosine


def vec0_distances():
    db = sqlite3.connect(":memory:")
    db.enable_load_extension(True)
    sqlite_vec.load(db)
    db.execute("CREATE VIRTUAL TABLE v USING vec0(e float[2])")
    for i, x in enumerate([(1, 0), (0, 1), (-1, 0)]):
        db.execute("INSERT INTO v(rowid, e) VALUES (?, ?)", (i + 1, struct.pack("2f", *x)))
    rows = db.execute("SELECT rowid, distance FROM v WHERE e MATCH ? AND k = 3",
                      (struct.pack("2f", 1, 0),)).fetchall()
    return dict(rows)


def main():
    fails = []

    d = vec0_distances()
    for rowid, want_dist, want_cos in [(1, 0.0, 1.0), (2, math.sqrt(2), 0.0), (3, 2.0, -1.0)]:
        if not math.isclose(d[rowid], want_dist, abs_tol=1e-6):
            fails.append(f"sqlite-vec distance for row {rowid}: {d[rowid]} != {want_dist}")
        if not math.isclose(l2_to_cosine(d[rowid]), want_cos, abs_tol=1e-6):
            fails.append(f"l2_to_cosine({d[rowid]}) = {l2_to_cosine(d[rowid])} != {want_cos}")

    # Random unit vectors: the helper agrees with the dot product.
    rng = np.random.default_rng(0)
    a = rng.normal(size=(200, 16)); a /= np.linalg.norm(a, axis=1, keepdims=True)
    b = rng.normal(size=(200, 16)); b /= np.linalg.norm(b, axis=1, keepdims=True)
    err = max(abs(l2_to_cosine(float(np.linalg.norm(x - y))) - float(x @ y)) for x, y in zip(a, b))
    if err > 1e-9:
        fails.append(f"l2_to_cosine disagrees with dot product by {err}")

    # The translated thresholds select exactly the distances the old ones did:
    # old helper said cos = 1 - d/2, so `old > t` <=> d < 2(1 - t).
    translated = {0.75: 0.875, 0.70: 0.82, 0.4: 0.28, 0.65: 0.755, 0.85: 0.955}
    for t_old, t_new in translated.items():
        boundary = 2 * (1 - t_old)
        for dist in np.linspace(0, 2, 4001):
            if abs(dist - boundary) < 1e-9:
                continue
            if (1 - dist / 2 > t_old) != (l2_to_cosine(dist) > t_new):
                fails.append(f"threshold {t_old}->{t_new} disagrees at L2 {dist:.4f}")
                break

    if fails:
        print("FAIL\n  " + "\n  ".join(fails))
        return 1
    print("ok: sqlite-vec returns L2; l2_to_cosine and translated thresholds agree")
    return 0


if __name__ == "__main__":
    sys.exit(main())
