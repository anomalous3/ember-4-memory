"""ember-4 archive backfill: retire superseded session-transcript exports.

Every compaction and session end archives the whole session transcript again,
so one session can have a dozen near-identical exports crowding
archive_search. A session's JSONL only grows (compaction and resume append to
the same file), so its newest export contains every earlier one. The export
hooks now retire earlier copies as they go (tag session:<id>); this script
does the same for exports written before that.

An older export is retired only when:
  * it can be tied to a session: its session: tag, else the raw JSONL copy
    written in the same second (archive/exports/<session>_<YYYYMMDD_HHMMSS>.jsonl),
    else the opening of its table of contents;
  * a newer export of that session exists, and the older export's first,
    middle and final message sections all appear verbatim in the newer one;
  * it has no header (text before its table of contents that no later
    export has).

Retiring is reversible: status 'archived', archived_at, and a superseded-by
tag. Dry run by default.

Usage:
    .venv/bin/python3 ember4_dedupe_archive.py            # report only
    .venv/bin/python3 ember4_dedupe_archive.py --apply
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone

from ember4.config import get_data_dir, get_db_path

KINDS = ("compact-export", "session-export")
PART_RE = re.compile(r"_p\d+$")
TOC_ENTRY_RE = re.compile(r"^- \[\d+\] .*$", re.M)
SECTION_RE = re.compile(r"^## \[(\d+)\] ", re.M)


def raw_copy_index():
    """(date, HHMMSS) -> session id, from the raw JSONL copies' filenames."""
    index = {}
    for f in (get_data_dir() / "archive" / "exports").glob("*.jsonl"):
        try:
            session, day, hms = f.stem.rsplit("_", 2)
        except ValueError:
            continue
        index[(day, hms)] = session
    return index


def session_for(export, raw_index):
    for t in export["tags"]:
        if t.startswith("session:"):
            return t
    # chunk ids end <date>_<project>_<kind>_<HHMMSS>; the raw copy's stamp is
    # taken in the same run, so allow a second either way
    m = re.match(r"(\d{4})-(\d{2})-(\d{2})_.*_(\d{6})$", export["base_id"])
    if m:
        day = "".join(m.group(1, 2, 3))
        stamp = datetime.strptime(day + m.group(4), "%Y%m%d%H%M%S")
        for delta in (0, -1, 1, -2, 2):
            t = stamp + timedelta(seconds=delta)
            hit = raw_index.get((t.strftime("%Y%m%d"), t.strftime("%H%M%S")))
            if hit:
                return f"session:{hit}"
    toc = TOC_ENTRY_RE.findall(export["content"])[:3]
    return "toc:" + "\n".join(toc) if toc else None


def probes(text, n=400):
    """First, middle and final message sections (n chars each): the
    containment probes. One probe alone can hit boilerplate two sessions share."""
    starts = [m.start() for m in SECTION_RE.finditer(text)]
    if not starts:
        return []
    bounds = list(zip(starts, starts[1:] + [len(text)]))
    picks = {0, len(bounds) // 2, len(bounds) - 1}
    return [text[a:b].strip()[:n] for i, (a, b) in enumerate(bounds) if i in picks]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--apply", action="store_true", help="write the changes")
    args = ap.parse_args()

    db = sqlite3.connect(str(get_db_path()), timeout=15)
    rows = db.execute(
        "SELECT chunk_id, content, tags, created_at FROM archive "
        f"WHERE status = 'active' AND chunk_type IN {KINDS!r} ORDER BY created_at"
    ).fetchall()

    # Group parts (_p1, _p2, ...) into one export each.
    exports = {}
    for chunk_id, content, tags, created in rows:
        base = PART_RE.sub("", chunk_id)
        e = exports.setdefault(base, {"base_id": base, "ids": [], "parts": [],
                                      "tags": set(), "created": created})
        e["ids"].append(chunk_id)
        e["parts"].append(content)
        e["tags"].update(json.loads(tags or "[]"))
    for e in exports.values():
        e["content"] = "\n".join(e["parts"])
        first = e["parts"][0].lstrip()
        e["has_header"] = "has-header" in e["tags"] or not first.startswith("## Table of Contents")

    raw_index = raw_copy_index()
    by_session = defaultdict(list)
    unmatched = 0
    for e in exports.values():
        key = session_for(e, raw_index)
        if key:
            by_session[key].append(e)
        else:
            unmatched += 1

    retire, kept_header, unverified = [], 0, 0
    for key, group in by_session.items():
        if len(group) < 2:
            continue
        group.sort(key=lambda e: e["created"])
        newest = group[-1]
        for old in group[:-1]:
            if old["has_header"]:
                kept_header += 1
                continue
            ps = probes(old["content"])
            if not ps or not all(p in newest["content"] for p in ps):
                unverified += 1
                continue
            retire.append((old, newest))

    n_chunks = sum(len(o["ids"]) for o, _ in retire)
    print(f"transcript exports: {len(exports)} ({len(rows)} chunks) in {len(by_session)} sessions; "
          f"{unmatched} unmatched")
    print(f"to retire: {len(retire)} exports ({n_chunks} chunks); kept for header: {kept_header}; "
          f"kept, containment not verified: {unverified}")

    if not args.apply:
        print("(dry run; --apply to write)")
        return 0

    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
    for old, newest in retire:
        for chunk_id in old["ids"]:
            tags = json.loads(db.execute(
                "SELECT tags FROM archive WHERE chunk_id = ?", (chunk_id,)).fetchone()[0] or "[]")
            tags.append(f"superseded-by:{newest['ids'][0]}")
            db.execute(
                "UPDATE archive SET status = 'archived', archived_at = ?, tags = ? WHERE chunk_id = ?",
                (now, json.dumps(tags), chunk_id),
            )
    db.commit()
    db.close()
    print(f"retired {n_chunks} chunks")
    return 0


if __name__ == "__main__":
    sys.exit(main())
