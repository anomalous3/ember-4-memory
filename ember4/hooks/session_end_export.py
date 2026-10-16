#!/usr/bin/env python3
"""
Session End Export Hook — Save session transcript on exit.

Triggered at SessionEnd. Reuses the JSONL parsing and transcript building
from pre_compact_export.py but writes with chunk_type "session-export"
and produces no systemMessage (session is ending, nobody to read it).

Then refreshes the recall graph (kNN + temporal + tag-cohort edges) if any
ember is missing its kNN edges, since live inserts don't add them. Both steps
run in a detached child, so the session exits without waiting.

write_archive_chunk dedups on content hash, so an unchanged transcript isn't stored twice.
"""

import argparse
import asyncio
import json
import os
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

try:
    import fcntl
except ImportError:  # Windows: no lock; concurrent rebuilds are just redundant
    fcntl = None

# Import shared logic from the pre-compact module
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from ember4.config import get_data_dir, get_db_path  # noqa: E402
from ember4.hooks.pre_compact_export import export_session  # noqa: E402

GRAPH_LOG = get_data_dir() / "graph_build.log"
GRAPH_LOCK = get_data_dir() / "graph_build.lock"


def parse_stdin():
    """Read session info from stdin (SessionEnd hook format)."""
    session_id = None
    cwd = None
    try:
        input_data = sys.stdin.read()
        if input_data:
            data = json.loads(input_data)
            session_id = data.get("session_id")
            cwd = data.get("cwd")
    except (json.JSONDecodeError, KeyError, TypeError):
        pass
    return session_id, cwd


def do_export(session_id, cwd):
    """Perform the actual session export work. Silent: the session is ending."""
    export_session(session_id, cwd, "session-export")


def graph_is_stale(db_path):
    """True when some active ember has no kNN edges (new since the last build).
    Stale embers are left out of the graph, so they never count."""
    db = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=15)
    try:
        row = db.execute(
            "SELECT 1 FROM embers e WHERE e.is_stale = 0 AND NOT EXISTS (SELECT 1 FROM edges g "
            "WHERE g.source_id = e.ember_id AND g.edge_type = 'knn_nomic_v15') LIMIT 1"
        ).fetchone()
        return row is not None
    finally:
        db.close()


def refresh_graph():
    """Rebuild the recall graph if it's stale. Output goes to graph_build.log."""
    db_path = get_db_path()
    if not db_path.exists():
        return
    with open(GRAPH_LOCK, "w") as lock:
        try:
            if fcntl:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return  # another session's rebuild is already running
        if not graph_is_stale(db_path):
            return
        import ember4_build_graph
        import ember4_build_tag_edges
        with open(GRAPH_LOG, "a", encoding="utf-8") as log:
            sys.stdout = sys.stderr = log
            print(f"\n=== {datetime.now().isoformat(timespec='seconds')} session-end graph refresh ===")
            try:
                asyncio.run(ember4_build_graph.main_async(argparse.Namespace(
                    k_knn=20, n_temporal=5, tau_days=7.0, dry_run=False)))
                asyncio.run(ember4_build_tag_edges.main_async(argparse.Namespace(
                    top_k=15, dry_run=False)))
            except Exception as e:
                print(f"graph refresh failed: {e!r}")
            log.flush()


def run_child(session_id, cwd):
    """Export, then refresh the graph; never let a lingering thread keep us alive."""
    try:
        do_export(session_id, cwd)
    except Exception:
        pass
    try:
        refresh_graph()
    except Exception:
        pass
    os._exit(0)


def main():
    session_id, cwd = parse_stdin()

    # Detach into background so Claude Code can exit without killing us.
    # Stdin is already consumed, so the child has everything it needs.
    if sys.platform == "win32":
        import subprocess
        subprocess.Popen(
            [sys.executable, __file__, "--child", session_id or "", cwd or ""],
            creationflags=subprocess.CREATE_NO_WINDOW | subprocess.DETACHED_PROCESS,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return
    else:
        pid = os.fork()
        if pid > 0:
            return
        os.setsid()

    run_child(session_id, cwd)


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "--child":
        # Windows detached child process — args passed on command line
        sid = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] else None
        cwd = sys.argv[3] if len(sys.argv) > 3 and sys.argv[3] else None
        run_child(sid, cwd)
    else:
        main()
