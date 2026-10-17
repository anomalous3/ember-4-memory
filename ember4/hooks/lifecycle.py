#!/usr/bin/env python3
"""Portable command hooks for Codex and Claude Code.

Archives use a durable raw snapshot + job file before detaching a worker.
Failed jobs stay in archive/pending; --drain retries them without a model call.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def atomic_json(path, data):
    from ember4.hooks.transcript_archive import private_dir
    private_dir(path.parent)
    temp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as out:
            json.dump(data, out, ensure_ascii=False)
            out.flush()
            os.fsync(out.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def spawn_worker(job, data_dir):
    """Never inherit hook stdio: a child holding the pipe defeats detachment."""
    env = dict(os.environ, EMBER4_DATA_DIR=str(data_dir))
    log_path = data_dir / "archive" / "hook.log"
    fd = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    kwargs = {"creationflags": subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP} \
        if os.name == "nt" else {"start_new_session": True}
    with os.fdopen(fd, "ab", buffering=0) as log:
        subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--worker", str(job)],
                         env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                         close_fds=True, **kwargs)


def process_job(job_path):
    job_path = Path(job_path)
    if not job_path.exists():
        return True  # Another worker finished the same durable job.
    job = json.loads(job_path.read_text())
    # Set before loading config / graph modules (which cache environment state).
    os.environ["EMBER4_DATA_DIR"] = job["data_dir"]
    from ember4.hooks.transcript_archive import export_session
    result = export_session(job["session_id"], job["cwd"], job["kind"], job.get("header", ""),
                            transcript_path=job["snapshot"], client=job["client"], snapshot=True)
    if not result["chunk_ids"]:
        print(f"[ember-4] export failed; job retained: {job_path}: {result['error']}", flush=True)
        return False
    print(f"[ember-4] archived {job['client']} session: {len(result['chunk_ids'])} parts"
          + (f"; {result['error']}" if result["error"] else ""), flush=True)
    job_path.unlink(missing_ok=True)
    if job["kind"] == "session-export":
        try:
            from ember4.hooks.session_end_export import refresh_graph
            refresh_graph()
        except Exception as exc:
            print(f"[ember-4] graph refresh failed: {exc}", file=sys.stderr, flush=True)
    return True


def timing(event, payload, client):
    from ember4.hooks.pre_compact_export import archive_db_path
    sid = payload.get("session_id")
    if not sid:
        return
    key = hashlib.sha256(str(sid).encode()).hexdigest()
    state = archive_db_path(payload.get("cwd")).parent / "hooks" / client / (key + ".json")
    if event == "Stop":
        atomic_json(state, {"stopped_at": time.time()})
        return
    try:
        delta = int((time.time() - float(json.loads(state.read_text())["stopped_at"])) // 60)
    except (OSError, ValueError, KeyError, TypeError):
        return
    if delta < 1:
        return
    elapsed = f"{delta}m" if delta < 60 else (
        f"{delta // 60}h{delta % 60}m" if delta < 1440 else f"{delta // 1440}d")
    marker = f"[time: {datetime.now().astimezone():%H:%M %Z} · +{elapsed} since last turn]"
    print(json.dumps({"hookSpecificOutput": {
        "hookEventName": "UserPromptSubmit", "additionalContext": marker}}))


def queue_export(payload, client, event, *, foreground=False):
    from ember4.hooks.transcript_archive import snapshot_transcript
    source, sid = payload.get("transcript_path"), payload.get("session_id")
    if not source or not sid:
        raise ValueError("archiving requires transcript_path and session_id; refusing to guess")
    raw, cwd, db_path = snapshot_transcript(source, payload.get("cwd"), client)
    kind = "compact-export" if event == "PreCompact" else "session-export"
    custom = payload.get("custom_instructions") if client == "claude" else None
    header = f"## /compact instructions\n\n{custom.strip()}\n\n---\n\n" if custom else ""
    job = db_path.parent / "archive" / "pending" / (raw.stem + ".json")
    atomic_json(job, {"client": client, "session_id": sid, "cwd": cwd,
                      "snapshot": str(raw), "data_dir": str(db_path.parent),
                      "kind": kind, "header": header})
    if foreground:
        return process_job(job)
    spawn_worker(job, db_path.parent)
    return True


def main(default_client=None, default_event=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--client", choices=("claude", "codex"), default=default_client)
    ap.add_argument("--event", choices=("PreCompact", "SessionEnd", "Stop", "UserPromptSubmit"),
                    default=default_event)
    ap.add_argument("--data-dir", type=Path)
    ap.add_argument("--worker", type=Path)
    ap.add_argument("--drain", action="store_true")
    ap.add_argument("--foreground", action="store_true", help="Process synchronously (tests/manual exports)")
    args = ap.parse_args()
    if args.data_dir:
        os.environ["EMBER4_DATA_DIR"] = str(args.data_dir.expanduser().resolve())
    try:
        if args.worker:
            return 0 if process_job(args.worker) else 1
        if args.drain:
            from ember4.config import get_data_dir
            jobs = sorted((get_data_dir() / "archive" / "pending").glob("*.json"))
            # One process per job keeps cached config isolated across stores.
            codes = [subprocess.run([sys.executable, __file__, "--worker", str(job)]).returncode
                     for job in jobs]
            return int(any(codes))
        if not args.client:
            ap.error("--client is required for lifecycle events")
        payload = json.load(sys.stdin)
        if not isinstance(payload, dict):
            raise ValueError("hook input must be a JSON object")
        event = args.event or payload.get("hook_event_name")
        if event in ("Stop", "UserPromptSubmit"):
            timing(event, payload, args.client)
            return 0
        if event not in ("PreCompact", "SessionEnd"):
            raise ValueError(f"unsupported hook event: {event}")
        return 0 if queue_export(payload, args.client, event, foreground=args.foreground) else 1
    except Exception as exc:
        print(f"[ember-4] hook failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
