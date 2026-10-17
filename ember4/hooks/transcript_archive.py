"""Claude/Codex transcript snapshots and transactional archive publication.

No model calls. Raw snapshots are retained even when parsing/publication fails.
Only a byte-prefix proof authorizes retiring an older export of this session.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sqlite3
import uuid
from datetime import datetime, timezone


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def private_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    return path


def snapshot_transcript(path, cwd, client):
    """Copy a fixed byte extent before a lifecycle hook returns.

    The first record's session cwd wins over a shell that has since changed
    directory. No search for a 'most recent' session is performed here.
    """
    from ember4.hooks.pre_compact_export import archive_db_path
    source = Path(path).expanduser()
    with source.open("rb") as src:
        size = os.fstat(src.fileno()).st_size
        first = src.readline()
        try:
            record = json.loads(first)
            meta = record.get("payload", {}) if client == "codex" else record
            cwd = meta.get("cwd") or cwd
        except (ValueError, AttributeError):
            pass
        src.seek(0)
        db_path = archive_db_path(cwd)
        directory = private_dir(db_path.parent / "archive" / "exports" / client)
        target = directory / (uuid.uuid4().hex + ".jsonl")
        fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "wb") as out:
            remaining = size
            while remaining:
                block = src.read(min(1024 * 1024, remaining))
                if not block:
                    raise ValueError("transcript shrank while taking snapshot; retained partial raw copy")
                out.write(block)
                remaining -= len(block)
            out.flush()
            os.fsync(out.fileno())
    return target, cwd, db_path


def read_records(path):
    data = Path(path).read_bytes()
    records = []
    lines = data.splitlines(keepends=True)
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError("JSONL record is not an object")
            records.append(record)
        except (ValueError, UnicodeDecodeError):
            # A concurrent append may leave a trailing partial record. Its bytes
            # remain in the snapshot, but it cannot authorize supersession.
            if i == len(lines) - 1 and not line.endswith(b"\n"):
                return data, records, False
            raise ValueError(f"invalid JSONL record at line {i + 1}") from None
    return data, records, True


def _text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                if isinstance(block.get("text"), str):
                    parts.append(block["text"])
                elif block.get("type") in ("input_image", "image", "image_url"):
                    parts.append("[Image — see raw transcript]")
                elif block.get("type") in ("audio", "input_audio", "output_audio"):
                    parts.append("[Audio — see raw transcript]")
        return "\n".join(parts)
    return json.dumps(content, ensure_ascii=False) if content is not None else ""


def codex_sections(records):
    """Render canonical response items, not duplicate event-stream mirrors.

    Older event-only transcripts get a fallback. Encrypted reasoning, repeated
    world state, accounting events and opaque media remain in the raw snapshot.
    """
    items = [r.get("payload", {}) for r in records if r.get("type") == "response_item"]
    has_messages = any(p.get("type") == "message" for p in items)
    names = {p.get("call_id"): p.get("name", "unknown") for p in items
             if p.get("type") in ("function_call", "custom_tool_call")}
    entries = []
    for record in records:
        p = record.get("payload", {})
        if not isinstance(p, dict):
            continue
        kind = p.get("type")
        label, body = "", ""
        if record.get("type") == "response_item":
            if kind == "message":
                label = p.get("role", "message").title()
                if p.get("phase"):
                    label += f" ({p['phase']})"
                body = _text(p.get("content"))
            elif kind in ("function_call", "custom_tool_call"):
                label = "Tool Call: " + p.get("name", "unknown")
                body = _text(p.get("arguments", p.get("input", "")))
            elif kind in ("function_call_output", "custom_tool_call_output"):
                label = "Tool Result: " + names.get(p.get("call_id"), "unknown")
                body = _text(p.get("output"))
            elif kind == "reasoning":
                label, body = "Reasoning summary", _text(p.get("summary"))
        elif record.get("type") == "event_msg" and not has_messages:
            if kind in ("user_message", "agent_message"):
                label = "User" if kind == "user_message" else "Assistant"
                body = _text(p.get("message"))
        elif record.get("type") == "compacted":
            label, body = "Compaction summary", _text(p.get("message"))
        if body.strip():
            entries.append((label, body))
    toc = "## Table of Contents\n\n" + "\n".join(
        f"- [{i}] {label}: {body.replace(chr(10), ' ')[:80]}"
        for i, (label, body) in enumerate(entries[:150], 1)) + "\n\n---\n\n"
    sections = [f"## [{i}] {label}\n\n{body}\n" for i, (label, body) in enumerate(entries, 1)]
    return toc, sections


def split_utf8(text, limit):
    """Hard byte limit, including a single oversized message or tool result."""
    raw = text.encode("utf-8")
    while raw:
        end = min(limit, len(raw))
        if end < len(raw):
            while end and raw[end] & 0xC0 == 0x80:
                end -= 1
        if not end:
            raise ValueError("chunk byte limit too small for UTF-8")
        yield raw[:end].decode("utf-8")
        raw = raw[end:]


def _prefix(path, size, sha):
    try:
        with Path(path).open("rb") as src:
            remaining = size
            h = hashlib.sha256()
            while remaining:
                block = src.read(min(remaining, 1024 * 1024))
                if not block:
                    return False
                h.update(block)
                remaining -= len(block)
            return h.hexdigest() == sha
    except OSError:
        return False


def publish(db_path, snapshot, raw, chunks, session_id, client, kind, project, header, complete):
    """Publish all parts + retire proven prefixes in one SQLite transaction."""
    import sqlite_vec
    if not db_path.exists():
        raise FileNotFoundError(f"initialize the Ember store first: {db_path}")
    sha = digest(raw)
    # Renderer version, session and client are part of identity. Case-sensitive.
    identity = digest(json.dumps(["transcript-v1", client, session_id, sha, header],
                                 ensure_ascii=False).encode())
    ids = [f"transcript-{client}-{identity}-p{i:04d}" for i in range(1, len(chunks) + 1)]
    now = datetime.now(timezone.utc).isoformat()
    conn = sqlite3.connect(db_path, timeout=15)
    try:
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.execute("BEGIN IMMEDIATE")
        # A small manifest separates export identity/coverage from human tags.
        conn.execute("""CREATE TABLE IF NOT EXISTS transcript_exports (
            export_id TEXT PRIMARY KEY, client TEXT NOT NULL, session_id TEXT NOT NULL,
            snapshot_path TEXT NOT NULL, source_bytes INTEGER NOT NULL,
            source_hash TEXT NOT NULL, header TEXT NOT NULL, complete INTEGER NOT NULL,
            chunk_ids TEXT NOT NULL, created_at TEXT NOT NULL)""")
        previous = conn.execute("""SELECT export_id,snapshot_path,source_bytes,source_hash,
            header,complete,chunk_ids FROM transcript_exports WHERE client=? AND session_id=?""",
                                (client, session_id)).fetchall()
        for old_id, path, size, old_sha, old_header, old_complete, old_ids in previous:
            old_ids = json.loads(old_ids)
            # A delayed smaller job must never reactivate an older generation.
            active = all(conn.execute("SELECT status FROM archive WHERE chunk_id=?", (cid,)).fetchone()
                         == ("active",) for cid in old_ids)
            if active and old_complete and old_header == header and size >= len(raw):
                if _prefix(path, len(raw), sha):
                    conn.commit()
                    return old_ids
        tags = [kind, "session", "full-transcript", f"session:{session_id}", f"client:{client}"]
        if header:
            tags.append("has-header")
        for i, (cid, content) in enumerate(zip(ids, chunks), 1):
            conn.execute("""INSERT INTO archive
                (chunk_id,summary,content,tags,project,domain,chunk_type,from_agent,status,
                 tokens_estimate,content_hash,created_at)
                VALUES (?,?,?,?,?,'',?,?,'active',?,?,?)
                ON CONFLICT(chunk_id) DO UPDATE SET status='active', archived_at=NULL""",
                (cid, f"[{client}] {kind} part {i}/{len(chunks)}", content,
                 json.dumps(tags + [f"part-{i}"]), project, kind, f"hook:{client}",
                 len(content) // 4, digest(content.encode()), now))
        conn.execute("INSERT OR REPLACE INTO transcript_exports VALUES (?,?,?,?,?,?,?,?,?,?)",
                     (identity, client, session_id, str(snapshot), len(raw), sha, header,
                      int(complete), json.dumps(ids), now))
        if complete:
            for old_id, path, size, old_sha, old_header, old_complete, old_ids in previous:
                if old_id == identity or old_header != header or size > len(raw):
                    continue
                if digest(raw[:size]) != old_sha:
                    continue
                for cid in json.loads(old_ids):
                    row = conn.execute("SELECT tags FROM archive WHERE chunk_id=?", (cid,)).fetchone()
                    if row:
                        old_tags = [t for t in json.loads(row[0]) if not t.startswith("superseded-by:")]
                        old_tags.append(f"superseded-by:{ids[0]}")
                        conn.execute("UPDATE archive SET status='archived',archived_at=?,tags=? WHERE chunk_id=?",
                                     (now, json.dumps(old_tags), cid))
        conn.commit()
        return ids
    finally:
        conn.close()  # Rolls back the entire export on any error.


def export_session(session_id, cwd, kind="compact-export", header="", ctx_pct=0,
                   *, transcript_path=None, client="claude", snapshot=False):
    from ember4.hooks import pre_compact_export as legacy
    result = {"chunk_ids": [], "export_path": None, "messages": [], "cwd": cwd, "error": None}
    try:
        if client not in ("claude", "codex"):
            raise ValueError("unsupported transcript client")
        if kind not in legacy.KINDS:
            raise ValueError("unsupported export kind")
        if not transcript_path:
            if client != "claude":
                raise ValueError("Codex requires the hook's transcript_path; refusing to guess a session")
            transcript_path = legacy.find_session_file(session_id, legacy.find_project_dir(cwd))
        if snapshot:
            path = Path(transcript_path)
            db_path = legacy.archive_db_path(cwd)
        else:
            path, cwd, db_path = snapshot_transcript(transcript_path, cwd, client)
        result.update(export_path=path, cwd=cwd)
        raw, records, complete = read_records(path)
        result["messages"] = records
        if not records:
            raise ValueError("empty or incomplete transcript; raw snapshot retained")
        embedded_ids = set()
        for r in records:
            if client == "codex" and r.get("type") == "session_meta":
                p = r.get("payload", {})
                embedded_ids.add(p.get("id") or p.get("session_id"))
            elif client == "claude" and r.get("sessionId"):
                embedded_ids.add(r["sessionId"])
        embedded_ids.discard(None)
        if len(embedded_ids) > 1 or (session_id and embedded_ids and session_id not in embedded_ids):
            raise ValueError("session identity mismatch; refusing to archive another session")
        session_id = session_id or next(iter(embedded_ids), None)
        if not session_id:
            raise ValueError("session_id is required")
        if client == "codex":
            toc, sections = codex_sections(records)
        else:
            toc, sections = legacy.build_transcript(records, legacy.build_tool_name_index(records))
        if not sections:
            raise ValueError("no supported transcript messages; raw snapshot retained")
        content = header + toc + "\n".join(sections)
        chunks = list(split_utf8(content, legacy.MAX_CHUNK_BYTES))
        project = Path(cwd or os.getcwd()).name
        result["chunk_ids"] = publish(db_path, path, raw, chunks, session_id, client, kind,
                                      project, header, complete)
        if not complete:
            result["error"] = "trailing partial record retained in raw copy; no older export retired"
    except Exception as exc:
        result["error"] = str(exc)
    return result
