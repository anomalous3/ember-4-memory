#!/usr/bin/env python3
"""
Pre-Compact Export Hook — Full session transcript preservation.

Triggered BEFORE /compact (manual or auto). Copies the current session's
JSONL conversation log, parses it into a readable markdown transcript,
and writes it as archive chunk(s) for BM25 search and line-range navigation.

The raw JSONL is also saved to the archive exports directory as a backup.
Prints a one-line status. export_session() is shared with the SessionEnd hook.
"""

import json
import hashlib
import os
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from ember4.config import get_data_dir, get_db_path

# ── Constants ──────────────────────────────────────────────────────────

PROJECTS_DIR = Path.home() / ".claude" / "projects"

# Raw JSONL backups. EMBER4_DATA_DIR moves them along with the database.
EXPORTS_DIR = get_data_dir() / "archive" / "exports"

MAX_CHUNK_BYTES = 1_800_000   # 1.8MB per chunk (under 2MB limit)
MAX_TOOL_RESULT_LINES = 200   # Truncate individual tool results in transcript
MAX_TOC_ENTRIES = 150          # Cap table of contents length

def session_home(messages, fallback):
    """The directory the session started in, from its first transcript entry.

    The hook's own cwd is wherever the session has wandered to by now; the
    archive label and persona routing should follow where it lives.
    """
    for m in messages:
        if m.get("cwd"):
            return m["cwd"]
    return fallback


# ── JSONL Parsing ────────────────────────────────────────────────────

def find_project_dir(cwd=None):
    """Find the Claude Code project directory for the given working dir.

    Tries the exact CWD first, then walks up parent directories. This handles
    cases where the CWD is a subdirectory (e.g., src/) of the actual
    project root.
    """
    cwd = cwd or os.getcwd()
    # Try exact match, then walk up parents. A filesystem root is its own
    # parent; comparing str(parent) with parent.root never ends on Windows,
    # where "C:\\" != "\\".
    candidates = [cwd]
    parent = Path(cwd).parent
    while parent != parent.parent:
        candidates.append(str(parent))
        parent = parent.parent

    for candidate in candidates:
        # Claude Code names project dirs by replacing every non-alphanumeric
        # character with "-" (C:\Users\me\proj -> C--Users-me-proj); the
        # "/"-only form is kept for stores created by older versions.
        for slug in dict.fromkeys([re.sub(r"[^A-Za-z0-9]", "-", candidate),
                                   candidate.replace("/", "-")]):
            for name in (slug, slug.lstrip("-")):
                if Path(name).anchor:
                    # still absolute (a Windows path has no "/"): joining it
                    # would return the path itself, not a project dir
                    continue
                project_dir = PROJECTS_DIR / name
                if project_dir.exists():
                    return project_dir

    raise FileNotFoundError(f"No project dir for {cwd} (checked {len(candidates)} parents)")


def find_session_file(session_id, project_dir):
    """Find the JSONL file for a given session or the most recent one."""
    if session_id:
        jsonl = project_dir / f"{session_id}.jsonl"
        if jsonl.exists():
            return jsonl
        raise FileNotFoundError(f"No transcript for session {session_id}")
    # Most recent by mtime
    jsonl_files = sorted(
        project_dir.glob("*.jsonl"),
        key=lambda f: f.stat().st_mtime,
        reverse=True,
    )
    if not jsonl_files:
        raise FileNotFoundError(f"No session files in {project_dir}")
    return jsonl_files[0]


def extract_content_text(msg):
    """Extract readable text content from any JSONL message type."""
    parts = []

    if msg.get("toolUseResult"):
        result = msg["toolUseResult"]
        if isinstance(result, dict):
            r = result.get("result", "")
            parts.append(r if isinstance(r, str) else json.dumps(r, default=str))
        elif isinstance(result, str):
            parts.append(result)

    message = msg.get("message", {})
    content = message.get("content", "")
    if isinstance(content, str):
        parts.append(content)
    elif isinstance(content, list):
        for block in content:
            if not isinstance(block, dict):
                continue
            btype = block.get("type", "")
            if btype == "text":
                parts.append(block.get("text", ""))
            elif btype == "tool_result":
                c = block.get("content", "")
                if isinstance(c, str):
                    parts.append(c)
                elif isinstance(c, list):
                    for sub in c:
                        if isinstance(sub, dict) and sub.get("type") == "text":
                            parts.append(sub.get("text", ""))
            elif btype == "tool_use":
                name = block.get("name", "unknown")
                inp = block.get("input", {})
                call_parts = [f"[Tool Call: {name}]"]
                for key, val in inp.items():
                    val_str = str(val)
                    if len(val_str) > 500:
                        val_str = val_str[:500] + "..."
                    call_parts.append(f"  {key}: {val_str}")
                parts.append("\n".join(call_parts))

    data = msg.get("data", {})
    if data.get("fullOutput"):
        parts.append(data["fullOutput"])
    elif data.get("output"):
        parts.append(data["output"])

    return "\n".join(p for p in parts if p)


def build_tool_name_index(messages):
    """Map tool_use_id → tool_name from assistant messages."""
    index = {}
    for msg in messages:
        if msg.get("type") != "assistant":
            continue
        content = msg.get("message", {}).get("content", [])
        if not isinstance(content, list):
            continue
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_use":
                tid = block.get("id", "")
                name = block.get("name", "")
                if tid and name:
                    index[tid] = name
    return index


# ── Transcript Generation ─────────────────────────────────────────────

def classify_message(msg, tool_index):
    """Classify a message and return (label, detail)."""
    msg_type = msg.get("type", "unknown")

    if msg_type == "assistant":
        content = msg.get("message", {}).get("content", [])
        tool_calls = []
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_use":
                    tool_calls.append(block.get("name", "unknown"))
        if tool_calls:
            return "Assistant + Tools", ", ".join(tool_calls)
        return "Assistant", ""

    if msg_type == "user":
        if msg.get("userType") == "external":
            return "User", ""
        content = msg.get("message", {}).get("content", [])
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    tool_use_id = block.get("tool_use_id", "")
                    tool_name = tool_index.get(tool_use_id, "unknown")
                    return f"Tool Result: {tool_name}", ""
        if msg.get("toolUseResult"):
            return "Tool Result", ""
        return "User (internal)", ""

    if msg_type == "progress":
        return "Progress", ""
    if msg_type == "system":
        return "System", msg.get("subtype", "")

    return msg_type, ""


def truncate_lines(text, max_lines):
    """Truncate text to max_lines, adding a note if truncated."""
    lines = text.split("\n")
    if len(lines) <= max_lines:
        return text
    kept = lines[:max_lines]
    omitted = len(lines) - max_lines
    kept.append(f"\n[... {omitted} more lines — see raw JSONL export ...]")
    return "\n".join(kept)


def build_transcript(messages, tool_index):
    """Convert parsed JSONL messages into a readable markdown transcript."""
    sections = []
    toc_entries = []

    skip_types = {"progress"}
    skip_subtypes = {"compact_boundary", "microcompact_boundary"}

    msg_num = 0
    for msg in messages:
        msg_type = msg.get("type", "")

        if msg_type in skip_types:
            continue
        if msg.get("subtype", "") in skip_subtypes:
            continue

        msg_num += 1
        label, detail = classify_message(msg, tool_index)
        content = extract_content_text(msg)

        if not content.strip():
            continue

        if "Tool Result" in label:
            content = truncate_lines(content, MAX_TOOL_RESULT_LINES)

        header = f"## [{msg_num}] {label}"
        if detail:
            header += f" ({detail})"

        sections.append(f"{header}\n\n{content}\n")

        if len(toc_entries) < MAX_TOC_ENTRIES:
            preview = content.replace("\n", " ")[:80].strip()
            toc_entries.append(f"- [{msg_num}] {label}: {preview}")

    if len(toc_entries) >= MAX_TOC_ENTRIES:
        toc_entries.append(f"- ... and {msg_num - MAX_TOC_ENTRIES} more messages")

    toc = "## Table of Contents\n\n" + "\n".join(toc_entries) + "\n\n---\n\n"
    return toc, sections


# ── Archive Chunk Writing ─────────────────────────────────────────────

def archive_db_path(cwd):
    """Where a session's archive goes: its persona's own store, or the default.

    Personas with their own ember-4 store are listed in persona_stores.json in
    the default data dir, as {"<home dir of the persona>": "<its data dir>"}.
    A session belongs to a persona when it runs in that dir or below it.
    """
    if not os.environ.get("EMBER4_DATA_DIR"):
        try:
            stores = json.loads((get_db_path().parent / "persona_stores.json").read_text())
            here = Path(cwd or os.getcwd()).expanduser().resolve()
            for home, data_dir in stores.items():
                root = Path(home).expanduser().resolve()
                if here == root or root in here.parents:
                    return Path(data_dir).expanduser() / "ember4.db"
        except FileNotFoundError:
            pass
        except Exception as e:
            print(f"[export] WARNING: persona_stores.json unreadable: {e}", file=sys.stderr)
    return get_db_path()


def write_archive_chunk(chunk_id, summary, content, tags_list, project, cwd=None,
                        chunk_type="compact-export", from_agent="hook:pre-compact"):
    """Write a chunk to the ember-4 archive (same store the MCP server reads).

    Uses synchronous sqlite3 (hooks run outside the async MCP server).
    """
    import sqlite3
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
    content_hash = hashlib.sha256(
        content.lower().encode("utf-8", errors="replace")
    ).hexdigest()
    tokens_est = len(content) // 4

    db_path = archive_db_path(cwd)
    if not db_path.exists():
        print(f"[export] WARNING: {db_path} not found, skipping archive write", file=sys.stderr)
        return None

    try:
        import sqlite_vec
        # the live MCP server holds the DB open (WAL); wait rather than fail on a lock
        db = sqlite3.connect(str(db_path), timeout=15)
        db.enable_load_extension(True)
        sqlite_vec.load(db)
    except Exception as e:
        print(f"[export] WARNING: Could not open ember-4 database: {e}", file=sys.stderr)
        return None

    try:
        # Dedup check
        row = db.execute(
            "SELECT chunk_id FROM archive WHERE content_hash = ? LIMIT 1",
            (content_hash,),
        ).fetchone()
        if row:
            db.close()
            return row[0]  # Already exists

        # Insert archive chunk (FTS5 triggers handle indexing automatically)
        db.execute(
            """INSERT INTO archive (
                chunk_id, summary, content, tags, project, domain, chunk_type,
                from_agent, for_agent, entity, category, reply_to, status,
                tokens_estimate, content_hash, access_count, last_accessed_at, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, '', '', '', '', 'active', ?, ?, 0, ?, ?)""",
            (
                chunk_id, summary, content, json.dumps(tags_list), project, "",
                chunk_type, from_agent, tokens_est, content_hash, now, now,
            ),
        )
        db.commit()
    except Exception as e:
        print(f"[export] WARNING: Archive write failed: {e}", file=sys.stderr)
        db.close()
        return None

    db.close()
    return chunk_id


# ── Stdin Parsing ─────────────────────────────────────────────────────

def parse_stdin():
    """Read context stats and session info from stdin."""
    ctx_pct = 0
    session_id = None
    cwd = None
    custom = ""

    try:
        input_data = sys.stdin.read()
        if not input_data:
            return ctx_pct, session_id, cwd, custom

        data = json.loads(input_data)
        ctx_window = data.get("context_window", {})
        usage = ctx_window.get("current_usage", {})
        size = ctx_window.get("context_window_size", 1)

        if usage and size > 0:
            current = (
                usage.get("input_tokens", 0)
                + usage.get("cache_creation_input_tokens", 0)
                + usage.get("cache_read_input_tokens", 0)
            )
            ctx_pct = int(current * 100 / size)

        session_id = data.get("session_id")
        cwd = data.get("cwd")
        custom = data.get("custom_instructions") or ""
    except (json.JSONDecodeError, KeyError, TypeError):
        pass

    return ctx_pct, session_id, cwd, custom


# ── Export ────────────────────────────────────────────────────────────

KINDS = {
    "compact-export": ("hook:pre-compact", "Pre-compact full session export", "Pre-compact export"),
    "session-export": ("hook:session-end", "Session end export", "Session end export"),
}


def export_session(session_id, cwd, kind="compact-export", header="", ctx_pct=0,
                   *, transcript_path=None, client="claude", snapshot=False):
    """Backward-compatible entry point shared by both transcript clients."""
    from ember4.hooks.transcript_archive import export_session as export
    return export(session_id, cwd, kind, header, ctx_pct,
                  transcript_path=transcript_path, client=client, snapshot=snapshot)


def status_line(result):
    """One line on where the transcript went."""
    if result["chunk_ids"]:
        where = f"archive chunk(s) {', '.join(result['chunk_ids'])}"
        if result["export_path"]:
            where += f"; raw JSONL at {result['export_path']}"
    elif result["export_path"]:
        where = f"raw JSONL only, at {result['export_path']}"
    else:
        where = "not saved"
    msg = f"[ember-4] Session transcript: {where}."
    if result["error"]:
        msg += f" ({result['error']})"
    return msg


# ── Main ──────────────────────────────────────────────────────────────

def main():
    from ember4.hooks.lifecycle import main as lifecycle_main
    return lifecycle_main(default_client="claude", default_event="PreCompact")


if __name__ == "__main__":
    raise SystemExit(main())
