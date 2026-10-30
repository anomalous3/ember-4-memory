"""Import Claude (claude.ai) and ChatGPT data exports into an Ember store.

Conversations become archive chunks (the history layer): one readable
markdown transcript per conversation, dated to when the conversation
happened, searchable with archive_search / archive_grep / deep_recall.
Claude's exported memories (the general summary, per-project summaries and
memory files) and project knowledge/instructions become reference chunks.
With --memories-as-embers the memories become embers instead.

No model calls. Nothing is summarized or rewritten; turning conversations
into embers is judgment work that belongs to a later pass.

Usage:
    python ember4_import_exports.py PATH [PATH ...] [options]

PATH can be an export .zip, an unzipped export directory, or a single
.json file. Format is detected from the JSON shape, not the filename, so
split exports (conversations-000.zip, memories-000.zip, ...) and older
single-zip exports both work.

Options:
    --dry-run            Parse and report; write nothing.
    --project NAME       Archive project for imported chunks (default: imported).
    --data-dir DIR       Ember data dir (default: EMBER4_DATA_DIR or the usual one).
    --since YYYY-MM-DD   Skip conversations last updated before this date.
    --include-thinking   Keep Claude extended-thinking / ChatGPT reasoning text.
    --memories-as-embers Store exported Claude memories as embers instead of
                         archive references (they're already distilled, but
                         another assistant's picture of you; opt in).
    --max-chunk-chars N  Split long conversations at section boundaries (default 120000).

Re-running is safe. Unchanged conversations are skipped by content hash. A
conversation that grew since the last import gets its new version, and the
older chunks are set to status 'archived' (kept, not deleted).

Exports are untrusted input: run with `python -I` and keep this script
outside the export directory.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import io
import json
import os
import re
import sys
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

# `python -I` leaves the script's own directory off sys.path; put the repo back
# (never the current directory, which may be the untrusted export).
sys.path.insert(0, str(Path(__file__).resolve().parent))

MAX_TOOL_CHARS = 4000  # tool inputs/results are clipped; the export keeps the rest


# ── Loading ─────────────────────────────────────────────────────────────────

def iter_json_docs(path: Path):
    """Yield (label, parsed_json) for every .json/.jsonl in a zip, dir, or file."""
    if path.is_dir():
        for p in sorted(path.rglob("*")):
            if p.is_file() and p.suffix.lower() in (".json", ".jsonl", ".zip"):
                yield from iter_json_docs(p)
        return
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as zf:
            for name in sorted(zf.namelist()):
                low = name.lower()
                if low.endswith(".zip"):
                    inner = zipfile.ZipFile(io.BytesIO(zf.read(name)))
                    for iname in sorted(inner.namelist()):
                        if iname.lower().endswith((".json", ".jsonl")):
                            yield from _parse(f"{path.name}!{name}!{iname}", inner.read(iname))
                elif low.endswith((".json", ".jsonl")):
                    yield from _parse(f"{path.name}!{name}", zf.read(name))
        return
    if path.suffix.lower() in (".json", ".jsonl"):
        yield from _parse(path.name, path.read_bytes())


def _parse(label, data: bytes):
    text = data.decode("utf-8-sig", errors="replace")
    try:
        yield label, json.loads(text)
        return
    except ValueError:
        pass
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except ValueError:
                print(f"  ! {label}: skipping unparseable line", file=sys.stderr)
    if rows:
        yield label, rows


def _records(doc):
    if isinstance(doc, list):
        return [r for r in doc if isinstance(r, dict)]
    if isinstance(doc, dict):
        for key in ("conversations", "projects", "memories", "data", "items"):
            if isinstance(doc.get(key), list):
                return [r for r in doc[key] if isinstance(r, dict)]
        return [doc]
    return []


def classify(label, doc):
    """Return (kind, records) where kind names the export shape, or (None, [])."""
    recs = _records(doc)
    if not recs:
        return None, []
    r = recs[0]
    if "mapping" in r and ("current_node" in r or "title" in r):
        return "chatgpt-conversations", recs
    if "chat_messages" in r:
        return "claude-conversations", recs
    if any(k in r for k in ("conversations_memory", "project_memories", "memory")) or \
            "memor" in label.lower():
        return "claude-memories", recs
    if "docs" in r or "prompt_template" in r or "project" in label.lower() and "name" in r:
        return "claude-projects", recs
    return None, []


# ── Normalized model ────────────────────────────────────────────────────────

@dataclass
class Turn:
    role: str
    text: str
    when: str = ""


@dataclass
class Conversation:
    source: str            # "claude-ai" | "chatgpt"
    conv_id: str
    title: str
    created: str           # ISO 8601 UTC
    updated: str
    model: str = ""
    overview: str = ""     # summary the source app generated, if any
    turns: list[Turn] = field(default_factory=list)


def iso(value) -> str:
    if value in (None, ""):
        return ""
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value, tz=timezone.utc).isoformat()
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).isoformat()
    except ValueError:
        return ""


def clip(text: str, limit: int = MAX_TOOL_CHARS) -> str:
    text = text if isinstance(text, str) else json.dumps(text, ensure_ascii=False)
    return text if len(text) <= limit else text[:limit] + f"\n[... {len(text) - limit} chars clipped]"


# ── Claude ──────────────────────────────────────────────────────────────────

def claude_main_branch(messages: list[dict]) -> list[dict]:
    """claude.ai exports flatten edits and retries into one list, linked by
    parent_message_uuid, without recording which branch was on screen. Follow
    the path to the most recently written message: where the user left off."""
    by_id = {m.get("uuid"): m for m in messages}
    if not any(m.get("parent_message_uuid") in by_id for m in messages):
        return messages  # older exports: already linear
    has_child = {m.get("parent_message_uuid") for m in messages}
    leaves = [m for m in messages if m.get("uuid") not in has_child] or messages
    node = max(leaves, key=lambda m: m.get("created_at") or "")
    chain, seen = [], set()
    while node is not None and node.get("uuid") not in seen:
        seen.add(node.get("uuid"))
        chain.append(node)
        node = by_id.get(node.get("parent_message_uuid"))
    return chain[::-1]


def _claude_tool_result(b: dict) -> str:
    body = b.get("content")
    if isinstance(body, list):
        lines = []
        for x in body:
            if not isinstance(x, dict):
                continue
            if x.get("type") == "text":
                lines.append(x.get("text", ""))
            elif x.get("title") or x.get("url"):  # search hits
                lines.append(f"- {x.get('title', '')} {x.get('url', '')}".rstrip())
        body = "\n".join(lines)
    return clip(body or "")


def claude_conversation(c: dict, thinking: bool) -> Conversation:
    conv = Conversation(
        source="claude-ai", conv_id=str(c.get("uuid") or c.get("id") or ""),
        title=(c.get("name") or "").strip() or "(untitled)",
        created=iso(c.get("created_at")), updated=iso(c.get("updated_at") or c.get("created_at")),
        model=c.get("model") or "", overview=(c.get("summary") or "").strip(),
    )
    for m in claude_main_branch(c.get("chat_messages") or []):
        role = {"human": "User", "assistant": "Assistant"}.get(m.get("sender"), (m.get("sender") or "?").title())
        parts = []
        blocks = m.get("content")
        if isinstance(blocks, list) and blocks:
            for b in blocks:
                if not isinstance(b, dict):
                    continue
                t = b.get("type")
                if t == "text" and b.get("text"):
                    parts.append(b["text"])
                elif t == "thinking" and thinking:
                    # Newer exports hide the raw thinking but keep its summaries.
                    th = b.get("thinking") or "\n".join(
                        s.get("summary", "") for s in b.get("summaries") or [] if isinstance(s, dict))
                    if th.strip():
                        parts.append("> **Thinking**\n> " + th.strip().replace("\n", "\n> "))
                elif t == "tool_use":
                    parts.append(f"**Tool call: {b.get('name', '?')}**\n```\n{clip(b.get('input', ''))}\n```")
                elif t == "tool_result":
                    parts.append(f"**Tool result: {b.get('name', '')}**\n```\n{_claude_tool_result(b)}\n```")
                # injected_prompt_block (memory snapshots re-sent each turn) is skipped
        elif m.get("text"):
            parts.append(m["text"])
        for a in m.get("attachments") or []:
            name = a.get("file_name") or "attachment"
            body = a.get("extracted_content")
            parts.append(f"**Attachment: {name}**" + (f"\n```\n{clip(body, 20000)}\n```" if body else ""))
        for f in m.get("files") or m.get("files_v2") or []:
            parts.append(f"[File: {f.get('file_name', 'file')}]")
        text = "\n\n".join(p for p in parts if p and p.strip())
        if text:
            conv.turns.append(Turn(role, text, iso(m.get("created_at"))))
    return conv


def claude_memory_items(rec: dict, project_names: dict):
    """Yield (name, content, tags) for each memory text in a memories record."""
    if isinstance(rec.get("conversations_memory"), str) and rec["conversations_memory"].strip():
        yield "claude.ai memory (general)", rec["conversations_memory"].strip(), []
    pm = rec.get("project_memories")
    if isinstance(pm, dict):
        for key, text in pm.items():
            if isinstance(text, str) and text.strip():
                label = project_names.get(key, key)
                yield (f"claude.ai memory (project: {label})",
                       f"claude.ai project memory for '{label}':\n\n{text.strip()}",
                       [f"for-project:{key}"])
    elif isinstance(pm, list):
        for p in pm:
            if isinstance(p, dict):
                text = p.get("memory") or p.get("content") or ""
                key = p.get("project_name") or p.get("project_uuid") or "?"
                if text.strip():
                    yield f"claude.ai memory (project {key})", text.strip(), [f"for-project:{key}"]
    for f in rec.get("memory_files") or []:
        if isinstance(f, dict) and (f.get("content") or "").strip():
            path = f.get("path") or "memory"
            yield (f"claude.ai memory file {path}",
                   f"claude.ai memory file {path}:\n\n{f['content'].strip()}",
                   ["claude-ai-memory-file"])
    if isinstance(rec.get("memory"), str) and rec["memory"].strip():
        yield "claude.ai memory", rec["memory"].strip(), []


def claude_project_doc(p: dict) -> tuple[str, str, str]:
    name = p.get("name") or "(unnamed project)"
    out = [f"# claude.ai project: {name}", ""]
    if p.get("description"):
        out += [p["description"], ""]
    if p.get("prompt_template"):
        out += ["## Project instructions", "", p["prompt_template"], ""]
    for d in p.get("docs") or []:
        out += [f"## Knowledge: {d.get('filename', 'doc')}", "", d.get("content", ""), ""]
    return str(p.get("uuid") or name), name, "\n".join(out)


# ── ChatGPT ─────────────────────────────────────────────────────────────────

def chatgpt_conversation(c: dict, thinking: bool) -> Conversation:
    conv = Conversation(
        source="chatgpt", conv_id=str(c.get("conversation_id") or c.get("id") or ""),
        title=(c.get("title") or "").strip() or "(untitled)",
        created=iso(c.get("create_time")), updated=iso(c.get("update_time") or c.get("create_time")),
        model=c.get("default_model_slug") or "",
    )
    mapping = c.get("mapping") or {}
    # Follow the branch the user last saw: current_node back to the root.
    node, chain, seen = c.get("current_node"), [], set()
    if node not in mapping:  # fall back to the deepest leaf
        leaves = [k for k, v in mapping.items() if not v.get("children")]
        node = leaves[-1] if leaves else None
    while node and node in mapping and node not in seen:
        seen.add(node)
        chain.append(mapping[node])
        node = mapping[node].get("parent")
    for n in reversed(chain):
        m = n.get("message")
        if not m:
            continue
        meta = m.get("metadata") or {}
        if meta.get("is_visually_hidden_from_conversation"):
            continue
        role = (m.get("author") or {}).get("role", "?")
        if role == "system":
            continue
        content = m.get("content") or {}
        ctype = content.get("content_type")
        text = ""
        if ctype in ("text", "multimodal_text"):
            bits = []
            for part in content.get("parts") or []:
                if isinstance(part, str):
                    bits.append(part)
                elif isinstance(part, dict):
                    if part.get("content_type") == "image_asset_pointer":
                        bits.append("[Image]")
                    elif part.get("content_type") in ("audio_transcription",) and part.get("text"):
                        bits.append(part["text"])
                    elif part.get("text"):
                        bits.append(part["text"])
            text = "\n".join(b for b in bits if b)
        elif ctype == "code":
            text = f"```{content.get('language') or ''}\n{clip(content.get('text', ''))}\n```"
        elif ctype == "execution_output":
            text = f"```\n{clip(content.get('text', ''))}\n```"
        elif ctype in ("thoughts", "reasoning_recap"):
            if not thinking:
                continue
            if ctype == "thoughts":
                text = "\n".join(t.get("content", "") for t in content.get("thoughts") or [] if isinstance(t, dict))
            else:
                text = content.get("content", "")
            text = "> " + text.replace("\n", "\n> ") if text else ""
        elif ctype == "tether_quote":
            text = f"> {content.get('title', '')}\n> {clip(content.get('text', ''), 1500)}"
        elif ctype == "user_editable_context":
            continue  # custom instructions, repeated in every conversation
        elif isinstance(content.get("text"), str):
            text = content["text"]
        if not text.strip():
            continue
        label = {"user": "User", "assistant": "Assistant"}.get(role, "Tool")
        if role == "tool":
            label = f"Tool: {(m.get('author') or {}).get('name') or 'tool'}"
            text = clip(text)
        conv.turns.append(Turn(label, text, iso(m.get("create_time"))))
    return conv


# ── Rendering ───────────────────────────────────────────────────────────────

SOURCE_NAME = {"claude-ai": "claude.ai", "chatgpt": "ChatGPT"}


def render(conv: Conversation, max_chars: int) -> list[str]:
    head = [f"# {conv.title}", "",
            f"- Source: {SOURCE_NAME[conv.source]} export",
            f"- Conversation: {conv.conv_id}",
            f"- Started: {conv.created[:16].replace('T', ' ')} UTC",
            f"- Last updated: {conv.updated[:16].replace('T', ' ')} UTC",
            f"- Turns: {len(conv.turns)}"]
    if conv.model:
        head.append(f"- Model: {conv.model}")
    if conv.overview:
        head += ["", "## Overview (generated by the source app)", "", conv.overview]
    header = "\n".join(head) + "\n\n---\n\n"
    sections = []
    for i, t in enumerate(conv.turns, 1):
        stamp = f" ({t.when[:16].replace('T', ' ')})" if t.when else ""
        sections.append(f"## [{i}] {t.role}{stamp}\n\n{t.text}\n\n")
    chunks, cur = [], header
    for s in sections:
        while len(s) > max_chars:  # one enormous turn: hard split
            if len(cur) > len(header):
                chunks.append(cur)
                cur = header
            chunks.append(cur + s[:max_chars])
            s, cur = s[max_chars:], header
        if len(cur) + len(s) > max_chars and len(cur) > len(header):
            chunks.append(cur)
            cur = header
        cur += s
    if len(cur) > len(header) or not chunks:
        chunks.append(cur)
    if len(chunks) > 1:
        chunks = [c.replace(f"# {conv.title}\n", f"# {conv.title} (part {i}/{len(chunks)})\n", 1)
                  for i, c in enumerate(chunks, 1)]
    return chunks


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ── Writing ─────────────────────────────────────────────────────────────────

class Writer:
    def __init__(self, project: str, dry_run: bool):
        self.project, self.dry = project, dry_run
        self.stats = {"new": 0, "updated": 0, "unchanged": 0, "embers": 0, "embers_skipped": 0, "refs": 0}
        self.server = None

    async def open(self):
        if self.dry:
            return
        from ember4 import server
        await server._ensure_init()
        self.server = server
        self.db = server._db

    async def close(self):
        if self.server and self.server._db is not None:
            await self.server._db.close()

    async def _active_chunks(self, key_tag: str):
        cur = await self.db.conn.execute(
            "SELECT chunk_id, content_hash FROM archive WHERE status='active' AND tags LIKE ?",
            (f'%"{key_tag}"%',))
        return await cur.fetchall()

    async def put_document(self, key_tag: str, base_id: str, chunks: list[str], summary: str,
                           tags: list[str], chunk_type: str, from_agent: str, created: str,
                           about: str = ""):
        """Publish one document's chunks; retire its older version if it changed."""
        version = sha("\x00".join(chunks))[:12]
        if self.dry:
            self.stats["new"] += 1
            return
        existing = await self._active_chunks(key_tag)
        if existing and {h for _, h in existing} == {sha(c) for c in chunks}:
            self.stats["unchanged"] += 1
            return
        now = datetime.now(timezone.utc).isoformat()
        conn = self.db.conn
        new_ids = []
        for i, content in enumerate(chunks, 1):
            cid = f"{base_id}-{version}-p{i:03d}"
            part = f" (part {i}/{len(chunks)})" if len(chunks) > 1 else ""
            # Like archive_store: embed what the chunk is about, not all of it.
            _, emb = self.server._embed(f"{summary}{part} {about[:1500]} " + " ".join(tags))
            await conn.execute("DELETE FROM archive WHERE chunk_id=?", (cid,))  # rerun after a crash
            await conn.execute("DELETE FROM vec_archive WHERE chunk_id=?", (cid,))
            await conn.execute(
                """INSERT INTO archive (chunk_id, summary, content, tags, project, domain, chunk_type,
                       from_agent, for_agent, entity, category, reply_to, status, tokens_estimate,
                       content_hash, access_count, last_accessed_at, created_at)
                   VALUES (?,?,?,?,?,'',?,?,'','','import','','active',?,?,0,?,?)""",
                (cid, summary + part, content, json.dumps(tags + [key_tag, f"part-{i}"]),
                 self.project, chunk_type, from_agent, len(content) // 4, sha(content),
                 now, created or now))
            await conn.execute("INSERT INTO vec_archive(chunk_id, embedding) VALUES (?, ?)", (cid, emb))
            new_ids.append(cid)
        for old_id, _ in existing:
            if old_id not in new_ids:
                cur = await conn.execute("SELECT tags FROM archive WHERE chunk_id=?", (old_id,))
                old_tags = json.loads((await cur.fetchone())[0])
                old_tags.append(f"superseded-by:{new_ids[0]}")
                await conn.execute(
                    "UPDATE archive SET status='archived', archived_at=?, tags=? WHERE chunk_id=?",
                    (now, json.dumps(old_tags), old_id))
        await conn.commit()
        self.stats["updated" if existing else "new"] += 1

    async def put_conversation(self, conv: Conversation, max_chars: int):
        chunks = render(conv, max_chars)
        src = SOURCE_NAME[conv.source]
        await self.put_document(
            key_tag=f"conversation:{conv.conv_id}",
            base_id=f"import-{conv.source}-{conv.conv_id}",
            chunks=chunks, summary=f"[{src}] {conv.title}",
            tags=["imported", f"source:{conv.source}", "conversation"] + ([f"model:{conv.model}"] if conv.model else []),
            chunk_type="session", from_agent=f"import:{conv.source}", created=conv.created,
            about=conv.overview or (conv.turns[0].text if conv.turns else ""))

    async def put_reference(self, key_tag: str, base_id: str, summary: str,
                            content: str, tags: list[str]):
        self.stats["refs"] += 1
        chunks = [content[i:i + 1_800_000] for i in range(0, len(content), 1_800_000)] or [content]
        await self.put_document(
            key_tag=key_tag, base_id=base_id, chunks=chunks, summary=summary,
            tags=["imported", "source:claude-ai"] + tags, chunk_type="reference",
            from_agent="import:claude-ai", created="", about=content)

    async def put_memory(self, name: str, content: str, tags: list[str]):
        if self.dry:
            self.stats["embers"] += 1
            return
        cur = await self.db.conn.execute("SELECT 1 FROM embers WHERE content=? LIMIT 1", (content,))
        if await cur.fetchone():
            self.stats["embers_skipped"] += 1
            return
        await self.server.ember_store(
            name=name, content=content, importance="context",
            tags=",".join(["imported", "source:claude-ai", "claude-ai-memory"] + tags))
        self.stats["embers"] += 1


# ── Main ────────────────────────────────────────────────────────────────────

async def run(args) -> int:
    since = iso(args.since) if args.since else ""
    convs: dict[tuple[str, str], Conversation] = {}
    memories, projects, unknown = [], [], []
    found: list[tuple[str, list]] = []
    emptied = 0
    for path in args.paths:
        p = Path(path).expanduser()
        if not p.exists():
            print(f"! not found: {p}", file=sys.stderr)
            return 2
        for label, doc in iter_json_docs(p):
            kind, recs = classify(label, doc)
            if kind is None:
                unknown.append(label)
                continue
            print(f"  {label}: {kind} ({len(recs)})")
            found.append((kind, recs))
    # Projects first, so project memories can be labeled by name.
    project_names = {}
    for kind, recs in found:
        if kind == "claude-projects":
            for r in recs:
                project_names[str(r.get("uuid"))] = r.get("name") or str(r.get("uuid"))
                if not r.get("is_starter_project"):
                    projects.append(claude_project_doc(r))
    for kind, recs in found:
        if kind == "claude-memories":
            for r in recs:
                memories.extend(claude_memory_items(r, project_names))
        if kind not in ("claude-conversations", "chatgpt-conversations"):
            continue
        for r in recs:
            if kind == "claude-conversations":
                c = claude_conversation(r, args.include_thinking)
            else:
                c = chatgpt_conversation(r, args.include_thinking)
            if not c.turns or not c.conv_id:
                emptied += 1  # e.g. claude.ai conversations whose text was purged upstream
                continue
            if since and (c.updated or c.created) < since:
                continue
            key = (c.source, c.conv_id)
            if key not in convs or c.updated > convs[key].updated:
                convs[key] = c  # overlapping exports: newest version wins
    if unknown:
        print(f"  (ignored {len(unknown)} file(s) with no recognized shape: "
              f"{', '.join(unknown[:6])}{' ...' if len(unknown) > 6 else ''})")

    by_src = {}
    for (src, _), c in convs.items():
        by_src[src] = by_src.get(src, 0) + 1
    print(f"\nFound: {by_src or 'no conversations'}, {len(memories)} memory text(s), "
          f"{len(projects)} project(s)"
          + (f"; skipped {emptied} conversation(s) with no message text" if emptied else ""))

    w = Writer(args.project, args.dry_run)
    await w.open()
    try:
        for c in sorted(convs.values(), key=lambda c: c.created):
            await w.put_conversation(c, args.max_chunk_chars)
        for key, name, content in projects:
            await w.put_reference(f"claude-project:{key}", f"import-claude-ai-project-{key}",
                                  f"[claude.ai] project: {name}", content, ["project-knowledge"])
        for name, content, tags in memories:
            if args.memories_as_embers:
                await w.put_memory(name, content, tags)
            else:
                mkey = sha(name)[:16]  # stable per memory slot; new text supersedes old
                await w.put_reference(f"claude-memory:{mkey}", f"import-claude-ai-memory-{mkey}",
                                      f"[claude.ai] {name}", content, ["claude-ai-memory"] + tags)
    finally:
        await w.close()
    s = w.stats
    verb = "Would import" if args.dry_run else "Imported"
    print(f"{verb}: {s['new']} new, {s['updated']} updated, {s['unchanged']} unchanged documents "
          f"(incl. {s['refs']} project/memory references)"
          + (f"; {s['embers']} memory embers" if s["embers"] else "")
          + (f", {s['embers_skipped']} already present" if s["embers_skipped"] else ""))
    if not args.dry_run and s["embers"]:
        print("New embers get graph edges on the next ember4_build_graph.py run "
              "(the SessionEnd hook does this).")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--project", default="imported")
    ap.add_argument("--data-dir")
    ap.add_argument("--since")
    ap.add_argument("--include-thinking", action="store_true")
    ap.add_argument("--memories-as-embers", action="store_true")
    ap.add_argument("--max-chunk-chars", type=int, default=120_000)
    args = ap.parse_args()
    if args.data_dir:
        os.environ["EMBER4_DATA_DIR"] = str(Path(args.data_dir).expanduser())
    sys.exit(asyncio.run(run(args)))


if __name__ == "__main__":
    main()
