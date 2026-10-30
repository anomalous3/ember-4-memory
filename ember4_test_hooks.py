"""Portable hook regression suite. Synthetic transcripts + temporary stores only.

    .venv/bin/python3 -m unittest -v ember4_test_hooks

No models, network, live databases, or user hook configuration are touched.
"""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import sqlite_vec

from ember4.db import SCHEMA_SQL
from ember4.hooks import pre_compact_export as legacy
from ember4.hooks import transcript_archive as archive
from ember4.hooks import lifecycle
from ember4_install_hooks import hook_config, install, merge_config

ROOT = Path(__file__).resolve().parent
RUNNER = ROOT / "ember4" / "hooks" / "lifecycle.py"


def item(kind, **fields):
    return {"type": "response_item", "payload": {"type": kind, **fields}}


class HookTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="ember-hooks-test-")
        self.root = Path(self.temp.name)
        self.data = self.root / "data"
        self.data.mkdir()
        self.env = patch.dict(os.environ, EMBER4_DATA_DIR=str(self.data), EMBER4_READ_ONLY="0")
        self.env.start()
        self.db = self.data / "ember4.db"
        with self.connect() as conn:
            conn.executescript(SCHEMA_SQL)
        self.path = self.root / "input.jsonl"
        self.sid = "synthetic-session"

    def tearDown(self):
        self.env.stop()
        self.temp.cleanup()

    @contextmanager
    def connect(self):
        # sqlite3's own context manager commits but leaves the connection open,
        # and Windows can't delete an open database file (tearDown, and the
        # missing-store test, both do).
        conn = sqlite3.connect(self.db)
        try:
            conn.enable_load_extension(True)
            sqlite_vec.load(conn)
            with conn:
                yield conn
        finally:
            conn.close()

    def records(self, client="codex", text="A sourdough starter needs feeding."):
        if client == "claude":
            return [{"type": "user", "userType": "external", "sessionId": self.sid,
                     "cwd": str(self.root), "message": {"content": text}}]
        return [{"type": "session_meta", "payload": {"id": self.sid, "cwd": str(self.root)}},
                item("message", role="user", content=[{"type": "input_text", "text": text}])]

    def write(self, records, path=None):
        (path or self.path).write_text("".join(json.dumps(r) + "\n" for r in records))

    def export(self, client="codex", **kwargs):
        return archive.export_session(self.sid, str(self.root), transcript_path=self.path,
                                      client=client, **kwargs)

    def rows(self):
        with self.connect() as conn:
            return conn.execute("SELECT chunk_id,content,status FROM archive ORDER BY chunk_id").fetchall()

    def test_codex_messages_tools_reasoning_and_no_event_duplicates(self):
        records = self.records() + [
            {"type": "event_msg", "payload": {"type": "user_message", "message": "DUPLICATE"}},
            item("function_call", call_id="call-a", name="lookup", arguments='{"term":"starter"}'),
            item("function_call_output", call_id="call-a", output="Feed daily."),
            item("custom_tool_call", call_id="call-b", name="script", input="inspect(starter)"),
            item("custom_tool_call_output", call_id="call-b", output=[{"type": "text", "text": "Ready."}]),
            item("reasoning", summary=[{"type": "summary_text", "text": "Visible summary."}],
                 encrypted_content="OPAQUE-CIPHERTEXT"),
            item("message", role="assistant", content=[{"type": "output_text", "text": "Done."}]),
            {"type": "compacted", "payload": {"message": "A compact summary."}},
        ]
        self.write(records)
        result = self.export()
        self.assertIsNone(result["error"])
        content = "".join(row[1] for row in self.rows())
        for text in ("Tool Call: lookup", "Feed daily.", "inspect(starter)", "Ready.", "Visible summary.", "Done.", "A compact summary."):
            self.assertIn(text, content)
        self.assertNotIn("DUPLICATE", content)
        self.assertNotIn("OPAQUE-CIPHERTEXT", content)
        self.assertIn("OPAQUE-CIPHERTEXT", result["export_path"].read_text())

    def test_event_only_codex_fallback(self):
        self.write([self.records()[0], {"type": "event_msg", "payload": {
            "type": "user_message", "message": "Older format"}}])
        self.assertTrue(self.export()["chunk_ids"])
        self.assertIn("Older format", self.rows()[0][1])

    def test_claude_format_and_legacy_entrypoint(self):
        records = self.records("claude") + [
            {"type": "assistant", "message": {"content": [
                {"type": "tool_use", "id": "call-a", "name": "Read", "input": {"path": "recipe.txt"}}]}},
            {"type": "user", "message": {"content": [
                {"type": "tool_result", "tool_use_id": "call-a", "content": "flour and water"}]}},
        ]
        self.write(records)
        result = legacy.export_session(self.sid, str(self.root), transcript_path=self.path)
        self.assertIsNone(result["error"])
        self.assertEqual(result["messages"], records)
        self.assertIn("Tool Result: Read", self.rows()[0][1])
        self.assertIn("flour and water", self.rows()[0][1])

    def test_repeat_extend_and_out_of_order_exports(self):
        records = self.records()
        self.write(records)
        old = self.export()
        repeated = self.export(kind="session-export")
        self.assertEqual(old["chunk_ids"], repeated["chunk_ids"])
        records.append(item("message", role="assistant", content="Now feed it."))
        self.write(records)
        new = self.export()
        self.assertNotEqual(old["chunk_ids"], new["chunk_ids"])
        late = archive.export_session(self.sid, str(self.root), client="codex",
                                      transcript_path=old["export_path"], snapshot=True)
        self.assertEqual(late["chunk_ids"], new["chunk_ids"])
        self.assertEqual(sum(r[2] == "active" for r in self.rows()), 1)
        self.assertEqual(sum(r[2] == "archived" for r in self.rows()), 1)

    def test_rewrite_and_header_preserve_old_exports(self):
        self.write(self.records())
        self.export()
        self.write(self.records(text="Replaced transcript after a fork."))
        self.export()
        self.export(header="Unique user instructions\n")
        self.assertEqual(sum(r[2] == "active" for r in self.rows()), 3)

    def test_partial_record_retains_raw_and_does_not_retire(self):
        self.write(self.records())
        self.export()
        with self.path.open("a") as out:
            out.write('{"type":')
        result = self.export()
        self.assertTrue(result["chunk_ids"])
        self.assertIn("partial", result["error"])
        self.assertTrue(result["export_path"].read_bytes().endswith(b'{"type":'))
        self.assertEqual(sum(r[2] == "active" for r in self.rows()), 2)

    def test_invalid_middle_record_and_wrong_session_fail_closed(self):
        self.write(self.records())
        self.path.write_text(self.path.read_text() + 'broken\n' + '{}\n')
        self.assertIn("invalid JSONL", self.export()["error"])
        self.write(self.records())
        result = archive.export_session("another-session", str(self.root), client="codex", transcript_path=self.path)
        self.assertIn("identity mismatch", result["error"])
        self.assertEqual(self.rows(), [])

    def test_oversized_unicode_message_respects_chunk_bound(self):
        self.write(self.records(text="🍞" * 700))
        with patch.object(legacy, "MAX_CHUNK_BYTES", 128):
            result = self.export()
        self.assertIsNone(result["error"])
        self.assertGreater(len(self.rows()), 1)
        self.assertTrue(all(len(r[1].encode()) <= 128 for r in self.rows()))
        self.assertEqual("".join(r[1] for r in self.rows()).count("🍞"), 700 + 80)

    def test_failed_part_rolls_back_all_parts_and_retirement(self):
        self.write(self.records())
        old = self.export()
        with self.connect() as conn:
            conn.execute("""CREATE TRIGGER fail_part BEFORE INSERT ON archive
                WHEN NEW.chunk_id LIKE '%p0002' BEGIN SELECT RAISE(ABORT,'injected part failure'); END""")
        self.write(self.records() + [item("message", role="assistant", content="x" * 500)])
        with patch.object(legacy, "MAX_CHUNK_BYTES", 100):
            result = self.export()
        self.assertIn("injected part failure", result["error"])
        self.assertEqual(result["chunk_ids"], [])
        self.assertEqual([(r[0], r[2]) for r in self.rows()], [(old["chunk_ids"][0], "active")])

    def test_client_identity_isolated(self):
        self.write(self.records())
        self.export()
        self.write(self.records("claude"))
        self.export("claude")
        self.assertEqual(sum(r[2] == "active" for r in self.rows()), 2)

    def test_hook_subprocess_detaches_and_archives(self):
        self.write(self.records())
        payload = {"session_id": self.sid, "cwd": str(self.root), "transcript_path": str(self.path),
                   "hook_event_name": "SessionEnd"}
        start = time.monotonic()
        proc = subprocess.run([sys.executable, str(RUNNER), "--client", "codex"],
                              input=json.dumps(payload), text=True, capture_output=True, timeout=3)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertLess(time.monotonic() - start, 3)
        for _ in range(100):
            log = self.data / "graph_build.log"
            if self.rows() and not list((self.data / "archive/pending").glob("*.json")):
                # Empty graph exits immediately; wait briefly for worker cleanup.
                time.sleep(.1)
                break
            time.sleep(.05)
        self.assertEqual(len(self.rows()), 1)
        self.assertFalse(list((self.data / "archive/pending").glob("*.json")))
        # The worker holds hook.log open until it exits, and Windows can't
        # delete (or rename) an open file; wait for it so tearDown can clean up.
        hook_log = self.data / "archive/hook.log"
        for _ in range(100):
            try:
                hook_log.rename(hook_log.with_suffix(".closed"))
                break
            except FileNotFoundError:
                break
            except PermissionError:
                time.sleep(.05)

    def test_failed_worker_leaves_retryable_job(self):
        self.write(self.records())
        self.db.unlink()
        payload = {"session_id": self.sid, "cwd": str(self.root), "transcript_path": str(self.path)}
        self.assertFalse(lifecycle.queue_export(payload, "codex", "PreCompact", foreground=True))
        jobs = list((self.data / "archive/pending").glob("*.json"))
        self.assertEqual(len(jobs), 1)
        with self.connect() as conn:
            conn.executescript(SCHEMA_SQL)
        self.assertTrue(lifecycle.process_job(jobs[0]))
        self.assertFalse(jobs[0].exists())

    def test_missing_transcript_never_guesses(self):
        with self.assertRaisesRegex(ValueError, "refusing to guess"):
            lifecycle.queue_export({"session_id": self.sid}, "codex", "PreCompact")
        with self.assertRaises(FileNotFoundError):
            legacy.find_session_file("missing-session", self.root)

    def test_timing_is_session_and_client_scoped(self):
        from contextlib import redirect_stdout
        import io
        payload = {"session_id": self.sid, "cwd": str(self.root)}
        with patch.object(lifecycle.time, "time", return_value=1000):
            lifecycle.timing("Stop", payload, "codex")
        out = io.StringIO()
        with patch.object(lifecycle.time, "time", return_value=1120), redirect_stdout(out):
            lifecycle.timing("UserPromptSubmit", payload, "codex")
            lifecycle.timing("UserPromptSubmit", payload, "claude")
            lifecycle.timing("UserPromptSubmit", {**payload, "session_id": "other"}, "codex")
        self.assertEqual(len(out.getvalue().splitlines()), 1)
        self.assertIn("+2m", json.loads(out.getvalue())["hookSpecificOutput"]["additionalContext"])

    def test_installer_merges_idempotently_and_backs_up(self):
        original = {"unrelated": True, "hooks": {"Stop": [{"hooks": [{"command": "other-hook"}]}]}}
        proposed = hook_config("codex", timing=True, data_dir=self.data)
        merged = merge_config(original, proposed)
        self.assertEqual(merge_config(merged, proposed), merged)
        path = self.root / "hooks.json"
        path.write_text(json.dumps(original))
        backup, changed = install(path, proposed)
        self.assertTrue(changed)
        self.assertEqual(json.loads(backup.read_text()), original)
        self.assertEqual(json.loads(path.read_text()), merged)
        self.assertEqual(install(path, proposed), (None, False))

    def test_concurrent_same_session_publications_converge(self):
        from concurrent.futures import ThreadPoolExecutor
        self.write(self.records())
        old_raw, _, _ = archive.snapshot_transcript(self.path, str(self.root), "codex")
        self.write(self.records() + [item("message", role="assistant", content="Newer response")])
        new_raw, _, _ = archive.snapshot_transcript(self.path, str(self.root), "codex")
        def run(path):
            return archive.export_session(self.sid, str(self.root), client="codex",
                                          transcript_path=path, snapshot=True)
        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(run, [new_raw, old_raw, new_raw, old_raw]))
        self.assertTrue(all(r["chunk_ids"] and not r["error"] for r in results))
        active = [r for r in self.rows() if r[2] == "active"]
        self.assertEqual(len(active), 1)
        self.assertIn("Newer response", active[0][1])

    def test_explicit_store_overrides_persona_routing(self):
        self.write(self.records())
        selected = self.root / "selected-store"
        selected.mkdir()
        with patch.dict(os.environ, EMBER4_DATA_DIR=str(selected)):
            raw, cwd, db_path = archive.snapshot_transcript(self.path, "/different/cwd", "codex")
        self.assertEqual(db_path, selected / "ember4.db")
        self.assertEqual(cwd, str(self.root))
        self.assertTrue(raw.is_relative_to(selected))

    def test_codex_hook_cli_rejects_missing_path(self):
        proc = subprocess.run([sys.executable, str(RUNNER), "--client", "codex", "--event", "PreCompact"],
                              input=json.dumps({"session_id": self.sid}), text=True,
                              capture_output=True, timeout=3)
        self.assertEqual(proc.returncode, 1)
        self.assertIn("refusing to guess", proc.stderr)
        self.assertEqual(self.rows(), [])


if __name__ == "__main__":
    unittest.main()
