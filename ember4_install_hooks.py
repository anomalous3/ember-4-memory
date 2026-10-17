"""Print or install portable Ember lifecycle hooks without replacing other hooks.

    .venv/bin/python3 ember4_install_hooks.py --client codex --timing
    .venv/bin/python3 ember4_install_hooks.py --client codex --timing --install

The default prints only the proposed hook entries. --install backs up and
merges into the chosen client's configuration. Codex still requires /hooks
review/trust; this installer never bypasses or edits hook trust.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile


def hook_config(client, timing=False, data_dir=None):
    script = Path(__file__).resolve().parent / "ember4" / "hooks" / "lifecycle.py"
    events = ["PreCompact", "SessionEnd"] + (["Stop", "UserPromptSubmit"] if timing else [])
    hooks = {}
    for event in events:
        argv = [sys.executable, str(script), "--client", client, "--event", event]
        if data_dir:
            argv += ["--data-dir", str(Path(data_dir).expanduser().resolve())]
        command = subprocess.list2cmdline(argv) if os.name == "nt" else shlex.join(argv)
        hooks[event] = [{"hooks": [{"type": "command", "command": command,
                                    "timeout": 30 if event == "PreCompact" else 3}]}]
    return {"hooks": hooks}


def merge_config(existing, proposed):
    merged = json.loads(json.dumps(existing))
    hooks = merged.setdefault("hooks", {})
    for event, groups in proposed["hooks"].items():
        current = hooks.setdefault(event, [])
        commands = {h.get("command") for g in current for h in g.get("hooks", [])}
        for group in groups:
            additions = [h for h in group["hooks"] if h["command"] not in commands]
            if additions:
                current.append({"hooks": additions})
    return merged


def install(path, proposed):
    existing = json.loads(path.read_text()) if path.exists() else {}
    merged = merge_config(existing, proposed)
    if merged == existing:
        return None, False
    path.parent.mkdir(parents=True, exist_ok=True)
    backup = None
    if path.exists():
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        backup = path.with_name(path.name + ".bak-" + stamp)
        shutil.copy2(path, backup)
    fd, temp = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as out:
            json.dump(merged, out, indent=2)
            out.write("\n")
            out.flush()
            os.fsync(out.fileno())
        os.replace(temp, path)
    finally:
        Path(temp).unlink(missing_ok=True)
    return backup, True


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--client", choices=("codex", "claude"), required=True)
    ap.add_argument("--timing", action="store_true", help="Also install Stop and UserPromptSubmit timing hooks")
    ap.add_argument("--data-dir", type=Path, help="Explicit target store (otherwise the hook environment/default applies)")
    ap.add_argument("--config", type=Path, help="Override the client config destination")
    ap.add_argument("--install", action="store_true", help="Back up and merge; otherwise print proposed entries")
    args = ap.parse_args()
    proposed = hook_config(args.client, args.timing, args.data_dir)
    if not args.install:
        print(json.dumps(proposed, indent=2))
        return 0
    path = args.config or (Path.home() / (".codex/hooks.json" if args.client == "codex" else ".claude/settings.json"))
    backup, changed = install(path.expanduser(), proposed)
    print(f"{'Installed' if changed else 'Already configured'}: {path}")
    if backup:
        print(f"Previous config: {backup}")
    if args.client == "codex":
        print("Open /hooks in Codex and review/trust these definitions; restart/resume if needed.")
    print("Existing hooks were retained. Disable any older Ember export/timing hooks to avoid duplicate work.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
