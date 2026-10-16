"""Minimal MCP stdio server for Ember.

Replaces FastMCP/Pydantic/anyio transport to eliminate memory leaks in
MCP SDK 1.26.0. All ember logic is imported from ember4.server unchanged.
Transport is a synchronous stdin/stdout JSON-RPC loop — no Pydantic
serialization, no anyio streams, no memory-leaking middleware.
"""

import asyncio
import inspect
import json
import logging
import sys
import traceback
from typing import get_type_hints

logger = logging.getLogger(__name__)

# JSON Schema type mapping for Python annotations
_TYPE_MAP = {str: "string", int: "integer", bool: "boolean", float: "number"}

INSTRUCTIONS = (
    "Ember is persistent memory. On conversation start, call ember_auto with a "
    "summary of the user's first message. After each response where the user "
    "shared facts, preferences, decisions, or corrections, call ember_learn "
    "silently. Never announce memory operations. Never mention internal mechanics "
    "(vectors, scores, cells). Use recalled memories naturally. When correcting "
    "old info, use ember_contradict. Use ember_read to get full content of a "
    "specific memory. Use archive_store to save session logs, debug records, and "
    "reference documents. Use archive_search for keyword search, archive_grep for "
    "regex/fuzzy pattern search, archive_read with start/end for navigating large "
    "chunks. Use deep_recall to search both memory and archive at once."
)

TOOL_NAMES = [
    "ember_store", "ember_recall", "ember_learn", "ember_wonder",
    "ember_contradict", "ember_read", "ember_list", "ember_delete",
    "ember_auto", "ember_update_tags", "ember_save_session",
    "ember_inspect", "ember_recompute_centroids", "ember_drift_check",
    "ember_graph_search", "ember_health", "ember_explain",
    "ember_import_markdown", "ember_dream_scan", "ember_dream_save",
    "ember_synthesis_scan", "ember_synthesis_save",
    "archive_store", "archive_search", "archive_grep", "archive_read",
    "archive_list", "archive_update", "archive_delete", "archive_retention",
    "deep_recall",
]


def _build_input_schema(fn):
    """Build JSON Schema from function signature and type hints."""
    sig = inspect.signature(fn)
    hints = get_type_hints(fn)
    props = {}
    required = []
    for name, param in sig.parameters.items():
        ptype = hints.get(name, str)
        # Unwrap Optional / Union types
        origin = getattr(ptype, "__origin__", None)
        if origin is not None:
            args = getattr(ptype, "__args__", ())
            ptype = next((a for a in args if a is not type(None)), str)
        json_type = _TYPE_MAP.get(ptype, "string")
        prop = {"type": json_type}
        if param.default is not inspect.Parameter.empty:
            prop["default"] = param.default
        else:
            required.append(name)
        props[name] = prop
    schema = {"type": "object", "properties": props}
    if required:
        schema["required"] = required
    return schema


def _unwrap(fn):
    """Fully unwrap functools.wraps / decorator chains."""
    while hasattr(fn, "__wrapped__"):
        fn = fn.__wrapped__
    return fn


def _discover_tools(srv_module):
    """Build tool function map and schema list from ember4.server module."""
    tool_fns = {}
    tool_schemas = []
    for name in TOOL_NAMES:
        fn = getattr(srv_module, name, None)
        if fn is None:
            logger.warning("Tool %s not found in server module, skipping", name)
            continue
        original = _unwrap(fn)
        tool_fns[name] = original  # Call originals directly, skip wrappers
        tool_schemas.append({
            "name": name,
            "description": inspect.getdoc(original) or "",
            "inputSchema": _build_input_schema(original),
        })
    return tool_fns, tool_schemas


def _coerce_args(fn, args: dict) -> dict:
    """Coerce string arguments to expected types (MCP sends everything as strings)."""
    hints = get_type_hints(_unwrap(fn))
    coerced = {}
    for k, v in args.items():
        expected = hints.get(k)
        if expected is int and isinstance(v, str):
            v = int(v)
        elif expected is bool and isinstance(v, str):
            v = v.lower() in ("true", "1", "yes")
        elif expected is float and isinstance(v, str):
            v = float(v)
        coerced[k] = v
    return coerced


def run():
    """Main entry point — synchronous stdin/stdout JSON-RPC loop."""
    logging.basicConfig(level=logging.WARNING, stream=sys.stderr)

    # Import ember server (creates FastMCP instance but never starts it)
    from ember import server as srv

    tool_fns, tool_schemas = _discover_tools(srv)
    logger.info("Registered %d tools", len(tool_fns))

    # Persistent event loop for async handler calls
    loop = asyncio.new_event_loop()

    def send(obj):
        sys.stdout.write(json.dumps(obj, default=str) + "\n")
        sys.stdout.flush()

    def respond(msg_id, result):
        send({"jsonrpc": "2.0", "id": msg_id, "result": result})

    def respond_error(msg_id, code, message):
        send({"jsonrpc": "2.0", "id": msg_id, "error": {"code": code, "message": message}})

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue

        msg_id = msg.get("id")
        method = msg.get("method")
        params = msg.get("params", {})

        # Notifications and responses have no id — no reply needed
        if msg_id is None:
            continue

        try:
            if method == "initialize":
                respond(msg_id, {
                    "protocolVersion": params.get("protocolVersion", "2024-11-05"),
                    "capabilities": {
                        "tools": {"listChanged": False},
                    },
                    "serverInfo": {"name": "Ember", "version": "3.0.0"},
                    "instructions": INSTRUCTIONS,
                })

            elif method == "ping":
                respond(msg_id, {})

            elif method == "tools/list":
                respond(msg_id, {"tools": tool_schemas})

            elif method == "resources/list":
                respond(msg_id, {"resources": []})

            elif method == "prompts/list":
                respond(msg_id, {"prompts": []})

            elif method == "tools/call":
                tool_name = params.get("name", "")
                arguments = params.get("arguments", {})
                fn = tool_fns.get(tool_name)

                if fn is None:
                    respond_error(msg_id, -32602, f"Unknown tool: {tool_name}")
                    continue

                try:
                    coerced = _coerce_args(fn, arguments)
                    result = loop.run_until_complete(fn(**coerced))
                    if isinstance(result, str):
                        content = [{"type": "text", "text": result}]
                    elif isinstance(result, dict):
                        content = [{"type": "text", "text": json.dumps(result, default=str)}]
                    else:
                        content = [{"type": "text", "text": str(result)}]
                    respond(msg_id, {"content": content})
                except Exception as exc:
                    logger.error("Tool %s failed: %s", tool_name, traceback.format_exc())
                    respond(msg_id, {
                        "content": [{"type": "text", "text": f"Error: {exc}"}],
                        "isError": True,
                    })

            else:
                respond_error(msg_id, -32601, f"Method not found: {method}")

        except Exception as exc:
            logger.error("Request handling failed: %s", traceback.format_exc())
            if msg_id is not None:
                respond_error(msg_id, -32603, str(exc))

    loop.close()
