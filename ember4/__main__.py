import argparse
import asyncio
import os
import sys


def _shutdown():
    """Close the DB, then exit even if worker threads linger.

    aiosqlite's connection thread is non-daemon and only stops when the
    connection closes, and onnxruntime keeps pools of its own; without this,
    a server that served any tool call never exits after its client goes away,
    and each new session adds another one. Every write is already committed,
    so the hard exit loses nothing.
    """
    try:
        from ember4 import server
        if server._db is not None:
            asyncio.run(asyncio.wait_for(server._db.close(), timeout=3))
    except Exception as e:
        print(f"[ember4] shutdown: DB close failed: {e}", file=sys.stderr)
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.flush()
        except (ValueError, OSError):  # the transport may have closed it
            pass
    os._exit(0)


def main():
    parser = argparse.ArgumentParser(description="Ember MCP Server")
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse", "streamable-http", "fastmcp"],
        default="fastmcp",
        help="MCP transport (default: fastmcp, stdio = bare JSON-RPC loop)",
    )
    parser.add_argument(
        "--port", type=int, default=6363, help="Port for SSE/HTTP transport"
    )
    parser.add_argument(
        "--host", default="127.0.0.1", help="Host for SSE/HTTP transport"
    )
    args = parser.parse_args()

    try:
        _run(args)
    finally:
        _shutdown()


def _run(args):
    if args.transport == "fastmcp":
        # FastMCP stdio transport — proper keepalive, async lifecycle, error recovery
        from ember4.server import mcp
        mcp.run(transport="stdio")
    elif args.transport in ("sse", "streamable-http"):
        from ember4.server import mcp
        mcp.settings.host = args.host
        mcp.settings.port = args.port
        mcp.run(transport=args.transport)
    else:
        # Default: minimal stdio handler (no FastMCP/Pydantic overhead)
        from ember4.mcp_stdio import run
        run()


main()
