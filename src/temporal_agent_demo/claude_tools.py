"""Worker-side registry of custom tools for the Claude activity (in-process SDK MCP server).

Activity inputs only NAME tools; code never travels through workflow history.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from claude_agent_sdk import SdkMcpTool, create_sdk_mcp_server, tool

SERVER = "demo"


@tool("word_count", "Count the words, lines and characters in a text.", {"text": str})
async def word_count(args: dict[str, Any]) -> dict[str, Any]:
    t = args.get("text", "")
    return {"content": [{"type": "text", "text": f"words={len(t.split())} lines={len(t.splitlines())} chars={len(t)}"}]}


@tool("current_utc_time", "Return the current UTC time in ISO 8601.", {})
async def current_utc_time(args: dict[str, Any]) -> dict[str, Any]:
    return {"content": [{"type": "text", "text": datetime.now(timezone.utc).isoformat(timespec="seconds")}]}


REGISTRY: dict[str, SdkMcpTool] = {t.name: t for t in (word_count, current_utc_time)}


def build_server(names: list[str]):
    """-> (mcp_servers dict, allowed tool ids). Unknown names raise ValueError."""
    unknown = [n for n in names if n not in REGISTRY]
    if unknown:
        raise ValueError(f"unknown tools: {unknown}; available: {sorted(REGISTRY)}")
    if not names:
        return {}, []
    server = create_sdk_mcp_server(SERVER, tools=[REGISTRY[n] for n in names])
    return {SERVER: server}, [f"mcp__{SERVER}__{n}" for n in names]
