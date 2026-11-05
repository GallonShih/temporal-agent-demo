"""Worker-side registry of custom tools for the Claude activity (in-process SDK MCP server).

Activity inputs only NAME tools; code never travels through workflow history. Tools that touch the
workspace are built per run, bound to that run's workspace root.
"""

from __future__ import annotations

import asyncio
import os
import resource
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from claude_agent_sdk import SdkMcpTool, create_sdk_mcp_server, tool

SERVER = "demo"
PYTEST_TIMEOUT_S = int(os.environ.get("RUN_PYTEST_TIMEOUT_S", "90"))
MAX_TOOL_OUTPUT = 6000
NOBODY = 65534


def _text(s: str) -> dict[str, Any]:
    return {"content": [{"type": "text", "text": s}]}


@tool("word_count", "Count the words, lines and characters in a text.", {"text": str})
async def word_count(args: dict[str, Any]) -> dict[str, Any]:
    t = args.get("text", "")
    return _text(f"words={len(t.split())} lines={len(t.splitlines())} chars={len(t)}")


@tool("current_utc_time", "Return the current UTC time in ISO 8601.", {})
async def current_utc_time(args: dict[str, Any]) -> dict[str, Any]:
    return _text(datetime.now(timezone.utc).isoformat(timespec="seconds"))


def _inside(root: Path, p: str) -> Path | None:
    cand = Path(p or ".")
    full = (cand if cand.is_absolute() else root / cand).resolve()
    return full if (full == root or root in full.parents) else None


def _limits() -> None:  # runs in the child before exec
    os.setsid()
    resource.setrlimit(resource.RLIMIT_CPU, (PYTEST_TIMEOUT_S, PYTEST_TIMEOUT_S))
    resource.setrlimit(resource.RLIMIT_AS, (1 << 30, 1 << 30))
    resource.setrlimit(resource.RLIMIT_NPROC, (64, 64))
    if os.getuid() == 0:  # drop root: the child can't read the worker's /proc/<pid>/environ
        os.setgid(NOBODY)
        os.setuid(NOBODY)


async def run_pytest_in(root: Path, path: str = ".", timeout_s: int | None = None) -> tuple[int, str]:
    """Run `python -m pytest -q <path>` with cwd=root. No shell, minimal env (no secrets),
    rlimits, unprivileged when running as root, wall-clock timeout, truncated output.
    Returns (exit code, output); exit code 124 = timeout, 2 = path rejected."""
    target = _inside(root, path)
    if target is None:
        return 2, f"rejected: {path!r} is outside the workspace"
    rel = os.path.relpath(target, root)
    env = {"PATH": "/usr/local/bin:/usr/bin:/bin", "PYTHONDONTWRITEBYTECODE": "1", "HOME": "/tmp",
           "PYTHONPATH": str(root), "LANG": "C.UTF-8"}
    proc = await asyncio.create_subprocess_exec(
        sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "--color=no", rel,
        cwd=str(root), env=env, stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT, preexec_fn=_limits)
    try:
        out, _ = await asyncio.wait_for(proc.communicate(), timeout_s or PYTEST_TIMEOUT_S)
        code = proc.returncode
    except asyncio.TimeoutError:
        try:
            os.killpg(proc.pid, 9)
        except ProcessLookupError:
            pass
        await proc.wait()
        return 124, f"timeout after {timeout_s or PYTEST_TIMEOUT_S}s"
    text = out.decode("utf-8", "replace")
    if len(text) > MAX_TOOL_OUTPUT:
        text = text[:1500] + "\n...[truncated]...\n" + text[-(MAX_TOOL_OUTPUT - 1500):]
    return code, text


def _make_run_pytest(root: Path) -> SdkMcpTool:
    @tool("run_pytest", "Run the pytest suite (python -m pytest -q) in the workspace. Optional `path` "
          "(file or directory relative to the workspace, default '.'). Returns exit code and output.",
          {"type": "object", "properties": {"path": {"type": "string"}}})
    async def run_pytest(args: dict[str, Any]) -> dict[str, Any]:
        code, out = await run_pytest_in(root, args.get("path") or ".")
        return _text(f"exit_code={code}\n{out}")
    return run_pytest


# name -> factory(workspace root) -> tool
REGISTRY: dict[str, Callable[[Path], SdkMcpTool]] = {
    "word_count": lambda root: word_count,
    "current_utc_time": lambda root: current_utc_time,
    "run_pytest": _make_run_pytest,
}


def build_server(names: list[str], root: Path | None = None):
    """-> (mcp_servers dict, allowed tool ids). Unknown names raise ValueError."""
    unknown = [n for n in names if n not in REGISTRY]
    if unknown:
        raise ValueError(f"unknown tools: {unknown}; available: {sorted(REGISTRY)}")
    if not names:
        return {}, []
    if root is None and "run_pytest" in names:
        raise ValueError("run_pytest needs a workspace root")
    server = create_sdk_mcp_server(SERVER, tools=[REGISTRY[n](root) for n in names])
    return {SERVER: server}, [f"mcp__{SERVER}__{n}" for n in names]
