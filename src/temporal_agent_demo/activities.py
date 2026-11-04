"""run_claude_agent: one Claude Agent SDK (Claude Code) run inside a single Temporal activity.

The whole agent loop runs here (no workflow determinism constraints). Claude Code talks to the
host llm_gateway's Anthropic-compatible /v1/messages endpoint:
    ANTHROPIC_BASE_URL   e.g. http://172.18.0.1:8787   (pod env)
    ANTHROPIC_AUTH_TOKEN the local gateway key          (pod env from Secret llm-api-key; never in
                                                         activity input/options.env/history)
Confinement: cwd = /data/workspaces/<execution id> (same rules as WorkspaceDisk), only the listed
built-in tools (never Bash/Web*), project settings only (so only this workspace's skills),
strict MCP config, permission_mode=dontAsk, and a PreToolUse hook that denies any path outside
the workspace. HOME / CLAUDE_CONFIG_DIR point to a throwaway dir. Temporal owns retries
(CLAUDE_CODE_MAX_RETRIES=0); the activity heartbeats so cancellation and worker loss are detected.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from temporalio import activity

from .shared import EXECUTION_ID_RE, ClaudeAgentInput, ClaudeAgentResult, SystemPromptPreset
from .workspace_fs import workspaces_root

PATH_KEYS = ("file_path", "path", "notebook_path")


def workspace_dir(execution_id: str) -> Path:
    if not EXECUTION_ID_RE.fullmatch(execution_id):
        raise ValueError(f"invalid execution id {execution_id!r}")
    base = workspaces_root().resolve()
    root = (base / execution_id).resolve()
    if root.parent != base:
        raise ValueError("workspace root outside workspaces base")
    root.mkdir(parents=True, exist_ok=True)
    return root


def inside(root: Path, p: str) -> bool:
    """True if p (absolute or relative to root) resolves inside root (symlinks followed)."""
    cand = Path(p)
    full = (cand if cand.is_absolute() else root / cand).resolve()
    return full == root or root in full.parents


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".tmp-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def install_skills(root: Path, skills: dict[str, str]) -> None:
    for name, text in skills.items():
        atomic_write(root / ".claude" / "skills" / name / "SKILL.md", text)


def make_guard(root: Path, denied: list[str]):
    async def guard(input_data: dict[str, Any], tool_use_id: str | None, context: Any) -> dict[str, Any]:
        ti = input_data.get("tool_input") or {}
        for k in PATH_KEYS:
            v = ti.get(k)
            if isinstance(v, str) and v and not inside(root, v):
                denied.append(f"{input_data.get('tool_name')}:{v}")
                return {"hookSpecificOutput": {"hookEventName": "PreToolUse", "permissionDecision": "deny",
                                               "permissionDecisionReason": "path outside the workspace"}}
        return {}
    return guard


def build_options(inp: ClaudeAgentInput, root: Path, home: str, denied: list[str]):
    from claude_agent_sdk import ClaudeAgentOptions, HookMatcher

    from .claude_tools import build_server

    mcp, tool_ids = build_server(inp.tools)
    skills = list(inp.skills) if inp.enabled_skills is None else inp.enabled_skills
    builtins = list(inp.builtin_tools) + (["Skill"] if skills else [])
    sp: Any = inp.system_prompt
    if isinstance(sp, SystemPromptPreset):
        sp = {"type": "preset", "preset": sp.preset, "append": sp.append}
    return ClaudeAgentOptions(
        cwd=str(root),
        model=inp.model,
        system_prompt=sp,
        tools=builtins,
        allowed_tools=builtins + tool_ids,
        disallowed_tools=["Bash", "WebFetch", "WebSearch", "Agent", "Task"],
        mcp_servers=mcp,
        strict_mcp_config=True,
        setting_sources=["project"],
        skills=skills or None,
        permission_mode="dontAsk",
        max_turns=inp.max_turns,
        hooks={"PreToolUse": [HookMatcher(matcher=None, hooks=[make_guard(root, denied)])]},
        # not secrets (the gateway key comes from the pod env and is inherited, never set here)
        env={"HOME": home, "CLAUDE_CONFIG_DIR": home, "CLAUDE_CODE_MAX_RETRIES": "0",
             "ANTHROPIC_MODEL": inp.model, "ANTHROPIC_SMALL_FAST_MODEL": inp.model,
             "ANTHROPIC_DEFAULT_HAIKU_MODEL": inp.model, "DISABLE_TELEMETRY": "1",
             "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1", "DISABLE_AUTOUPDATER": "1"},
    )


def list_files(root: Path) -> list[str]:
    out = []
    for p in sorted(root.rglob("*")):
        rel = p.relative_to(root)
        if p.is_file() and rel.parts[0] != ".claude" and not p.name.startswith(".tmp-"):
            out.append(str(rel))
    return out


async def _heartbeat_loop(state: dict) -> None:
    while True:
        activity.heartbeat(state)
        await asyncio.sleep(10)


@activity.defn
async def run_claude_agent(inp: ClaudeAgentInput) -> ClaudeAgentResult:
    from claude_agent_sdk import AssistantMessage, ClaudeSDKClient, ResultMessage, ToolUseBlock

    root = workspace_dir(inp.execution_id)
    install_skills(root, inp.skills)
    home = tempfile.mkdtemp(prefix="claude-home-")
    denied: list[str] = []
    state = {"messages": 0}
    tool_uses: list[str] = []
    final: ResultMessage | None = None
    hb = asyncio.create_task(_heartbeat_loop(state))
    try:
        options = build_options(inp, root, home, denied)
        async with ClaudeSDKClient(options=options) as client:  # disconnects (kills CLI) on cancel/error
            await client.query(inp.prompt)
            async for msg in client.receive_response():
                state["messages"] += 1
                activity.heartbeat(state)
                if isinstance(msg, AssistantMessage):
                    tool_uses += [b.name for b in msg.content if isinstance(b, ToolUseBlock)]
                elif isinstance(msg, ResultMessage):
                    final = msg
    finally:
        hb.cancel()
        shutil.rmtree(home, ignore_errors=True)
    if final is None:
        return ClaudeAgentResult(execution_id=inp.execution_id, is_error=True, subtype="no_result", result="",
                                 tool_uses=tool_uses, denied=denied, files=list_files(root))
    return ClaudeAgentResult(
        execution_id=inp.execution_id, is_error=bool(final.is_error), subtype=final.subtype,
        result=(final.result or "")[:4000], num_turns=final.num_turns, session_id=final.session_id,
        usage={k: v for k, v in (final.usage or {}).items() if isinstance(v, (int, float))},
        tool_uses=tool_uses, denied=denied, files=list_files(root))
