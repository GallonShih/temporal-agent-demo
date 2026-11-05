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

from .shared import EXECUTION_ID_RE, ClaudeAgentInput, ClaudeAgentResult, SystemPromptPreset, PytestRun
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


def sessions_root() -> Path:
    return Path(os.environ.get("SESSIONS_ROOT", "/data/sessions"))


def session_dir(execution_id: str) -> Path:
    """Persistent Claude Code HOME/CLAUDE_CONFIG_DIR for one execution, outside the workspace."""
    if not EXECUTION_ID_RE.fullmatch(execution_id):
        raise ValueError(f"invalid execution id {execution_id!r}")
    base = sessions_root().resolve()
    d = (base / execution_id).resolve()
    if d.parent != base:
        raise ValueError("session dir outside sessions base")
    ws = workspaces_root().resolve()
    if d == ws or ws in d.parents or base == ws or ws in base.parents:
        raise ValueError("session dir must not be inside the workspaces tree")
    d.mkdir(parents=True, exist_ok=True)
    os.chmod(d, 0o700)
    return d


def worker_id() -> str:
    return os.environ.get("POD_NAME") or os.uname().nodename


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
            os.fchmod(f.fileno(), 0o644)  # readable by the unprivileged pytest child
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

    mcp, tool_ids = build_server(inp.tools, root)
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
        resume=inp.resume_session_id,
        fork_session=bool(inp.resume_session_id and inp.fork_session),
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
    if inp.resume_session_id and not inp.persist_session:
        raise ValueError("resume_session_id requires persist_session=True")
    persistent = inp.persist_session
    home = str(session_dir(inp.execution_id)) if persistent else tempfile.mkdtemp(prefix="claude-home-")
    info = activity.info()
    meta = dict(step=inp.step, resumed_from=inp.resume_session_id, worker=worker_id(), attempt=info.attempt)
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
        if not persistent:
            shutil.rmtree(home, ignore_errors=True)
    if final is None:
        return ClaudeAgentResult(execution_id=inp.execution_id, is_error=True, subtype="no_result", result="",
                                 tool_uses=tool_uses, denied=denied, files=list_files(root), **meta)
    return ClaudeAgentResult(
        execution_id=inp.execution_id, is_error=bool(final.is_error), subtype=final.subtype,
        result=(final.result or "")[:4000], num_turns=final.num_turns, session_id=final.session_id,
        usage={k: v for k, v in (final.usage or {}).items() if isinstance(v, (int, float))},
        tool_uses=tool_uses, denied=denied, files=list_files(root), **meta)


@activity.defn
async def write_workspace_files(execution_id: str, files: dict[str, str]) -> list[str]:
    """Write files (relative path -> text) into the workspace atomically (e.g. hidden acceptance tests)."""
    root = workspace_dir(execution_id)
    written = []
    for rel, text in files.items():
        if Path(rel).is_absolute() or not inside(root, rel):
            raise ValueError(f"path outside the workspace: {rel!r}")
        atomic_write((root / rel).resolve(), text)
        written.append(rel)
    return written


@activity.defn
async def run_workspace_tests(execution_id: str) -> PytestRun:
    """Ground-truth test run (no LLM): python -m pytest in the workspace, same sandboxing as run_pytest."""
    from .claude_tools import run_pytest_in

    root = workspace_dir(execution_id)
    code, out = await run_pytest_in(root, ".")
    return PytestRun(exit_code=code, passed=code == 0, output=out, worker=worker_id())
