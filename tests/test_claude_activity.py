"""Claude Agent SDK activity: confinement helpers, options, tool registry, workflow wiring (SDK mocked)."""

import asyncio
from pathlib import Path

import pytest
from pydantic import ValidationError
from temporalio import activity
from temporalio.worker import Worker

from temporal_agent_demo import activities as A
from temporal_agent_demo.claude_tools import REGISTRY, build_server, word_count
from temporal_agent_demo.claude_workflow import ClaudeTaskWorkflow
from temporal_agent_demo.shared import (CLAUDE_WORKFLOW, ClaudeAgentInput, ClaudeAgentResult, SystemPromptPreset,
                                        new_execution_id)


def test_input_validation():
    eid = new_execution_id()
    ClaudeAgentInput(execution_id=eid, prompt="x", skills={"my-skill": "---\nname: my-skill\n---"})
    for bad in ({"execution_id": "../etc"}, {"skills": {"../x": "y"}}, {"builtin_tools": ["Bash"]},
                {"builtin_tools": ["WebFetch"]}):
        with pytest.raises(ValidationError):
            ClaudeAgentInput(**{"execution_id": eid, "prompt": "x", **bad})


def test_workspace_and_inside(ws_root):
    eid = new_execution_id()
    root = A.workspace_dir(eid)
    assert root == (Path(ws_root) / eid).resolve() and root.is_dir()
    assert A.inside(root, "out/a.md") and A.inside(root, str(root / "x"))
    assert not A.inside(root, "../other") and not A.inside(root, "/etc/passwd")
    (root / "link").symlink_to("/etc")
    assert not A.inside(root, "link/passwd")
    with pytest.raises(ValueError):
        A.workspace_dir("exec-../../x")


def test_install_skills_and_list_files():
    root = A.workspace_dir(new_execution_id())
    A.install_skills(root, {"haiku": "---\nname: haiku\ndescription: d\n---\nbody"})
    assert (root / ".claude/skills/haiku/SKILL.md").read_text().endswith("body")
    (root / "out").mkdir()
    (root / "out/r.md").write_text("x")
    assert A.list_files(root) == ["out/r.md"]


async def test_guard_denies_outside_paths():
    root = A.workspace_dir(new_execution_id())
    denied: list[str] = []
    g = A.make_guard(root, denied)
    ok = await g({"tool_name": "Write", "tool_input": {"file_path": str(root / "a.md")}}, "t1", None)
    assert ok == {}
    no = await g({"tool_name": "Read", "tool_input": {"file_path": "/etc/passwd"}}, "t2", None)
    assert no["hookSpecificOutput"]["permissionDecision"] == "deny"
    no = await g({"tool_name": "Glob", "tool_input": {"pattern": "*", "path": "../"}}, "t3", None)
    assert no["hookSpecificOutput"]["permissionDecision"] == "deny"
    assert denied == ["Read:/etc/passwd", "Glob:../"]


def test_build_options():
    eid = new_execution_id()
    root = A.workspace_dir(eid)
    inp = ClaudeAgentInput(execution_id=eid, prompt="p", skills={"haiku": "s"}, tools=["word_count"],
                           system_prompt=SystemPromptPreset(append="be brief"), model="grok-mock")
    o = A.build_options(inp, root, "/tmp/h", [])
    assert o.cwd == str(root) and o.model == "grok-mock" and o.permission_mode == "dontAsk"
    assert o.tools == ["Read", "Write", "Edit", "Glob", "Grep", "Skill"] and "Bash" in o.disallowed_tools
    assert "mcp__demo__word_count" in o.allowed_tools and set(o.mcp_servers) == {"demo"}
    assert o.setting_sources == ["project"] and o.strict_mcp_config and o.skills == ["haiku"]
    assert o.system_prompt == {"type": "preset", "preset": "claude_code", "append": "be brief"}
    assert o.env["CLAUDE_CODE_MAX_RETRIES"] == "0" and o.env["HOME"] == "/tmp/h"
    assert not any("KEY" in k or "TOKEN" in k for k in o.env)  # gateway key only via inherited pod env


async def test_tool_registry():
    assert set(REGISTRY) == {"word_count", "current_utc_time", "run_pytest"}
    assert (await word_count.handler({"text": "a b c"}))["content"][0]["text"].startswith("words=3")
    with pytest.raises(ValueError):
        build_server(["nope"])
    assert build_server([]) == ({}, [])


async def test_workflow_calls_activity(env):
    seen = []

    @activity.defn(name="run_claude_agent")
    async def fake(inp: ClaudeAgentInput) -> ClaudeAgentResult:
        seen.append(inp)
        return ClaudeAgentResult(execution_id=inp.execution_id, is_error=False, subtype="success", result="ok")

    from temporal_agent_demo.config import plugins
    from temporalio.client import Client
    c = await Client.connect(env.client.service_client.config.target_host, plugins=plugins("http://unused/v1", "x"))
    eid = new_execution_id()
    async with Worker(c, task_queue="claude-test", workflows=[ClaudeTaskWorkflow], activities=[fake]):
        res = await c.execute_workflow(CLAUDE_WORKFLOW, ClaudeAgentInput(execution_id=eid, prompt="hi", tools=["word_count"]),
                                       id=eid, task_queue="claude-test", result_type=ClaudeAgentResult)
    assert res.result == "ok" and seen[0].tools == ["word_count"]
