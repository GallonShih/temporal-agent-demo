"""Harness-native subagents: workspace resolution (inherit / isolated), read-only, backward compat,
and an end-to-end OrchestratorAgent run on the local dev server with the scripted mock model."""

import asyncio
import json
from pathlib import Path

import pytest
from pydantic import ValidationError
from temporalio.client import Client
from temporal_agent_harness.harness.agent_client import AgentClient
from temporal_agent_harness.harness.agent_protocol import AgentConfig
from temporal_agent_harness.harness.subagent_toolset import subagent_toolset

from temporal_agent_demo.config import plugins
from temporal_agent_demo.orchestrator import REVIEWER, OrchestratorAgent, helper_role
from temporal_agent_demo.shared import (ORCHESTRATOR_NAME, TASK_QUEUE, AgentRole, OrchestratorSession, TaskResult,
                                        WorkspaceSession, isolated_workspace_id, new_execution_id,
                                        resolve_workspace_id)
from temporal_agent_demo.workflows import WorkspaceAgent
from temporal_agent_demo.workspace_fs import WorkspaceDisk, WorkspaceDiskConfig
from temporal_agent_demo.worker import build_worker

from .conftest import MOCK_KEY

P = new_execution_id()


def test_resolve_top_level_backward_compat():
    eid = new_execution_id()
    assert resolve_workspace_id(WorkspaceSession(execution_id=eid), eid, None) == eid
    with pytest.raises(ValueError):
        resolve_workspace_id(WorkspaceSession(execution_id=eid), new_execution_id(), None)


def test_resolve_inherit_and_isolated():
    inh = WorkspaceSession(role=AgentRole(name="reviewer", workspace_mode="inherit", read_only=True))
    assert resolve_workspace_id(inh, "reviewer-subagent-abc", P) == P
    iso = WorkspaceSession(role=AgentRole(name="researcher", workspace_mode="isolated"))
    a = resolve_workspace_id(iso, "researcher-subagent-1", P)
    assert a == isolated_workspace_id("researcher-subagent-1") != P and a.startswith("exec-")
    assert a == resolve_workspace_id(iso, "researcher-subagent-1", P)  # deterministic (replay-safe)
    assert a != resolve_workspace_id(iso, "researcher-subagent-2", P)
    for d in (inh, iso):  # a role session must be a child workflow
        with pytest.raises(ValueError):
            resolve_workspace_id(d, "x", None)
    with pytest.raises(ValueError):  # parent id must itself be a valid workspace id
        resolve_workspace_id(inh, "c", "../etc")


def test_init_data_validation_no_injection():
    with pytest.raises(ValidationError):
        WorkspaceSession()  # neither
    with pytest.raises(ValidationError):  # both: a subagent cannot name a workspace
        WorkspaceSession(execution_id=P, role=REVIEWER)
    with pytest.raises(ValidationError):
        WorkspaceSession(role={"name": "x", "workspace_mode": "../exec-1"})
    with pytest.raises(ValidationError):
        AgentRole(name="../evil")
    for extra in ({"workspace_id": P}, {"path": "/data/workspaces/x"}):  # nothing to inject a workspace with
        with pytest.raises(ValidationError):
            WorkspaceSession.model_validate({"role": {"name": "r"}, **extra})
    with pytest.raises(ValidationError):
        AgentRole.model_validate({"name": "r", "workspace_id": P})
    with pytest.raises(ValidationError):
        OrchestratorSession(execution_id=P, helper_workspace_mode="shared")
    assert OrchestratorSession(execution_id=P).helper_workspace_mode == "isolated"


def test_start_tools_take_no_arguments():
    import inspect
    tools = subagent_toolset(WorkspaceAgent, key="reviewer", task_queue=TASK_QUEUE,
                             init_data=lambda: WorkspaceSession(role=REVIEWER))
    start = next(t for t in tools if t.__name__ == "start_reviewer")
    assert list(inspect.signature(start).parameters) == []
    assert [t.__name__ for t in tools] == ["start_reviewer", "reviewer_run_task", "stop_reviewer"]
    assert helper_role("inherit").workspace_mode == "inherit" and not helper_role("inherit").read_only


async def test_read_only_disk(ws_root):
    eid = new_execution_id()
    rw = WorkspaceDisk(WorkspaceDiskConfig(workspace_id=eid))
    await rw.write(Path("a.md"), b"hi")
    ro = WorkspaceDisk(WorkspaceDiskConfig(workspace_id=eid, read_only=True))
    assert await ro.read(Path("a.md")) == b"hi" and await ro.list(Path(".")) == ["a.md"]
    for op, args in (("write", (Path("b.md"), b"x")), ("mkdir", (Path("d"),)), ("delete", (Path("a.md"),)),
                     ("rename", (Path("a.md"), Path("c.md")))):
        with pytest.raises(PermissionError):
            await getattr(ro, op)(*args)
    assert sorted(p.name for p in (Path(ws_root) / eid).iterdir()) == ["a.md"]


async def test_orchestrator_with_inherit_and_isolated_children(env, ws_root, mock):
    _, url = mock
    c = await Client.connect(env.client.service_client.config.target_host, plugins=plugins(url, MOCK_KEY))
    eid = new_execution_id()
    async with build_worker(c, identity="w1"):
        await AgentClient(c, eid).start_and_submit_message(
            "run_task", {"task": "[delegate] write a draft and get it reviewed"}, workflow_name=ORCHESTRATOR_NAME,
            task_queue=TASK_QUEUE, start_config=AgentConfig(),
            start_data=OrchestratorSession(execution_id=eid, model="grok-mock"))
        h = c.get_workflow_handle(eid)
        await asyncio.wait_for(h.result(), 120)
        res = await h.query("result", result_type=TaskResult)
    assert res.outcome == "completed", res.final
    names = [t.name for t in res.tool_steps]
    assert names[:3] == ["write_file", "start_reviewer", "start_researcher"], names
    out = json.loads(res.final.removeprefix("ORCH DONE "))
    rev = TaskResult.model_validate_json(out["reviewer"]) if out["reviewer"].startswith("{") else None
    resr = TaskResult.model_validate_json(out["researcher"]) if out["researcher"].startswith("{") else None
    assert rev and resr, out
    # reviewer: inherit + read-only -> same workspace, sees the parent's draft, cannot write
    assert rev.workspace_id == eid and rev.workspace_mode == "inherit" and rev.read_only
    rc = json.loads(rev.final.removeprefix("CHILD "))
    assert "PARENT-DRAFT v1" in rc["c1"] and rc["c2"] == "" and "read-only" in rc["c3"]
    # researcher: isolated -> own workspace, cannot see the draft, its file stays out of the parent's
    assert resr.workspace_id == isolated_workspace_id(resr.execution_id) != eid and resr.workspace_mode == "isolated"
    sc = json.loads(resr.final.removeprefix("CHILD "))
    assert "PARENT-DRAFT" not in sc["c1"] and "wrote" in sc["c2"]
    parent, child = Path(ws_root) / eid, Path(ws_root) / resr.workspace_id
    assert (child / "output/child.md").read_text() == "CHILD-NOTE\n" and (child / "session.json").exists()
    assert not (parent / "output/child.md").exists() and not (parent / "output/child_code.md").exists()
    assert sorted(p.name for p in (parent / "output").iterdir()) == ["draft.md"]
    # the parent's own bookkeeping was not clobbered by the inherit-mode child
    assert json.loads((parent / "session.json").read_text())["workflow_id"] == eid
    # child workflows: linked to the parent, closed
    for r in (rev, resr):
        d = await c.get_workflow_handle(r.execution_id).describe()
        assert d.parent_id == eid and d.status.name in ("COMPLETED", "TERMINATED")
