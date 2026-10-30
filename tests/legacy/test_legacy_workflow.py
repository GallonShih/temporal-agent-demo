import asyncio
import json
import os
import uuid
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from temporalio.api.enums.v1 import EventType

from temporal_agent_demo import TASK_QUEUE
from temporal_agent_demo.api import create_app
from temporal_agent_demo.models import AgentInput, AgentResult, ExecutionContext
from temporal_agent_demo.worker import build_worker


async def other_client(env):
    """A 2nd client/connection = a 2nd worker process (the SDK forbids two workers on one client+queue)."""
    from temporalio.client import Client
    from temporal_agent_demo.config import harness_plugin

    return await Client.connect(env.client.service_client.config.target_host, plugins=[harness_plugin()])


async def start(client, task, ctx=None):
    ctx = ctx or ExecutionContext.new()
    return ctx, await client.start_workflow(
        "AgentWorkflow", AgentInput(ctx=ctx, task=task), id=ctx.execution_id, task_queue=TASK_QUEUE, result_type=AgentResult
    )


async def activity_identities(handle) -> dict[str, set[str]]:
    """activity type -> identities of workers that STARTED it (from workflow history)."""
    sched, out = {}, {}
    async for ev in handle.fetch_history_events():
        if ev.event_type == EventType.EVENT_TYPE_ACTIVITY_TASK_SCHEDULED:
            sched[ev.event_id] = ev.activity_task_scheduled_event_attributes.activity_type.name
        elif ev.event_type == EventType.EVENT_TYPE_ACTIVITY_TASK_STARTED:
            a = ev.activity_task_started_event_attributes
            out.setdefault(sched[a.scheduled_event_id], set()).add(a.identity)
    return out


async def test_single_execution(env, ws_root):
    async with build_worker(env.client, identity="w1", legacy=True):
        ctx, h = await start(env.client, "analyse python repo")
        res = await h.result()
    assert res.session_status == "closed" and res.final.startswith(f"DONE {ctx.execution_id}")
    assert res.run_id != ctx.execution_id
    root = Path(ws_root) / ctx.workspace_id
    assert (root / "input/task.md").read_text() == "analyse python repo"
    assert (root / "output/plan.md").exists() and (root / "output/result.md").exists()
    assert json.loads((root / "session.json").read_text())["status"] == "closed"
    assert sorted(p.name for p in (root / "temp").iterdir()) == ["step-000.json", "step-001.json", "step-002.json"]


async def test_ten_executions_two_workers_isolated(env, ws_root):
    c2 = await other_client(env)
    async with build_worker(env.client, identity="worker-A", legacy=True), build_worker(c2, identity="worker-B", legacy=True):
        started = [await start(env.client, f"task {i:03d}") for i in range(10)]
        results = await asyncio.gather(*(h.result() for _, h in started))
        ids = [await activity_identities(h) for _, h in started]
    eids = {c.execution_id for c, _ in started}
    assert len(eids) == 10
    for (ctx, _), res in zip(started, results):
        root = Path(ws_root) / ctx.workspace_id
        assert ctx.execution_id in (root / "output/plan.md").read_text()
        assert res.session_status == "closed"
    for i, (ctx, _) in enumerate(started):
        assert (Path(ws_root) / ctx.workspace_id / "input/task.md").read_text() == f"task {i:03d}"
    used = set().union(*(s for d in ids for s in d.values()))
    assert used == {"worker-A", "worker-B"}, used  # both workers did work


async def test_traversal_probe_rejected_inside_workflow(env, ws_root):
    async with build_worker(env.client, identity="w1", legacy=True):
        victim, hv = await start(env.client, "victim task")
        await hv.result()
        ctx, h = await start(env.client, f"[probe-traversal][probe-target={victim.execution_id}] try escape")
        res = await h.result()
    probe = next(s for s in res.steps if any(c.id == "p1" for c in s.tool_calls))
    assert [r.ok for r in probe.tool_results] == [False, False, False, False]
    assert all(r.output.startswith("PermissionError") for r in probe.tool_results)
    assert not (Path(ws_root) / victim.workspace_id / "output/pwned.md").exists()
    assert (Path(ws_root) / victim.workspace_id / "input/task.md").read_text() == "victim task"


async def test_worker_crash_other_worker_finishes_same_workspace(env, ws_root, monkeypatch):
    monkeypatch.setenv("MOCK_LLM_DELAY_S", "1.5")
    wa = build_worker(env.client, identity="worker-A", legacy=True)
    ta = asyncio.create_task(wa.run())
    ctx, h = await start(env.client, "crash me")
    # wait until worker-A finished at least one LLM step, then kill it mid-run
    for _ in range(100):
        done = [e async for e in h.fetch_history_events() if e.event_type == EventType.EVENT_TYPE_ACTIVITY_TASK_COMPLETED]
        if len(done) >= 6:  # mkdirs + session.json + first llm_call done; worker-A is mid-run
            break
        await asyncio.sleep(0.2)
    ta.cancel()
    try:
        await ta
    except (asyncio.CancelledError, Exception):
        pass
    async with build_worker(await other_client(env), identity="worker-B", legacy=True):
        res = await asyncio.wait_for(h.result(), 120)
    ids = await activity_identities(h)
    assert "worker-A" in ids["llm_call"] and "worker-B" in ids["llm_call"]
    assert res.session_status == "closed"
    root = Path(ws_root) / ctx.workspace_id
    assert (root / "output/result.md").exists() and (root / "input/task.md").read_text() == "crash me"


async def test_api_post_executions(env):
    app = create_app(env.client)
    async with build_worker(env.client, identity="w-api", legacy=True):
        with TestClient(app) as tc:
            r = tc.post("/executions", json={"task": "via api", "engine": "legacy"})
            assert r.status_code == 201
            body = r.json()
            assert body["execution_id"] == body["workflow_id"] == body["session_id"] == body["workspace_id"]
            assert body["run_id"] != body["execution_id"]
            await env.client.get_workflow_handle(body["workflow_id"]).result()
            g = tc.get(f"/executions/{body['execution_id']}").json()
            assert g["status"] == "COMPLETED" and g["result"]["session_status"] == "closed"
            assert tc.get("/executions/../../etc").status_code in (400, 404)
