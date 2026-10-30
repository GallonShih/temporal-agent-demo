"""Native harness agent (WorkspaceAgent) on a local Temporal dev server, model = mock-llm over HTTP
(OpenAIAgentsPlugin -> chat-completions -> mock_llm_server)."""

import asyncio
import json
from pathlib import Path

import httpx
import pytest
from temporalio.api.enums.v1 import EventType
from temporalio.client import Client, WorkflowFailureError
from temporal_agent_harness.harness.agent_client import AgentClient
from temporal_agent_harness.harness.agent_protocol import AgentConfig

from temporal_agent_demo.api import create_app
from temporal_agent_demo.config import plugins
from temporal_agent_demo.shared import AGENT_NAME, TASK_QUEUE, TaskResult, WorkspaceSession, new_execution_id
from temporal_agent_demo.starter import start_execution, wait_result
from temporal_agent_demo.worker import build_worker

from .conftest import MOCK_KEY

MODEL_ACT = "invoke_model_activity"


async def model_client(env, url) -> Client:
    """One client == one 'pod' (the SDK forbids two workers on one client+queue)."""
    return await Client.connect(env.client.service_client.config.target_host, plugins=plugins(url, MOCK_KEY))


async def start(client: Client, task: str, eid: str | None = None, data_id: str | None = None):
    eid = eid or new_execution_id()
    await AgentClient(client, eid).start_and_submit_message(
        "run_task", {"task": task}, workflow_name=AGENT_NAME, task_queue=TASK_QUEUE,
        start_config=AgentConfig(), start_data=WorkspaceSession(execution_id=data_id or eid, model="grok-mock"))
    return eid, client.get_workflow_handle(eid)


async def finish(h) -> TaskResult:
    await asyncio.wait_for(h.result(), 120)
    return await h.query("result", result_type=TaskResult)


async def activity_identities(handle) -> dict[str, list[tuple[str, int]]]:
    sched, out = {}, {}
    async for ev in handle.fetch_history_events():
        if ev.event_type == EventType.EVENT_TYPE_ACTIVITY_TASK_SCHEDULED:
            sched[ev.event_id] = ev.activity_task_scheduled_event_attributes.activity_type.name
        elif ev.event_type == EventType.EVENT_TYPE_ACTIVITY_TASK_STARTED:
            a = ev.activity_task_started_event_attributes
            out.setdefault(sched[a.scheduled_event_id], []).append((a.identity, a.attempt))
    return out


def tree(root: Path) -> list[str]:
    return sorted(str(p.relative_to(root)) for p in root.rglob("*") if p.is_file())


async def test_single_session(env, ws_root, mock):
    app, url = mock
    c = await model_client(env, url)
    async with build_worker(c, identity="w1"):
        eid, h = await start(c, "analyse python repo")
        res = await finish(h)
    assert res.outcome == "completed" and res.final.startswith(f"DONE {eid}")
    assert res.execution_id == res.session_id == res.workspace_id == eid != res.run_id
    root = Path(ws_root) / eid
    assert tree(root) == ["input/task.md", "output/plan.md", "output/result.md", "session.json", "temp/steps.json"]
    assert (root / "input/task.md").read_text() == "analyse python repo"
    assert f"plan for {eid}" in (root / "output/plan.md").read_text()  # written by a Code Mode script
    s = json.loads((root / "session.json").read_text())
    assert s["status"] == "closed" and s["outcome"] == "completed" and s["session_id"] == eid
    assert [t.name for t in res.tool_steps] == ["run_code", "list_files", "write_file", "read_file"]
    assert all(t.ok for t in res.tool_steps), res.tool_steps
    assert res.model_calls == 3 and app.state.stats["completions"] == 3
    desc = await h.describe()
    assert desc.status.name == "COMPLETED"


async def test_ten_sessions_two_workers_isolated(env, ws_root, mock):
    _, url = mock
    ca, cb = await model_client(env, url), await model_client(env, url)
    async with build_worker(ca, identity="worker-A"), build_worker(cb, identity="worker-B"):
        started = [await start(ca, f"task {i:03d}") for i in range(10)]
        results = await asyncio.gather(*(finish(h) for _, h in started))
        idents = [await activity_identities(h) for _, h in started]
    assert len({e for e, _ in started}) == 10
    for i, ((eid, _), res) in enumerate(zip(started, results)):
        root = Path(ws_root) / eid
        assert res.outcome == "completed" and res.session_status == "closed"
        assert (root / "input/task.md").read_text() == f"task {i:03d}"
        assert eid in (root / "output/plan.md").read_text() and eid in (root / "output/result.md").read_text()
        assert json.loads((root / "session.json").read_text())["status"] == "closed"
    used = {ident for d in idents for runs in d.values() for ident, _ in runs}
    assert used == {"worker-A", "worker-B"}, used


async def test_traversal_rejected(env, ws_root, mock):
    _, url = mock
    c = await model_client(env, url)
    async with build_worker(c, identity="w1"):
        victim, hv = await start(c, "victim task")
        await finish(hv)
        before = tree(Path(ws_root) / victim)
        eid, h = await start(c, f"[probe-traversal][probe-target={victim}] try escape")
        res = await finish(h)
    probes = {t.call_id: t for t in res.tool_steps if t.call_id.startswith("p")}
    assert sorted(probes) == ["p1", "p2", "p3", "p4", "p5", "p6"]
    assert not any(t.ok for t in probes.values()), probes
    for cid in ("p1", "p3", "p4"):  # reached WorkspaceDisk inside the activity -> its PermissionError
        assert "PermissionError" in probes[cid].output and "escapes workspace" in probes[cid].output, probes[cid]
    assert "PermissionError" in probes["p2"].output  # absolute path outside /workspace
    for cid in ("p5", "p6"):  # Code Mode: no mount at that path -> sandbox error
        assert "Error" in probes[cid].output, probes[cid]
    assert tree(Path(ws_root) / victim) == before
    assert not list(Path(ws_root).parent.rglob("pwned*"))
    assert res.outcome == "completed" and "errors: 6" in (Path(ws_root) / eid / "output/result.md").read_text()


async def test_crash_other_worker_finishes_same_workspace(env, ws_root, mock):
    app, url = mock
    app.state.settings.latency_s = 1.5
    ca, cb = await model_client(env, url), await model_client(env, url)
    wa = build_worker(ca, identity="worker-A")
    ta = asyncio.create_task(wa.run())
    eid, h = await start(ca, "crash me")
    for _ in range(200):  # first model call done -> worker-A is mid-session (2nd call in flight)
        ids = await activity_identities(h)
        if len(ids.get(MODEL_ACT, [])) >= 2:
            break
        await asyncio.sleep(0.1)
    ta.cancel()
    try:
        await ta
    except (asyncio.CancelledError, Exception):
        pass
    async with build_worker(cb, identity="worker-B"):
        res = await finish(h)
    ids = await activity_identities(h)
    who = {i for i, _ in ids[MODEL_ACT]}
    assert who == {"worker-A", "worker-B"}, ids
    assert res.outcome == "completed" and res.session_status == "closed"
    root = Path(ws_root) / eid
    assert tree(root) == ["input/task.md", "output/plan.md", "output/result.md", "session.json", "temp/steps.json"]
    assert json.loads((root / "session.json").read_text())["status"] == "closed"


async def test_model_429_500_are_retried(env, ws_root, mock):
    app, url = mock
    st = app.state.settings
    st.fault_every, st.fault_attempts, st.fault_codes = 1, 2, [429, 500]  # every request fails twice
    c = await model_client(env, url)
    async with build_worker(c, identity="w1"):
        eid, h = await start(c, "flaky provider")
        res = await finish(h)
    assert res.outcome == "completed"
    attempts = [a for _, a in (await activity_identities(h))[MODEL_ACT]]
    assert attempts.count(3) == 3 and max(attempts) == 3, attempts  # 3 model calls x (fail, fail, ok)
    assert app.state.stats["faults_injected"] == 6
    assert app.state.stats["status_429"] + app.state.stats["status_500"] == 6


async def test_bad_key_fails_task_but_session_still_closes(env, ws_root, mock):
    _, url = mock
    c = await Client.connect(env.client.service_client.config.target_host, plugins=plugins(url, "sk-wrong"))
    async with build_worker(c, identity="w1"):
        eid, h = await start(c, "no auth")
        res = await finish(h)
    assert res.outcome == "failed" and "401" in res.final
    attempts = [a for _, a in (await activity_identities(h))[MODEL_ACT]]
    assert attempts == [1]  # non-retryable
    assert json.loads((Path(ws_root) / eid / "session.json").read_text())["status"] == "closed"


async def test_id_mismatch_rejected(env, mock):
    _, url = mock
    c = await model_client(env, url)
    async with build_worker(c, identity="w1"):
        eid, h = await start(c, "x", data_id=new_execution_id())
        with pytest.raises(WorkflowFailureError):
            await asyncio.wait_for(h.result(), 60)


async def test_api_post_executions_native(env, ws_root, mock):
    # httpx.ASGITransport keeps the app on THIS event loop: the update-with-start inside POST waits
    # for the worker (same loop) to accept, so a sync TestClient would deadlock.
    _, url = mock
    c = await model_client(env, url)
    app = create_app(c)
    app.state.client = c  # ASGITransport does not run lifespan
    async with build_worker(c, identity="w-api"):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://api") as tc:
            r = await tc.post("/executions", json={"task": "via api", "model": "grok-mock"})
            assert r.status_code == 201, r.text
            b = r.json()
            assert b["execution_id"] == b["workflow_id"] == b["session_id"] == b["workspace_id"] != b["run_id"]
            await asyncio.wait_for(c.get_workflow_handle(b["workflow_id"]).result(), 60)
            g = (await tc.get(f"/executions/{b['execution_id']}")).json()
            assert g["status"] == "COMPLETED" and g["session_status"] == "closed"
            assert g["result"]["outcome"] == "completed" and g["result"]["final"].startswith("DONE")
            assert (await tc.get("/executions/exec-nothex")).status_code == 400
            assert (await tc.post("/executions/../close")).status_code in (400, 404)


async def test_starter_start_execution(env, ws_root, mock):
    _, url = mock
    c = await model_client(env, url)
    async with build_worker(c, identity="w-starter"):
        started = await start_execution(c, "via starter", model="grok-mock")
        assert started["execution_id"] == started["workflow_id"] == started["workspace_id"] != started["run_id"]
        res = await asyncio.wait_for(wait_result(c, started["execution_id"]), 60)
    assert res.outcome == "completed" and res.session_status == "closed"
    assert (Path(ws_root) / started["execution_id"] / "output/result.md").exists()
