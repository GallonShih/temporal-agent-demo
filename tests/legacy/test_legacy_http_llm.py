"""Workflow end-to-end through the real HTTP client (OpenAICompatibleLLM) against a live mock-llm."""

import asyncio
import socket
import threading
import time
from pathlib import Path

import pytest
import uvicorn
from temporalio.api.enums.v1 import EventType
from temporalio.client import WorkflowFailureError
from temporalio.exceptions import ActivityError, ApplicationError

from mock_llm_server.app import Settings, create_app
from temporal_agent_demo.worker import build_worker

from .test_legacy_workflow import start

from ..conftest import MOCK_KEY as KEY


@pytest.fixture
def http_llm(mock_llm, monkeypatch):
    app, url = mock_llm
    st = app.state.settings
    saved = (st.fault_every, st.fault_attempts, st.fault_codes, st.latency_s)
    app.state.stats.clear()
    app.state.fault_attempts.clear()
    monkeypatch.setenv("LLM_BACKEND", "openai")
    monkeypatch.setenv("OPENAI_BASE_URL", url)
    monkeypatch.setenv("OPENAI_API_KEY", KEY)
    monkeypatch.setenv("OPENAI_MODEL", "grok-mock")
    yield app
    st.fault_every, st.fault_attempts, st.fault_codes, st.latency_s = saved


async def llm_attempts(handle) -> list[int]:
    sched, out = {}, []
    async for ev in handle.fetch_history_events():
        if ev.event_type == EventType.EVENT_TYPE_ACTIVITY_TASK_SCHEDULED:
            sched[ev.event_id] = ev.activity_task_scheduled_event_attributes.activity_type.name
        elif ev.event_type == EventType.EVENT_TYPE_ACTIVITY_TASK_STARTED:
            a = ev.activity_task_started_event_attributes
            if sched[a.scheduled_event_id] == "llm_call":
                out.append(a.attempt)
    return out


async def test_workflow_over_http(env, ws_root, http_llm):
    async with build_worker(env.client, identity="w-http", legacy=True):
        ctx, h = await start(env.client, "analyse via http")
        res = await h.result()
    assert res.final.startswith(f"DONE {ctx.session_id}")
    assert http_llm.state.stats["completions"] == 4 and http_llm.state.stats["faults_injected"] == 0
    root = Path(ws_root) / ctx.workspace_id
    assert ctx.session_id in (root / "output/plan.md").read_text()
    assert (root / "input/task.md").read_text() == "analyse via http"


async def test_retry_on_429_and_500(env, ws_root, http_llm):
    st = http_llm.state.settings
    st.fault_every, st.fault_attempts, st.fault_codes = 1, 1, [429, 500]  # every request fails once
    async with build_worker(env.client, identity="w-http", legacy=True):
        ctx, h = await start(env.client, "flaky provider")
        res = await asyncio.wait_for(h.result(), 120)
    attempts = await llm_attempts(h)
    stats = http_llm.state.stats
    assert res.session_status == "closed"
    assert attempts == [2, 2, 2, 2]  # each of the 4 model turns succeeded on its 2nd attempt
    assert stats["faults_injected"] == 4 and stats["completions"] == 4
    assert stats["status_429"] + stats["status_500"] == 4


async def test_bad_api_key_is_non_retryable(env, http_llm, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "wrong")
    async with build_worker(env.client, identity="w-http", legacy=True):
        ctx, h = await start(env.client, "bad key")
        with pytest.raises(WorkflowFailureError) as ei:
            await asyncio.wait_for(h.result(), 30)
    cause = ei.value.cause
    assert isinstance(cause, ActivityError) and isinstance(cause.cause, ApplicationError)
    assert cause.cause.type == "LLMClientError" and cause.cause.non_retryable
    assert await llm_attempts(h) == [1]
