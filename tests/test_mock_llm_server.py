"""mock-llm server in isolation (ASGI transport, no network)."""

import httpx
import pytest

from mock_llm_server.app import Settings, create_app

KEY = "sk-test"


@pytest.fixture
def client():
    app = create_app(Settings(api_keys={KEY}))
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://mock/v1"), app


def body(messages, **kw):
    tools = [{"type": "function", "function": {"name": n, "parameters": {}}} for n in ("write_file", "read_file", "list_files", "run_code")]
    return {"model": "grok-mock", "messages": messages, "tools": tools, **kw}


H = {"Authorization": f"Bearer {KEY}"}
BASE = [{"role": "system", "content": "sys\nsession_id=exec-abc"}, {"role": "user", "content": "do it"}]


async def test_models_and_auth(client):
    c, _ = client
    assert (await c.get("/models")).status_code == 401
    assert (await c.get("/models", headers={"Authorization": "Bearer nope"})).status_code == 401
    r = await c.get("/models", headers=H)
    assert r.json()["data"][0]["id"] == "grok-mock"
    assert (await c.post("/chat/completions", json=body(BASE))).status_code == 401


async def test_rejects_stream_and_unknown_model(client):
    c, _ = client
    assert (await c.post("/chat/completions", json=body(BASE, stream=True), headers=H)).status_code == 400
    assert (await c.post("/chat/completions", json={**body(BASE), "model": "gpt-x"}, headers=H)).status_code == 404


async def test_scripted_conversation_is_deterministic(client):
    c, _ = client
    msgs = list(BASE)
    finals = []
    for _ in range(6):
        r1 = (await c.post("/chat/completions", json=body(msgs), headers=H)).json()
        r2 = (await c.post("/chat/completions", json=body(msgs), headers=H)).json()
        m1, m2 = r1["choices"][0]["message"], r2["choices"][0]["message"]
        assert m1 == m2
        if not m1.get("tool_calls"):
            assert r1["choices"][0]["finish_reason"] == "stop"
            finals.append(m1["content"])
            break
        assert r1["choices"][0]["finish_reason"] == "tool_calls"
        msgs.append(m1)
        msgs += [{"role": "tool", "tool_call_id": tc["id"], "content": "ok"} for tc in m1["tool_calls"]]
    assert finals and finals[0].startswith("DONE exec-abc")


async def test_fault_injection(client):
    c, app = client
    r = await c.post("/chat/completions", json=body(BASE), headers={**H, "X-Mock-Fault": "429"})
    assert r.status_code == 429 and r.headers["retry-after"] == "1"
    app.state.settings.fault_every, app.state.settings.fault_attempts = 1, 2
    codes = [(await c.post("/chat/completions", json=body(BASE), headers=H)).status_code for _ in range(3)]
    assert codes[:2] != [200, 200] and codes[2] == 200  # same request fails twice then succeeds
    assert app.state.stats["faults_injected"] == 3
