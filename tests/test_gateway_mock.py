"""llm_gateway in mock mode (no Hermes / no network)."""

import importlib
import os

import httpx

os.environ.setdefault("GATEWAY_API_KEYS", "sk-gw-test")
os.environ.setdefault("GATEWAY_MODE", "mock")
gw = importlib.import_module("llm_gateway.server")

H = {"Authorization": "Bearer sk-gw-test"}


def client():
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=gw.create_app()), base_url="http://gw/v1")


async def test_gateway_auth_models_and_mock_chat():
    async with client() as c:
        assert (await c.get("/models")).status_code == 401
        assert [m["id"] for m in (await c.get("/models", headers=H)).json()["data"]] == ["grok-mock"]
        body = {"model": "anything", "messages": [{"role": "system", "content": "s\nsession_id=exec-1"},
                                                  {"role": "user", "content": "t"}],
                "tools": [{"type": "function", "function": {"name": n}} for n in ("write_file", "list_files")]}
        r = await c.post("/chat/completions", json=body, headers=H)
        assert r.status_code == 200 and r.json()["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == "write_file"
        assert (await c.post("/chat/completions", json={**body, "stream": True}, headers=H)).status_code == 400
        r = await c.post("/chat/completions", json=body, headers={**H, "X-Mock-Fault": "429"})
        assert r.status_code == 429 and r.headers.get("retry-after") == "1"
