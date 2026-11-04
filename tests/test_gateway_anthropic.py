"""Anthropic /v1/messages compat endpoint on llm_gateway (mock mode, no network)."""

import importlib
import json
import os

import httpx

os.environ.setdefault("GATEWAY_API_KEYS", "sk-gw-test")
os.environ.setdefault("GATEWAY_MODE", "mock")
gw = importlib.import_module("llm_gateway.server")
from llm_gateway import anthropic_compat as ac  # noqa: E402

KEY = os.environ["GATEWAY_API_KEYS"].split(",")[0]
TOOL = {"name": "word_count", "description": "count words",
        "input_schema": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}}


def client():
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=gw.create_app()), base_url="http://gw/v1")


def test_to_openai_translation():
    req = {"model": "grok-4.6", "max_tokens": 50, "temperature": 0.2, "stop_sequences": ["X"],
           "system": [{"type": "text", "text": "sys", "cache_control": {"type": "ephemeral"}}],
           "tools": [TOOL, {"type": "web_search_20250305", "name": "web_search"}],
           "tool_choice": {"type": "tool", "name": "word_count"}, "thinking": {"type": "enabled", "budget_tokens": 9},
           "messages": [
               {"role": "user", "content": [{"type": "text", "text": "hi"},
                                            {"type": "image", "source": {"type": "base64", "data": "x"}}]},
               {"role": "assistant", "content": [{"type": "text", "text": "ok"},
                                                 {"type": "tool_use", "id": "t1", "name": "word_count", "input": {"text": "a b"}}]},
               {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1",
                                             "content": [{"type": "text", "text": "2"}]},
                                            {"type": "text", "text": "thanks"}]}]}
    o = ac.to_openai(req)
    assert o["messages"][0] == {"role": "system", "content": "sys"}
    assert o["messages"][1] == {"role": "user", "content": "hi"}
    a = o["messages"][2]
    assert a["content"] == "ok" and a["tool_calls"][0]["id"] == "t1"
    assert json.loads(a["tool_calls"][0]["function"]["arguments"]) == {"text": "a b"}
    assert o["messages"][3] == {"role": "tool", "tool_call_id": "t1", "content": "2"}
    assert o["messages"][4] == {"role": "user", "content": "thanks"}
    assert [t["function"]["name"] for t in o["tools"]] == ["word_count"]
    assert o["tool_choice"] == {"type": "function", "function": {"name": "word_count"}}
    assert o["max_tokens"] == 50 and o["stop"] == ["X"] and "thinking" not in o


def test_from_openai_translation():
    resp = {"choices": [{"message": {"content": "x", "tool_calls": [
        {"id": "c1", "function": {"name": "f", "arguments": "{\"a\": 1}"}}]}, "finish_reason": "tool_calls"}],
            "usage": {"prompt_tokens": 7, "completion_tokens": 3}}
    m = ac.from_openai(resp, "grok-4.6")
    assert m["id"].startswith("msg_") and m["model"] == "grok-4.6" and m["stop_reason"] == "tool_use"
    assert m["content"] == [{"type": "text", "text": "x"}, {"type": "tool_use", "id": "c1", "name": "f", "input": {"a": 1}}]
    assert m["usage"] == {"input_tokens": 7, "output_tokens": 3}
    assert ac.from_openai({"choices": [{"message": {"content": "y"}, "finish_reason": "length"}]}, "m")["stop_reason"] == "max_tokens"


def test_sse_sequence():
    m = ac.from_openai({"choices": [{"message": {"content": "hello", "tool_calls": [
        {"id": "c1", "function": {"name": "f", "arguments": "{\"a\": 1}"}}]}, "finish_reason": "tool_calls"}]}, "m")
    evs = [e.split("\n")[0][7:] for e in ac.sse_events(m)]
    assert evs == ["message_start", "content_block_start", "content_block_delta", "content_block_stop",
                   "content_block_start", "content_block_delta", "content_block_stop", "message_delta", "message_stop"]


async def test_auth_and_mock_roundtrip():
    body = {"model": "grok-mock", "max_tokens": 64, "tools": [TOOL],
            "messages": [{"role": "user", "content": "count"}]}
    async with client() as c:
        r = await c.post("/messages", json=body)
        assert r.status_code == 401 and r.json()["error"]["type"] == "authentication_error"
        assert (await c.post("/messages", json=body, headers={"x-api-key": "bad"})).status_code == 401
        r = await c.post("/messages", json=body, headers={"x-api-key": KEY})
        assert r.status_code == 200
        tu = r.json()["content"][0]
        assert r.json()["stop_reason"] == "tool_use" and tu["name"] == "word_count" and tu["input"] == {"text": "mock"}
        body["messages"] += [{"role": "assistant", "content": [tu]},
                             {"role": "user", "content": [{"type": "tool_result", "tool_use_id": tu["id"], "content": "1"}]}]
        r = await c.post("/messages", json=body, headers={"Authorization": f"Bearer {KEY}"})
        assert r.json()["stop_reason"] == "end_turn" and "tool_results=1" in r.json()["content"][0]["text"]
        r = await c.post("/messages", json={**body, "stream": True}, headers={"x-api-key": KEY})
        assert r.headers["content-type"].startswith("text/event-stream") and "event: message_stop" in r.text
        r = await c.post("/messages/count_tokens", json=body, headers={"x-api-key": KEY})
        assert r.status_code == 200 and r.json()["input_tokens"] > 0
        assert (await c.post("/messages/count_tokens", json=body)).status_code == 401
