"""Anthropic Messages API <-> OpenAI chat-completions translation for llm_gateway.

Lets Anthropic-shaped clients (Claude Code / Claude Agent SDK via ANTHROPIC_BASE_URL) use the
gateway's existing upstream (xAI grok via Hermes OAuth) or the grok-mock route. Pure functions
(no I/O) so they are unit-testable. Unsupported content (images, documents, thinking,
cache_control, server tools) is dropped rather than rejected.
"""

from __future__ import annotations

import json
import uuid
from typing import Any, Iterator

_STOP = {"stop": "end_turn", "tool_calls": "tool_use", "function_call": "tool_use",
         "length": "max_tokens", "content_filter": "refusal"}


def anthropic_error(status: int, msg: str, typ: str) -> dict:
    return {"type": "error", "error": {"type": typ, "message": msg}}


def error_type_for(status: int) -> str:
    return {400: "invalid_request_error", 401: "authentication_error", 403: "permission_error",
            404: "not_found_error", 413: "request_too_large", 429: "rate_limit_error",
            529: "overloaded_error"}.get(status, "api_error")


def _text_of(content: Any) -> str:
    """Flatten Anthropic content (str or blocks) to text; non-text blocks are dropped."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    parts = []
    for b in content:
        if isinstance(b, dict) and b.get("type") == "text":
            parts.append(b.get("text", ""))
        elif isinstance(b, str):
            parts.append(b)
    return "\n".join(p for p in parts if p)


def to_openai(req: dict) -> dict:
    """Anthropic /v1/messages request -> OpenAI /v1/chat/completions request (non-streaming)."""
    msgs: list[dict] = []
    system = _text_of(req.get("system"))
    if system:
        msgs.append({"role": "system", "content": system})
    for m in req.get("messages") or []:
        role, content = m.get("role"), m.get("content")
        if isinstance(content, str):
            msgs.append({"role": "system" if role == "system" else role, "content": content})
            continue
        blocks = content or []
        if role == "assistant":
            text = _text_of(blocks)
            calls = [{"id": b.get("id") or f"toolu_{uuid.uuid4().hex[:24]}", "type": "function",
                      "function": {"name": b.get("name", ""), "arguments": json.dumps(b.get("input") or {})}}
                     for b in blocks if isinstance(b, dict) and b.get("type") == "tool_use"]
            out: dict = {"role": "assistant", "content": text or None}
            if calls:
                out["tool_calls"] = calls
            if text or calls:
                msgs.append(out)
            continue
        # user turn: tool_result blocks become role=tool messages (must directly follow the call)
        pending_text: list[str] = []
        for b in blocks:
            if not isinstance(b, dict):
                continue
            if b.get("type") == "tool_result":
                rc = _text_of(b.get("content"))
                if b.get("is_error"):
                    rc = f"ERROR: {rc}"
                msgs.append({"role": "tool", "tool_call_id": b.get("tool_use_id", ""), "content": rc or ""})
            elif b.get("type") == "text" and b.get("text"):
                pending_text.append(b["text"])
        if pending_text:
            msgs.append({"role": "user", "content": "\n".join(pending_text)})
    out_req: dict = {"model": req.get("model"), "messages": msgs}
    if req.get("max_tokens") is not None:
        out_req["max_tokens"] = req["max_tokens"]
    for k in ("temperature", "top_p"):
        if req.get(k) is not None:
            out_req[k] = req[k]
    if req.get("stop_sequences"):
        out_req["stop"] = req["stop_sequences"][:4]
    tools = [t for t in req.get("tools") or [] if isinstance(t, dict) and "input_schema" in t]
    if tools:
        out_req["tools"] = [{"type": "function", "function": {
            "name": t["name"], "description": t.get("description") or t["name"],
            "parameters": t.get("input_schema") or {"type": "object", "properties": {}}}} for t in tools]
        tc = req.get("tool_choice") or {}
        kind = tc.get("type")
        if kind == "any":
            out_req["tool_choice"] = "required"
        elif kind == "tool" and tc.get("name"):
            out_req["tool_choice"] = {"type": "function", "function": {"name": tc["name"]}}
        elif kind == "none":
            out_req["tool_choice"] = "none"
        if tc.get("disable_parallel_tool_use"):
            out_req["parallel_tool_calls"] = False
    return out_req


def from_openai(resp: dict, model: str) -> dict:
    """OpenAI chat.completion -> Anthropic message."""
    choice = (resp.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    content: list[dict] = []
    if msg.get("content"):
        content.append({"type": "text", "text": msg["content"]})
    for tc in msg.get("tool_calls") or []:
        fn = tc.get("function") or {}
        try:
            args = json.loads(fn.get("arguments") or "{}")
        except ValueError:
            args = {"_raw": fn.get("arguments")}
        content.append({"type": "tool_use", "id": tc.get("id") or f"toolu_{uuid.uuid4().hex[:24]}",
                        "name": fn.get("name", ""), "input": args if isinstance(args, dict) else {"value": args}})
    stop = _STOP.get(choice.get("finish_reason") or "stop", "end_turn")
    if any(b["type"] == "tool_use" for b in content):
        stop = "tool_use"
    u = resp.get("usage") or {}
    return {"id": f"msg_{uuid.uuid4().hex[:24]}", "type": "message", "role": "assistant", "model": model,
            "content": content, "stop_reason": stop, "stop_sequence": None,
            "usage": {"input_tokens": int(u.get("prompt_tokens") or 0),
                      "output_tokens": int(u.get("completion_tokens") or 0)}}


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def sse_events(message: dict, chunk: int = 64) -> Iterator[str]:
    """Replay a complete Anthropic message as the Messages streaming event sequence."""
    start = {**message, "content": [], "stop_reason": None, "stop_sequence": None,
             "usage": {"input_tokens": message["usage"]["input_tokens"], "output_tokens": 0}}
    yield _sse("message_start", {"type": "message_start", "message": start})
    for i, b in enumerate(message["content"]):
        if b["type"] == "text":
            yield _sse("content_block_start", {"type": "content_block_start", "index": i,
                                               "content_block": {"type": "text", "text": ""}})
            t = b["text"]
            for j in range(0, len(t), chunk):
                yield _sse("content_block_delta", {"type": "content_block_delta", "index": i,
                                                   "delta": {"type": "text_delta", "text": t[j:j + chunk]}})
        else:
            yield _sse("content_block_start", {"type": "content_block_start", "index": i, "content_block":
                                               {"type": "tool_use", "id": b["id"], "name": b["name"], "input": {}}})
            js = json.dumps(b["input"], ensure_ascii=False)
            for j in range(0, len(js), chunk):
                yield _sse("content_block_delta", {"type": "content_block_delta", "index": i,
                                                   "delta": {"type": "input_json_delta", "partial_json": js[j:j + chunk]}})
        yield _sse("content_block_stop", {"type": "content_block_stop", "index": i})
    yield _sse("message_delta", {"type": "message_delta",
                                 "delta": {"stop_reason": message["stop_reason"], "stop_sequence": None},
                                 "usage": {"output_tokens": message["usage"]["output_tokens"]}})
    yield _sse("message_stop", {"type": "message_stop"})


def count_tokens(req: dict) -> int:
    """Rough estimate: ~4 chars per token over system, messages and tool definitions."""
    n = len(json.dumps(req.get("system") or "", ensure_ascii=False))
    n += len(json.dumps(req.get("messages") or [], ensure_ascii=False))
    n += len(json.dumps(req.get("tools") or [], ensure_ascii=False))
    return max(1, n // 4)


def _placeholder(schema: dict) -> Any:
    t = (schema or {}).get("type")
    if "enum" in (schema or {}):
        return schema["enum"][0]
    if t == "object":
        props = schema.get("properties") or {}
        req = schema.get("required") or list(props)
        return {k: _placeholder(props[k]) for k in req if k in props}
    return {"string": "mock", "integer": 1, "number": 1, "boolean": True, "array": []}.get(t, "mock")


def mock_openai_reply(oreq: dict) -> dict:
    """Generic deterministic mock for the Anthropic path: if tools are offered and the last turn is
    not a tool result, call the first (or forced) tool with schema-derived args; else answer text."""
    msgs = oreq.get("messages") or []
    tools = oreq.get("tools") or []
    last = msgs[-1] if msgs else {}
    tc = oreq.get("tool_choice")
    usage = {"prompt_tokens": max(1, len(json.dumps(msgs)) // 4)}
    if tools and last.get("role") != "tool" and tc != "none":
        fn = tools[0]["function"]
        if isinstance(tc, dict):
            fn = next((t["function"] for t in tools if t["function"]["name"] == tc["function"]["name"]), fn)
        args = _placeholder(fn.get("parameters") or {})
        call = {"id": f"toolu_mock{len(msgs):04d}", "type": "function",
                "function": {"name": fn["name"], "arguments": json.dumps(args)}}
        return {"choices": [{"message": {"role": "assistant", "content": None, "tool_calls": [call]},
                             "finish_reason": "tool_calls"}], "usage": {**usage, "completion_tokens": 10}}
    n_tool = sum(1 for m in msgs if m.get("role") == "tool")
    text = f"MOCK OK (messages={len(msgs)}, tool_results={n_tool})"
    return {"choices": [{"message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
            "usage": {**usage, "completion_tokens": len(text) // 4}}
