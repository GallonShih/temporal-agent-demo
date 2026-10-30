"""Mock OpenAI-compatible LLM server (stands in for xAI / SuperGrok / OpenAI).

Endpoints
  GET  /v1/models                -> {"data":[{"id": MOCK_MODEL}]}
  POST /v1/chat/completions      -> OpenAI chat.completion (function calling via `tools`)
  GET  /v1/stats                 -> counters (requests, faults, per-status)
  GET  /healthz

Behaviour is deterministic in the request messages: it counts prior assistant turns and returns
the next step of a scripted plan (tool_calls ... then a final assistant message). Same script as the
in-process MockLLM, but expressed purely over the OpenAI wire format.

Env
  MOCK_API_KEYS        comma-separated accepted bearer keys (required header; empty list = any non-empty key)
  MOCK_MODEL           model id (default grok-mock)
  MOCK_LATENCY_S       sleep before answering (default 0)
  MOCK_FAULT_EVERY     inject a fault for requests whose content-hash % N == 0 (0 = off)
  MOCK_FAULT_CODES     comma list cycled by hash, default "429,500"
  MOCK_FAULT_ATTEMPTS  how many times the SAME request is failed before succeeding (default 1)
  MOCK_FAULT_RATE      additional random fault probability 0..1 (default 0; non-deterministic)
Header (per request, for tests/curl)
  X-Mock-Fault: 429|500|503|400  -> force that status for this request
Streaming (`stream: true`) is NOT supported -> 400 invalid_request_error.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import random
import re
import time
import uuid
from collections import Counter
from dataclasses import dataclass, field

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

log = logging.getLogger("mock_llm")


@dataclass
class Settings:
    api_keys: set[str] = field(default_factory=set)
    model: str = "grok-mock"
    latency_s: float = 0.0
    fault_every: int = 0
    fault_codes: list[int] = field(default_factory=lambda: [429, 500])
    fault_attempts: int = 1
    fault_rate: float = 0.0

    @classmethod
    def from_env(cls) -> "Settings":
        keys = {k.strip() for k in os.environ.get("MOCK_API_KEYS", "").split(",") if k.strip()}
        return cls(
            api_keys=keys,
            model=os.environ.get("MOCK_MODEL", "grok-mock"),
            latency_s=float(os.environ.get("MOCK_LATENCY_S", "0")),
            fault_every=int(os.environ.get("MOCK_FAULT_EVERY", "0")),
            fault_codes=[int(c) for c in os.environ.get("MOCK_FAULT_CODES", "429,500").split(",") if c.strip()],
            fault_attempts=int(os.environ.get("MOCK_FAULT_ATTEMPTS", "1")),
            fault_rate=float(os.environ.get("MOCK_FAULT_RATE", "0")),
        )


def _err(status: int, msg: str, typ: str, headers: dict | None = None) -> JSONResponse:
    return JSONResponse({"error": {"message": msg, "type": typ, "code": status}}, status_code=status, headers=headers)


def _text(content) -> str:
    """OpenAI content may be a string or a list of {type:text,text} parts (Agents SDK)."""
    if isinstance(content, list):
        return "".join(p.get("text", "") for p in content if isinstance(p, dict))
    return content or ""


def _tool_call(cid: str, name: str, args: dict) -> dict:
    return {"id": cid, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


def _native_plan(task: str, sid: str, digest: str, tool_msgs: list[dict]) -> list[list[dict] | str]:
    """Plan for the native harness agent (WorkspaceAgent): virtual /workspace paths, Code Mode."""
    errors = sum(1 for m in tool_msgs if "failed:" in str(m.get("content", ""))[:80]
                 or "Error" in str(m.get("content", ""))[:60])
    script = (
        "from pathlib import Path\n"
        "task = Path('/workspace/input/task.md').read_text()\n"
        f"Path('/workspace/output/plan.md').write_text('# plan for {sid}\\n1. read task\\n2. analyse\\n3. write result\\n'"
        f" + '(task digest {digest}, ' + str(len(task)) + ' chars)\\n')\n"
        "sorted(p.name for p in Path('/workspace/output').iterdir())\n"
    )
    plan: list[list[dict] | str] = [
        [_tool_call("n1", "run_code", {"script": script}),
         _tool_call("n2", "list_files", {"path": "/workspace"})],
    ]
    if "[probe-traversal]" in task:
        m = re.search(r"\[probe-target=(exec-[0-9a-f]{32})\]", task)
        other = m.group(1) if m else "exec-" + "0" * 32
        plan.append([
            _tool_call("p1", "read_file", {"path": f"/workspace/../{other}/input/task.md"}),
            _tool_call("p2", "read_file", {"path": "/etc/passwd"}),
            _tool_call("p3", "write_file", {"path": f"/workspace/../{other}/output/pwned.md", "content": "x"}),
            _tool_call("p4", "list_files", {"path": "/workspace/../.."}),
            _tool_call("p5", "run_code", {"script": f"open('/data/workspaces/{other}/output/pwned2.md', 'w').write('x')"}),
            _tool_call("p6", "run_code", {"script": f"from pathlib import Path\nPath('/workspace/../{other}/output/pwned3.md').write_text('x')"}),
        ])
    plan.append([
        _tool_call("n3", "write_file", {"path": "/workspace/output/result.md",
                                        "content": f"# result for {sid}\ntask digest: {digest}\n"
                                                   f"tool results so far: {len(tool_msgs)} (errors: {errors})\n"}),
        _tool_call("n4", "read_file", {"path": "/workspace/output/plan.md"}),
    ])
    plan.append(f"DONE {sid} digest={digest}")
    return plan


def scripted_reply(messages: list[dict], tools_available: set[str]) -> dict:
    """Next assistant message for this conversation (deterministic)."""
    task = next((_text(m.get("content")) for m in messages if m.get("role") == "user"), "")
    session = re.search(r"session_id=(\S+)", next((_text(m.get("content")) for m in messages if m.get("role") == "system"), ""))
    sid = session.group(1) if session else "unknown"
    turn = sum(1 for m in messages if m.get("role") == "assistant")
    tool_msgs = [m for m in messages if m.get("role") == "tool"]
    errors = sum(1 for m in tool_msgs if str(m.get("content", "")).startswith(("PermissionError", "ERROR")))
    digest = hashlib.sha256(task.encode()).hexdigest()[:12]

    if "run_code" in tools_available:
        plan = _native_plan(task, sid, digest, tool_msgs)
        # Position = first step whose calls are not all answered yet (robust to clients that
        # split one multi-call assistant message into several, as the Agents SDK may).
        answered = {m.get("tool_call_id") for m in tool_msgs}
        step = next((st for st in plan if isinstance(st, str) or not {c["id"] for c in st} <= answered), plan[-1])
        if isinstance(step, str):
            return {"role": "assistant", "content": step}
        return {"role": "assistant", "content": None, "tool_calls": step}

    plan: list[list[dict] | str] = [
        [_tool_call("c1", "write_file", {"path": "input/task.md", "content": task}),
         _tool_call("c2", "list_files", {"path": "."})],
        [_tool_call("c3", "write_file", {"path": "output/plan.md",
                                         "content": f"# plan for {sid}\n1. read task\n2. analyse\n3. write result\n(task digest {digest})\n"}),
         _tool_call("c4", "read_file", {"path": "input/task.md"})],
    ]
    if "[probe-traversal]" in task:
        m = re.search(r"\[probe-target=(exec-[0-9a-f]{32})\]", task)
        other = m.group(1) if m else "exec-" + "0" * 32
        plan.append([
            _tool_call("p1", "read_file", {"path": f"../{other}/input/task.md"}),
            _tool_call("p2", "read_file", {"path": "/etc/passwd"}),
            _tool_call("p3", "write_file", {"path": f"../{other}/output/pwned.md", "content": "x"}),
            _tool_call("p4", "list_files", {"path": "../.."}),
        ])
    plan.append([
        _tool_call("c5", "write_file", {"path": "output/result.md",
                                        "content": f"# result for {sid}\ntask digest: {digest}\n"
                                                   f"tool results so far: {len(tool_msgs)} (errors: {errors})\n"}),
        _tool_call("c6", "list_files", {"path": "output"}),
    ])
    plan.append(f"DONE {sid} digest={digest}")
    step = plan[min(turn, len(plan) - 1)]
    if isinstance(step, str) or not tools_available:
        return {"role": "assistant", "content": step if isinstance(step, str) else f"DONE {sid} (no tools offered)"}
    step = [c for c in step if c["function"]["name"] in tools_available]
    return {"role": "assistant", "content": None, "tool_calls": step}


def create_app(settings: Settings | None = None) -> FastAPI:
    app = FastAPI(title="mock-llm (OpenAI-compatible)")
    app.state.settings = settings or Settings.from_env()
    app.state.stats = Counter()
    app.state.fault_attempts = Counter()  # request-hash -> times faulted
    app.state.rng = random.Random(0)

    def auth(request: Request) -> JSONResponse | None:
        h = request.headers.get("authorization", "")
        if not h.lower().startswith("bearer ") or not h[7:].strip():
            return _err(401, "missing bearer API key", "invalid_api_key")
        keys = app.state.settings.api_keys
        if keys and h[7:].strip() not in keys:
            return _err(401, "invalid API key", "invalid_api_key")
        return None

    @app.get("/healthz")
    async def healthz():
        return {"ok": True}

    @app.get("/v1/models")
    async def models(request: Request):
        if (e := auth(request)) is not None:
            return e
        return {"object": "list", "data": [{"id": app.state.settings.model, "object": "model", "owned_by": "mock"}]}

    @app.get("/v1/stats")
    async def stats():
        return dict(app.state.stats)

    @app.post("/v1/chat/completions")
    async def chat(request: Request):
        s: Settings = app.state.settings
        t0 = time.monotonic()
        rid = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        status, fault = 200, None
        body: dict = {}
        try:
            if (e := auth(request)) is not None:
                status = 401
                return e
            try:
                body = await request.json()
            except Exception:
                status = 400
                return _err(400, "invalid JSON", "invalid_request_error")
            if body.get("model") != s.model:
                status = 404
                return _err(404, f"model {body.get('model')!r} not found", "model_not_found")
            if body.get("stream"):
                status = 400
                return _err(400, "stream=true is not supported by mock-llm", "invalid_request_error")
            messages = body.get("messages")
            if not isinstance(messages, list) or not messages:
                status = 400
                return _err(400, "messages required", "invalid_request_error")

            req_hash = hashlib.sha256(json.dumps(messages, sort_keys=True).encode()).hexdigest()[:16]
            app.state.stats["requests"] += 1
            forced = request.headers.get("x-mock-fault")
            if forced:
                fault = int(forced)
            elif s.fault_every and int(req_hash, 16) % s.fault_every == 0 and app.state.fault_attempts[req_hash] < s.fault_attempts:
                app.state.fault_attempts[req_hash] += 1
                fault = s.fault_codes[int(req_hash, 16) % len(s.fault_codes)]
            elif s.fault_rate and app.state.rng.random() < s.fault_rate:
                fault = s.fault_codes[app.state.rng.randrange(len(s.fault_codes))]
            if s.latency_s:
                await asyncio.sleep(s.latency_s)
            if fault:
                status = fault
                app.state.stats["faults_injected"] += 1
                if fault == 429:
                    return _err(429, "rate limited (injected)", "rate_limit_exceeded", {"retry-after": "1"})
                return _err(fault, f"injected fault {fault}", "server_error" if fault >= 500 else "invalid_request_error")

            tools = {t["function"]["name"] for t in body.get("tools") or [] if t.get("type") == "function"}
            msg = scripted_reply(messages, tools)
            finish = "tool_calls" if msg.get("tool_calls") else "stop"
            app.state.stats["completions"] += 1
            return {
                "id": rid, "object": "chat.completion", "created": int(time.time()), "model": s.model,
                "choices": [{"index": 0, "message": msg, "finish_reason": finish}],
                "usage": {"prompt_tokens": sum(len(str(m.get("content") or "")) for m in messages) // 4,
                          "completion_tokens": 16, "total_tokens": 0},
            }
        finally:
            app.state.stats[f"status_{status}"] += 1
            log.info(json.dumps({
                "event": "chat.completions", "id": rid, "status": status, "fault": fault,
                "req_hash": locals().get("req_hash"), "turn": sum(1 for m in body.get("messages", []) if isinstance(m, dict) and m.get("role") == "assistant"),
                "n_messages": len(body.get("messages", []) or []), "client": request.headers.get("x-client-id"),
                "ms": round((time.monotonic() - t0) * 1000),
            }))

    return app


logging.basicConfig(level=logging.INFO, format="%(message)s")
app = create_app()
