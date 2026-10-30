"""LLM interface + in-process deterministic mock + OpenAI-compatible HTTP client, and the activity.

Select with env LLM_BACKEND=mock (default) | openai.
  openai: OPENAI_BASE_URL (default https://api.openai.com/v1), OPENAI_API_KEY, OPENAI_MODEL.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import socket
from datetime import timedelta
from typing import Protocol

import httpx
from temporalio import activity
from temporalio.exceptions import ApplicationError

from ..models import LLMRequest, LLMResponse, Message, ToolCall

TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write (replace) a UTF-8 file in the workspace.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a UTF-8 file from the workspace.",
            "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List a directory in the workspace.",
            "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
        },
    },
]


class LLMClient(Protocol):
    name: str

    async def complete(self, req: LLMRequest) -> LLMResponse: ...


def worker_name() -> str:
    return os.environ.get("POD_NAME") or socket.gethostname()


class MockLLM:
    """Deterministic scripted 'model'. Output depends only on (task, number of prior assistant turns).

    Special markers in the task:
      [probe-traversal]          -> also tries ../<other>/input/task.md, /etc/passwd, ../../..
      [probe-target=exec-<hex>]  -> the other workspace to try to read
    """

    name = "mock"

    async def complete(self, req: LLMRequest) -> LLMResponse:
        task = next((m.content for m in req.messages if m.role == "user"), "")
        turn = sum(1 for m in req.messages if m.role == "assistant")
        digest = hashlib.sha256(task.encode()).hexdigest()[:12]
        tool_results = [m.tool_result for m in req.messages if m.role == "tool" and m.tool_result]
        plan: list[list[ToolCall] | str] = [
            [
                ToolCall(id="c1", name="write_file", args={"path": "input/task.md", "content": task}),
                ToolCall(id="c2", name="list_files", args={"path": "."}),
            ],
            [
                ToolCall(
                    id="c3",
                    name="write_file",
                    args={
                        "path": "output/plan.md",
                        "content": f"# plan for {req.session_id}\n1. read task\n2. analyse\n3. write result\n(task digest {digest})\n",
                    },
                ),
                ToolCall(id="c4", name="read_file", args={"path": "input/task.md"}),
            ],
        ]
        if "[probe-traversal]" in task:
            m = re.search(r"\[probe-target=(exec-[0-9a-f]{32})\]", task)
            other = m.group(1) if m else "exec-" + "0" * 32
            plan.append(
                [
                    ToolCall(id="p1", name="read_file", args={"path": f"../{other}/input/task.md"}),
                    ToolCall(id="p2", name="read_file", args={"path": "/etc/passwd"}),
                    ToolCall(id="p3", name="write_file", args={"path": f"../{other}/output/pwned.md", "content": "x"}),
                    ToolCall(id="p4", name="list_files", args={"path": "../.."}),
                ]
            )
        plan.append(
            [
                ToolCall(
                    id="c5",
                    name="write_file",
                    args={
                        "path": "output/result.md",
                        "content": (
                            f"# result for {req.session_id}\ntask digest: {digest}\n"
                            f"tool results so far: {len(tool_results)} "
                            f"(errors: {sum(1 for r in tool_results if not r.ok)})\n"
                        ),
                    },
                ),
                ToolCall(id="c6", name="list_files", args={"path": "output"}),
            ]
        )
        plan.append(f"DONE {req.session_id} digest={digest}")

        step = plan[min(turn, len(plan) - 1)]
        if isinstance(step, str):
            return LLMResponse(final=step, backend=self.name)
        return LLMResponse(tool_calls=step, backend=self.name)


RETRYABLE_STATUS = {408, 409, 429}  # plus every 5xx


class OpenAICompatibleLLM:
    """Any OpenAI-compatible /chat/completions endpoint: mock-llm, xAI (https://api.x.ai/v1), OpenAI, vLLM...

    Errors: 429 / 408 / 409 / 5xx / network -> retryable ApplicationError (Temporal retries the activity);
    other 4xx (bad key, bad model, bad request) -> non-retryable ApplicationError.
    """

    name = "openai"

    def __init__(self) -> None:
        self.base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        self.api_key = os.environ.get("OPENAI_API_KEY", "").strip()  # Secrets from files often end in \n
        self.model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        self.timeout = float(os.environ.get("OPENAI_TIMEOUT_S", "120"))

    @staticmethod
    def _to_openai(m: Message) -> dict:
        if m.role == "assistant":
            d: dict = {"role": "assistant", "content": m.content or None}
            if m.tool_calls:
                d["tool_calls"] = [
                    {"id": c.id, "type": "function", "function": {"name": c.name, "arguments": json.dumps(c.args)}}
                    for c in m.tool_calls
                ]
            return d
        if m.role == "tool" and m.tool_result:
            return {"role": "tool", "tool_call_id": m.tool_result.call_id, "content": m.tool_result.output}
        return {"role": m.role, "content": m.content}

    async def complete(self, req: LLMRequest) -> LLMResponse:
        if not self.api_key:
            raise ApplicationError("OPENAI_API_KEY not set", type="LLMConfigError", non_retryable=True)
        body = {
            "model": self.model,
            "messages": [self._to_openai(m) for m in req.messages],
            "tools": TOOLS_SCHEMA,
            "tool_choice": "auto",
        }
        headers = {"Authorization": f"Bearer {self.api_key}", "X-Client-Id": worker_name()}
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as c:
                r = await c.post(f"{self.base_url}/chat/completions", json=body, headers=headers)
        except httpx.TransportError as e:  # connect/read errors, timeouts
            # type name only, no repr/chain: httpx errors can echo request headers (the bearer key)
            raise ApplicationError(f"LLM transport error: {type(e).__name__}", type="LLMTransportError") from None
        if r.status_code >= 400:
            detail = r.text[:300]
            retry_after = r.headers.get("retry-after")
            if r.status_code in RETRYABLE_STATUS or r.status_code >= 500:
                kind = "LLMRateLimited" if r.status_code == 429 else "LLMServerError"
                delay = timedelta(seconds=float(retry_after)) if retry_after and retry_after.isdigit() else None
                raise ApplicationError(f"HTTP {r.status_code}: {detail}", type=kind, next_retry_delay=delay)
            raise ApplicationError(f"HTTP {r.status_code}: {detail}", type="LLMClientError", non_retryable=True)
        try:
            msg = r.json()["choices"][0]["message"]
            calls = [
                ToolCall(id=tc["id"], name=tc["function"]["name"], args=json.loads(tc["function"]["arguments"] or "{}"))
                for tc in msg.get("tool_calls") or []
            ]
        except (KeyError, IndexError, ValueError, TypeError) as e:
            raise ApplicationError(f"malformed LLM response: {e!r}: {r.text[:300]}", type="LLMBadResponse") from e
        if calls:
            return LLMResponse(tool_calls=calls, backend=self.name)
        return LLMResponse(final=msg.get("content") or "", backend=self.name)


def make_llm() -> LLMClient:
    backend = os.environ.get("LLM_BACKEND", "mock").lower()
    if backend == "mock":
        return MockLLM()
    if backend == "openai":
        return OpenAICompatibleLLM()
    raise ValueError(f"unknown LLM_BACKEND {backend!r}")


@activity.defn(name="llm_call")
async def llm_call(req: LLMRequest) -> LLMResponse:
    """One model turn. Heartbeats every second while waiting, so a dead worker pod is detected in
    ~heartbeat_timeout and the call is retried elsewhere. MOCK_LLM_DELAY_S adds artificial latency."""
    delay = float(os.environ.get("MOCK_LLM_DELAY_S", "0"))

    async def call() -> LLMResponse:
        if delay:
            await asyncio.sleep(delay)
        return await make_llm().complete(req)

    task = asyncio.create_task(call())
    try:
        while True:
            done, _ = await asyncio.wait({task}, timeout=1.0)
            if done:
                break
            activity.heartbeat(req.step)
    except BaseException:
        task.cancel()
        raise
    resp = task.result()
    resp.worker = worker_name()
    return resp
