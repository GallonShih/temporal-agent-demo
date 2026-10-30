"""Shared pydantic models (workflow inputs/outputs, LLM protocol)."""

from __future__ import annotations

import re
import uuid
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

# exec-<uuid4 hex>. The ONLY shape a workspace id may take; validated again by the filesystem.
EXECUTION_ID_RE = re.compile(r"exec-[0-9a-f]{32}")


def new_execution_id() -> str:
    return f"exec-{uuid.uuid4().hex}"


class ExecutionContext(BaseModel):
    """execution_id == workflow_id == session_id == workspace_id (never the Temporal run_id)."""

    execution_id: str
    session_id: str
    workspace_id: str

    @field_validator("execution_id", "session_id", "workspace_id")
    @classmethod
    def _shape(cls, v: str) -> str:
        if not EXECUTION_ID_RE.fullmatch(v):
            raise ValueError(f"invalid execution id {v!r}")
        return v

    @classmethod
    def new(cls) -> "ExecutionContext":
        eid = new_execution_id()
        return cls(execution_id=eid, session_id=eid, workspace_id=eid)


class AgentInput(BaseModel):
    ctx: ExecutionContext
    task: str
    max_steps: int = 12


ToolName = str  # real models may hallucinate tool names; the workflow rejects unknown ones


class ToolCall(BaseModel):
    id: str
    name: ToolName
    args: dict[str, Any] = Field(default_factory=dict)


class ToolResult(BaseModel):
    call_id: str
    name: str
    ok: bool
    output: str


class Message(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    tool_result: ToolResult | None = None


class LLMRequest(BaseModel):
    session_id: str
    step: int
    messages: list[Message]


class LLMResponse(BaseModel):
    tool_calls: list[ToolCall] = Field(default_factory=list)
    final: str | None = None  # set => agent is done
    worker: str = ""  # which worker process served the call (for the demo evidence)
    backend: str = "mock"


class StepRecord(BaseModel):
    step: int
    llm_worker: str
    tool_calls: list[ToolCall]
    tool_results: list[ToolResult]


class AgentResult(BaseModel):
    ctx: ExecutionContext
    run_id: str
    final: str
    session_status: Literal["closed"]
    steps: list[StepRecord]
