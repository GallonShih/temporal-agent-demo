"""Shared constants and pydantic models (workflow input, messages, results, ids).

Imported by the workflow (inside the sandbox, passed through), the worker, the API and the starter.
"""

from __future__ import annotations

import re
import uuid

from pydantic import BaseModel, Field, field_validator

TASK_QUEUE = "agent"
AGENT_NAME = "WorkspaceAgent"  # workflow type
MOUNT = "/workspace"  # the only path the model sees; bound to /data/workspaces/<execution id>
MAX_OUTPUT = 4000  # max chars of a tool output kept / returned
DEFAULT_MODEL = "grok-4.6"

# exec-<uuid4 hex>. The ONLY shape a workspace id may take; validated again by the filesystem.
EXECUTION_ID_RE = re.compile(r"exec-[0-9a-f]{32}")


def new_execution_id() -> str:
    return f"exec-{uuid.uuid4().hex}"


def _check_id(v: str) -> str:
    if not EXECUTION_ID_RE.fullmatch(v):
        raise ValueError(f"invalid execution id {v!r}")
    return v


class ExecutionContext(BaseModel):
    """execution_id == workflow_id == session_id == workspace_id (never the Temporal run_id)."""

    execution_id: str
    session_id: str
    workspace_id: str

    @field_validator("execution_id", "session_id", "workspace_id")
    @classmethod
    def _shape(cls, v: str) -> str:
        return _check_id(v)

    @classmethod
    def new(cls) -> "ExecutionContext":
        eid = new_execution_id()
        return cls(execution_id=eid, session_id=eid, workspace_id=eid)


class WorkspaceSession(BaseModel):
    """Init data (second workflow argument). Chosen by the Execution Manager, never the model."""

    execution_id: str
    model: str = DEFAULT_MODEL
    max_turns: int = Field(default=12, ge=1, le=50)

    @field_validator("execution_id")
    @classmethod
    def _shape(cls, v: str) -> str:
        return _check_id(v)


class TaskMessage(BaseModel):
    """The task for this session's agent to carry out in its workspace."""

    task: str = Field(min_length=1, max_length=20_000)


class ToolStep(BaseModel):
    """One tool call the model made and what it returned."""

    call_id: str
    name: str
    arguments: str
    output: str | None = None
    ok: bool | None = None


class TaskResult(BaseModel):
    """Outcome of the session's task."""

    execution_id: str
    session_id: str
    workspace_id: str
    run_id: str
    outcome: str  # completed | failed
    final: str
    session_status: str  # open | closed
    model: str
    tool_steps: list[ToolStep] = []
    model_calls: int = 0
