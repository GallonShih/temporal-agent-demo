"""Shared constants and pydantic models (workflow input, messages, results, ids).

Imported by the workflow (inside the sandbox, passed through), the worker, the API and the starter.
"""

from __future__ import annotations

import hashlib
import re
import uuid
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

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


ORCHESTRATOR_NAME = "OrchestratorAgent"  # workflow type of the parent with subagent tools
WorkspaceMode = Literal["inherit", "isolated"]
ROLE_NAME_RE = re.compile(r"[a-z][a-z0-9_]{0,31}")


class AgentRole(BaseModel):
    """How a WorkspaceAgent runs as a SUBAGENT. Built by developer code (subagent init_data), never by a model."""

    model_config = ConfigDict(extra="forbid")

    name: str
    instructions: str = Field(default="", max_length=4000)
    read_only: bool = False  # enforced by WorkspaceDisk (all writes rejected) + no write_file tool
    workspace_mode: WorkspaceMode = "isolated"

    @field_validator("name")
    @classmethod
    def _name(cls, v: str) -> str:
        if not ROLE_NAME_RE.fullmatch(v):
            raise ValueError(f"invalid role name {v!r}")
        return v


class WorkspaceSession(BaseModel):
    """Init data (second workflow argument). Chosen by the Execution Manager / parent code, never the model.

    Top-level session: execution_id required (== workflow_id == workspace_id), role None.
    Subagent: role set, execution_id None; the workspace is resolved from the Temporal parent link
    (inherit) or derived from the child's own workflow id (isolated) - see resolve_workspace_id."""

    model_config = ConfigDict(extra="forbid")  # no workspace_id / path can be smuggled in

    execution_id: str | None = None
    model: str = DEFAULT_MODEL
    max_turns: int = Field(default=12, ge=1, le=50)
    role: AgentRole | None = None

    @field_validator("execution_id")
    @classmethod
    def _shape(cls, v: str | None) -> str | None:
        return None if v is None else _check_id(v)

    @model_validator(mode="after")
    def _exactly_one(self) -> "WorkspaceSession":
        if (self.role is None) == (self.execution_id is None):
            raise ValueError("exactly one of execution_id (top-level) or role (subagent) must be set")
        return self


class OrchestratorSession(WorkspaceSession):
    """Init data of OrchestratorAgent. `helper_workspace_mode` (from the start request) picks the
    workspace mode of the `helper` subagent; reviewer (inherit) and researcher (isolated) are fixed."""

    helper_workspace_mode: WorkspaceMode = "isolated"

    @model_validator(mode="after")
    def _top_level(self) -> "OrchestratorSession":
        if self.role is not None:
            raise ValueError("the orchestrator is a top-level session")
        return self


def isolated_workspace_id(workflow_id: str) -> str:
    """Deterministic own workspace for an isolated subagent: exec-<sha256(child workflow id)[:32]>."""
    return "exec-" + hashlib.sha256(workflow_id.encode()).hexdigest()[:32]


def resolve_workspace_id(data: WorkspaceSession, workflow_id: str, parent_workflow_id: str | None) -> str:
    """The ONLY place a session's workspace id is decided (raises ValueError on any mismatch)."""
    if data.role is None:
        if data.execution_id != workflow_id:
            raise ValueError(f"workflow_id {workflow_id!r} != execution_id {data.execution_id!r}")
        return _check_id(workflow_id)
    if parent_workflow_id is None:
        raise ValueError("a role (subagent) session must be started as a child workflow")
    if data.role.workspace_mode == "inherit":
        return _check_id(parent_workflow_id)  # parent's workspace == parent's workflow id
    return isolated_workspace_id(workflow_id)


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

    role: str | None = None
    workspace_mode: str | None = None  # None (top-level) | inherit | isolated
    read_only: bool = False

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


# ---- Claude Agent SDK activity (activities.py / claude_workflow.py) ------------------------------
CLAUDE_WORKFLOW = "ClaudeTaskWorkflow"
SKILL_NAME_RE = re.compile(r"[a-z0-9][a-z0-9-]{0,63}")
DEFAULT_CLAUDE_BUILTINS = ["Read", "Write", "Edit", "Glob", "Grep"]
ALLOWED_CLAUDE_BUILTINS = {"Read", "Write", "Edit", "Glob", "Grep", "TodoWrite"}  # never Bash/WebFetch/WebSearch


class SystemPromptPreset(BaseModel):
    """Claude Code's own system prompt plus our text appended."""

    preset: str = "claude_code"
    append: str = ""


class ClaudeAgentInput(BaseModel):
    """Input of the run_claude_agent activity. Recorded in workflow history: never put secrets here."""

    execution_id: str
    prompt: str = Field(min_length=1, max_length=20_000)
    system_prompt: str | SystemPromptPreset | None = None
    skills: dict[str, str] = {}  # skill name -> SKILL.md text, written to <ws>/.claude/skills/<name>/SKILL.md
    enabled_skills: list[str] | None = None  # None = all provided skills
    tools: list[str] = []  # names from the worker-side registry (claude_tools.REGISTRY)
    builtin_tools: list[str] = Field(default_factory=lambda: list(DEFAULT_CLAUDE_BUILTINS))
    model: str = DEFAULT_MODEL
    max_turns: int = Field(default=10, ge=1, le=50)
    # Session persistence / resume. persist_session keeps Claude Code's HOME + CLAUDE_CONFIG_DIR
    # (transcripts) at <SESSIONS_ROOT>/<execution id> on the shared PVC, so any worker can resume.
    persist_session: bool = False
    resume_session_id: str | None = None  # continue this Claude Code session (needs persist_session)
    fork_session: bool = False  # with resume: branch into a new session id instead of appending
    step: str | None = None  # label for logs / results

    @field_validator("execution_id")
    @classmethod
    def _shape(cls, v: str) -> str:
        return _check_id(v)

    @field_validator("resume_session_id")
    @classmethod
    def _sid(cls, v: str | None) -> str | None:
        if v is not None and not re.fullmatch(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", v):
            raise ValueError("resume_session_id must be a session UUID")
        return v

    @field_validator("skills")
    @classmethod
    def _skills(cls, v: dict[str, str]) -> dict[str, str]:
        for name, text in v.items():
            if not SKILL_NAME_RE.fullmatch(name):
                raise ValueError(f"invalid skill name {name!r}")
            if len(text) > 50_000:
                raise ValueError(f"skill {name!r} too large")
        return v

    @field_validator("builtin_tools")
    @classmethod
    def _builtins(cls, v: list[str]) -> list[str]:
        bad = set(v) - ALLOWED_CLAUDE_BUILTINS
        if bad:
            raise ValueError(f"builtin tools not allowed: {sorted(bad)}")
        return v


class ClaudeAgentResult(BaseModel):
    execution_id: str
    is_error: bool
    subtype: str
    result: str
    num_turns: int = 0
    session_id: str | None = None
    usage: dict = {}
    tool_uses: list[str] = []  # tool names in call order
    denied: list[str] = []  # tool calls rejected by the workspace hook
    files: list[str] = []  # workspace files after the run (relative, excluding .claude/)
    step: str | None = None
    resumed_from: str | None = None
    worker: str | None = None  # pod that ran the activity
    attempt: int = 1


# ---- DevTaskWorkflow (dev_workflow.py) --------------------------------------------------------
DEV_WORKFLOW = "DevTaskWorkflow"


class DevTaskInput(BaseModel):
    execution_id: str
    task: str = Field(min_length=1, max_length=20_000)
    model: str = DEFAULT_MODEL
    max_fix_iterations: int = Field(default=3, ge=0, le=10)
    max_turns_per_step: int = Field(default=16, ge=1, le=50)
    # Hidden acceptance tests (relative path -> content) written AFTER implement, before testing.
    acceptance_tests: dict[str, str] = {}

    @field_validator("execution_id")
    @classmethod
    def _shape(cls, v: str) -> str:
        return _check_id(v)


class PytestRun(BaseModel):
    exit_code: int
    passed: bool
    output: str
    worker: str | None = None


class StepRecord(BaseModel):
    name: str  # plan | implement | test-N | fix-N | review
    status: str  # running | ok | failed | error
    session_id: str | None = None
    resumed_from: str | None = None
    worker: str | None = None
    attempt: int = 1
    turns: int = 0
    usage: dict = {}
    tool_uses: list[str] = []
    denied: list[str] = []
    summary: str = ""


class DevTaskResult(BaseModel):
    execution_id: str
    status: str  # passed | failed | running
    fix_iterations: int = 0
    steps: list[StepRecord] = []
    files: list[str] = []
