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
