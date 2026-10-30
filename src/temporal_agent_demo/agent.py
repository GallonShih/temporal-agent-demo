"""WorkspaceAgent: the agent on temporal-agent-harness's NATIVE session runtime.

One agent session == one workflow execution == one workspace:
    execution_id == workflow_id == session_id == workspace_id   (never the Temporal run_id)

Harness pieces used
* ``@agent.defn`` / ``@agent.init`` / ``@agent.setup`` / ``@agent.teardown`` / ``@agent.accepts``:
  the harness generates the workflow ``run`` (turn loop), the ``send_agent_message`` update,
  the ``close`` signal and the ``agent_status`` / ``agent_interface`` queries.
* ``AgentWorkflowRunner``: turn lifecycle, tool events, approval policy (skip-all: our tools are
  confined to the workspace anyway).
* ``agent.vfs_mount("/workspace", WorkspaceDisk)``, bound per session to
  ``WorkspaceDiskConfig(workspace_id=<execution id>)``: the model only ever sees the virtual path
  ``/workspace``; every file op is a ``vfs.workspace-disk.*`` activity confined to
  ``/data/workspaces/<execution id>`` (atomic, idempotent writes).
* ``agent.code_mode_tool([], name="run_code", mounts=[workspace])``: Code Mode (Monty sandbox)
  scripts that read/write ``/workspace`` with ``open()`` / ``pathlib``.
* Three plain ``@agent.tool_defn`` file tools over the same bound mount (cheap single-file ops).
* OpenAI Agents SDK ``Agent`` + ``Runner.run`` (non-streamed) with harness tools adapted by
  ``as_openai_agent_tool``; the ``OpenAIAgentsPlugin`` turns every model call into a
  ``invoke_model_activity`` (chat-completions against the host llm_gateway / mock).

Lifecycle: the API starts the session AND submits the task in one call
(``AgentClient.start_and_submit_message`` = Temporal update-with-start). ``run_task`` runs the
task, then CLOSES the session itself (the harness ``close`` signal, sent to its own workflow);
``@agent.teardown`` then writes ``session.json`` with status ``closed`` and the workflow completes.
"""

from __future__ import annotations

import json
from datetime import timedelta
from pathlib import PurePosixPath
from typing import Any

from pydantic import BaseModel, Field, field_validator
from temporalio import workflow
from temporalio.contrib.workflow_streams import WorkflowStream
from temporalio.exceptions import ActivityError, ApplicationError

with workflow.unsafe.imports_passed_through():
    from agents import Agent as SDKAgent
    from agents import Runner
    from agents.exceptions import AgentsException

    from temporal_agent_harness.ai_sdks.openai_agents_harness import as_openai_agent_tool
    from temporal_agent_harness.harness import agent
    from temporal_agent_harness.harness.agent_protocol import AgentConfig, MidTurn, ToolApprovalPolicy
    from temporal_agent_harness.harness.agent_workflow import AgentWorkflowRunner, Injected

    from .models import EXECUTION_ID_RE
    from .workspace_fs import WorkspaceDisk, WorkspaceDiskConfig

AGENT_NAME = "WorkspaceAgent"
MOUNT = "/workspace"
MAX_OUTPUT = 4000

SYSTEM_PROMPT = f"""\
You are a file-working agent. Your private workspace is mounted at `{MOUNT}` (input/, output/, temp/).
Only paths under `{MOUNT}` exist for you. Do the task by calling tools and put deliverables under \
`{MOUNT}/output/`. Use `write_file` / `read_file` / `list_files` for single-file operations, and \
`run_code` for multi-step file work in one script. When finished, reply with a one-paragraph \
summary and no tool calls."""


# ---------------------------------------------------------------- session data / messages / results
class WorkspaceSession(BaseModel):
    """Init data (second workflow argument). Chosen by the Execution Manager, never the model."""

    execution_id: str
    model: str = "grok-4.6"
    max_turns: int = Field(default=12, ge=1, le=50)

    @field_validator("execution_id")
    @classmethod
    def _shape(cls, v: str) -> str:
        if not EXECUTION_ID_RE.fullmatch(v):
            raise ValueError(f"invalid execution id {v!r}")
        return v


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


# ---------------------------------------------------------------- tools over the bound mount
# NB: the harness calls user_fn(*model_args, **injected, **model_kwargs), so Injected params must
# come after the model params (keyword-only here).
def _rel(path: str) -> PurePosixPath:
    """Virtual path -> path relative to the workspace. '..' is NOT collapsed here: WorkspaceDisk
    (inside the activity) is the authority and rejects it."""
    p = PurePosixPath(path)
    if p.is_absolute():
        if p == PurePosixPath(MOUNT):
            return PurePosixPath(".")
        if not str(p).startswith(MOUNT + "/"):
            raise PermissionError(f"PermissionError: path outside {MOUNT}: {path!r}")
        p = PurePosixPath(str(p)[len(MOUNT) + 1 :])
    return p


async def _fs(op: str, fs: Any, *args: Any) -> Any:
    try:
        return await getattr(fs, op)(*args)
    except ActivityError as e:  # surface WorkspaceDisk's PermissionError etc. to the model
        c = e.cause
        kind = getattr(c, "type", None) or type(c).__name__
        raise ApplicationError(f"{kind}: {getattr(c, 'message', c)}", type=kind, non_retryable=True) from None


@agent.tool_defn(inherently_safe=True)
async def write_file(path: str, content: str, *, ws: Injected[Any]) -> str:
    """Write (create or replace) a UTF-8 text file. `path` is under /workspace, e.g. /workspace/output/report.md."""
    await _fs("write", ws, _rel(path), content.encode())
    return f"wrote {path} ({len(content.encode())} bytes)"


@agent.tool_defn(inherently_safe=True)
async def read_file(path: str, *, ws: Injected[Any]) -> str:
    """Read a UTF-8 text file under /workspace."""
    data: bytes = await _fs("read", ws, _rel(path))
    return data.decode(errors="replace")[:MAX_OUTPUT]


@agent.tool_defn(inherently_safe=True)
async def list_files(path: str = MOUNT, *, ws: Injected[Any]) -> list[str]:
    """List a directory under /workspace (default: the workspace root)."""
    return await _fs("list", ws, _rel(path))


# ---------------------------------------------------------------- the agent
@agent.defn(name=AGENT_NAME)
class WorkspaceAgent:
    workspace = agent.vfs_mount(
        MOUNT, WorkspaceDisk,
        description="This execution's private workspace (input/, output/, temp/). Persistent, shared PVC.",
    )

    @agent.init
    def __init__(self, config: AgentConfig, data: WorkspaceSession) -> None:
        self._runner = AgentWorkflowRunner(
            config, stream=WorkflowStream(),
            approval_policy_default=ToolApprovalPolicy.dangerously_skip_all(),
        )
        self._data = data
        # Bound by the workflow input (Execution Manager), NOT by the model.
        self._ws = self.workspace.bind(WorkspaceDiskConfig(workspace_id=data.execution_id),
                                       timeout=timedelta(seconds=30))
        self._code_tool = agent.code_mode_tool([], name="run_code", mounts=[self._ws])
        self._result: TaskResult | None = None
        self._outcome = "open"
        self._task: str | None = None

    # -- invariants + workspace layout --------------------------------------------------
    @agent.setup
    async def setup(self) -> None:
        info = workflow.info()
        if info.workflow_id != self._data.execution_id:
            raise ApplicationError(
                f"workflow_id {info.workflow_id!r} != execution_id {self._data.execution_id!r}",
                type="IdMismatch", non_retryable=True)
        fs = self._ws.backend
        for d in (".", "input", "output", "temp"):
            await fs.mkdir(PurePosixPath(d))
        await self._write_session("open")

    @agent.teardown
    async def teardown(self) -> None:
        await self._write_session("closed")

    # -- the one message type ---------------------------------------------------------------
    @agent.accepts(mid_turn=MidTurn.REJECT)
    async def run_task(self, message: TaskMessage) -> TaskResult:
        """Run one task in this session's workspace, then close the session."""
        if self._result is not None:
            raise ApplicationError("this session already ran its task", type="SessionDone", non_retryable=True)
        self._task = message.task
        fs = self._ws.backend
        await fs.write(PurePosixPath("input/task.md"), message.task.encode())
        injections = {"ws": fs}
        sdk_agent = SDKAgent(
            name=AGENT_NAME,
            instructions=f"{SYSTEM_PROMPT}\nsession_id={self._data.execution_id}",
            model=self._data.model,
            tools=[as_openai_agent_tool(self._runner, t, injections=injections)
                   for t in (write_file, read_file, list_files)]
                  + [as_openai_agent_tool(self._runner, self._code_tool)],
        )
        items: list[Any] = []
        model_calls = 0
        try:
            res = await Runner.run(sdk_agent, input=message.task, max_turns=self._data.max_turns)
            items, model_calls = res.new_items, len(res.raw_responses)
            outcome, final = "completed", str(res.final_output)
        except AgentsException as e:  # max turns, bad model output, ... -> fail the TASK, still close
            rd = getattr(e, "run_data", None)
            items = list(getattr(rd, "new_items", []) or [])
            model_calls = len(getattr(rd, "raw_responses", []) or [])
            outcome, final = "failed", f"{type(e).__name__}: {e}"
        except ActivityError as e:  # model activity exhausted its retries / non-retryable 4xx
            outcome, final = "failed", f"model call failed: {getattr(e.cause, 'message', e.cause)}"
        steps = _tool_steps(items)
        await fs.write(PurePosixPath("temp/steps.json"),
                       json.dumps([s.model_dump() for s in steps], indent=2).encode())
        info = workflow.info()
        self._outcome = outcome
        self._result = TaskResult(
            execution_id=info.workflow_id, session_id=info.workflow_id, workspace_id=self._data.execution_id,
            run_id=info.run_id, outcome=outcome, final=final, session_status="closed",
            model=self._data.model, tool_steps=steps, model_calls=model_calls)
        # Explicitly end the session: the harness's own `close` signal, sent to this workflow.
        # The turn loop drains this turn, then teardown writes session.json "closed".
        # (A workflow cannot signal itself -- Temporal answers EXTERNAL_WORKFLOW_EXECUTION_NOT_FOUND --
        # so call the harness's own close-signal handler directly; same effect as the signal.)
        self._runner._handle_close()
        return self._result

    @workflow.query
    def result(self) -> TaskResult | None:
        return self._result

    async def _write_session(self, status: str) -> None:
        info = workflow.info()
        doc = {
            "execution_id": info.workflow_id, "session_id": info.workflow_id,
            "workspace_id": self._data.execution_id, "workflow_id": info.workflow_id, "run_id": info.run_id,
            "agent": AGENT_NAME, "model": self._data.model, "status": status, "outcome": self._outcome,
            "at": workflow.now().isoformat(),
        }
        if self._result:
            doc["final"] = self._result.final
        await self._ws.backend.write(PurePosixPath("session.json"), json.dumps(doc, indent=2).encode())


def _tool_steps(items: list[Any]) -> list[ToolStep]:
    steps: dict[str, ToolStep] = {}
    for it in items:
        raw = getattr(it, "raw_item", None)
        kind = getattr(it, "type", "")
        if kind == "tool_call_item":
            cid = getattr(raw, "call_id", None) or (raw.get("call_id") if isinstance(raw, dict) else "")
            name = getattr(raw, "name", None) or (raw.get("name") if isinstance(raw, dict) else "")
            args = getattr(raw, "arguments", None) or (raw.get("arguments") if isinstance(raw, dict) else "")
            steps[cid] = ToolStep(call_id=cid, name=name, arguments=args)
        elif kind == "tool_call_output_item":
            cid = raw.get("call_id") if isinstance(raw, dict) else getattr(raw, "call_id", "")
            out = str(getattr(it, "output", ""))
            if cid in steps:
                steps[cid].output = out[:MAX_OUTPUT]
                steps[cid].ok = not (out.startswith("Tool '") and "' failed:" in out[:80]) and "Error" not in out[:40]
    return list(steps.values())
