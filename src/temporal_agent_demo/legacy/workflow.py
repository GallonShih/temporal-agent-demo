"""AgentWorkflow: one workflow execution == one agent session == one workspace."""

from __future__ import annotations

import json
from datetime import timedelta
from pathlib import PurePosixPath

from temporalio import workflow
from temporalio.common import RetryPolicy
from temporalio.exceptions import ActivityError, ApplicationError

with workflow.unsafe.imports_passed_through():
    from .llm import llm_call
    from ..models import (
        AgentInput,
        AgentResult,
        LLMRequest,
        LLMResponse,
        Message,
        StepRecord,
        ToolCall,
        ToolResult,
    )
    from ..workspace_fs import WorkspaceDisk, WorkspaceDiskConfig

SYSTEM_PROMPT = (
    "You are a file-working agent. Your workspace is mounted at '.'; use relative paths only. "
    "Tools: write_file(path, content), read_file(path), list_files(path). "
    "Do the task by calling tools; put deliverables under output/. "
    "When finished, reply with a one-paragraph summary and no tool calls."
)


@workflow.defn(name="AgentWorkflow")
class AgentWorkflow:
    def __init__(self) -> None:
        self._status = "starting"
        self._step = 0

    @workflow.query
    def status(self) -> dict:
        return {"status": self._status, "step": self._step}

    @workflow.run
    async def run(self, inp: AgentInput) -> AgentResult:
        info = workflow.info()
        ctx = inp.ctx
        # Execution Manager invariant: workflow_id == session_id == workspace_id == execution_id.
        if not (info.workflow_id == ctx.execution_id == ctx.session_id == ctx.workspace_id):
            raise ApplicationError("execution/workflow/session/workspace ids differ", non_retryable=True)

        # The workspace is bound by the workflow input, NOT by the model.
        fs = WorkspaceDisk.backend(WorkspaceDiskConfig(workspace_id=ctx.workspace_id))
        for d in (".", "input", "output", "temp"):
            await fs.mkdir(PurePosixPath(d))
        await fs.write(PurePosixPath("session.json"), self._session_doc(inp, "open").encode())
        self._status = "open"

        messages = [
            Message(role="system", content=f"{SYSTEM_PROMPT}\nsession_id={ctx.session_id}"),
            Message(role="user", content=inp.task),
        ]
        steps: list[StepRecord] = []
        final = None
        for step in range(inp.max_steps):
            self._step = step
            resp: LLMResponse = await workflow.execute_activity(
                llm_call,
                LLMRequest(session_id=ctx.session_id, step=step, messages=messages),
                start_to_close_timeout=timedelta(minutes=2),
                heartbeat_timeout=timedelta(seconds=5),
                # 429/5xx/network -> retried here; other 4xx raise non-retryable ApplicationError
                retry_policy=RetryPolicy(
                    maximum_attempts=6, initial_interval=timedelta(seconds=1), backoff_coefficient=2.0,
                    maximum_interval=timedelta(seconds=20),
                ),
            )
            if resp.final is not None:
                final = resp.final
                messages.append(Message(role="assistant", content=resp.final))
                break
            messages.append(Message(role="assistant", tool_calls=resp.tool_calls))
            results = [await self._run_tool(fs, c) for c in resp.tool_calls]
            messages.extend(Message(role="tool", tool_result=r) for r in results)
            rec = StepRecord(step=step, llm_worker=resp.worker, tool_calls=resp.tool_calls, tool_results=results)
            steps.append(rec)
            # one file per step, fixed name => retry-safe (no append)
            await fs.write(PurePosixPath(f"temp/step-{step:03d}.json"), rec.model_dump_json(indent=2).encode())

        if final is None:
            final = "max_steps reached"
        # Explicitly close the session (don't just stop after the last model reply).
        await fs.write(PurePosixPath("session.json"), self._session_doc(inp, "closed", final).encode())
        self._status = "closed"
        return AgentResult(ctx=ctx, run_id=info.run_id, final=final, session_status="closed", steps=steps)

    @staticmethod
    def _session_doc(inp: AgentInput, status: str, final: str | None = None) -> str:
        return json.dumps(
            {**inp.ctx.model_dump(), "task": inp.task, "status": status, "final": final}, indent=2
        )

    async def _run_tool(self, fs, call: ToolCall) -> ToolResult:
        try:
            path = PurePosixPath(str(call.args.get("path", ".")))
            if call.name == "write_file":
                await fs.write(path, str(call.args.get("content", "")).encode())
                out = f"wrote {path}"
            elif call.name == "read_file":
                out = (await fs.read(path)).decode(errors="replace")
            elif call.name == "list_files":
                out = "\n".join(await fs.list(path))
            else:
                return ToolResult(call_id=call.id, name=call.name, ok=False, output="unknown tool")
            return ToolResult(call_id=call.id, name=call.name, ok=True, output=out)
        except ActivityError as e:
            cause = e.cause
            kind = cause.type if isinstance(cause, ApplicationError) else type(cause).__name__
            msg = cause.message if isinstance(cause, ApplicationError) else str(cause)
            return ToolResult(call_id=call.id, name=call.name, ok=False, output=f"{kind}: {msg}")
