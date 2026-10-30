"""Execution Manager API.

POST /executions generates execution_id (= workflow_id = session_id = workspace_id) and starts the
agent SESSION directly on task queue `agent`, submitting the task as its first message in the
same call (harness AgentClient.start_and_submit_message = Temporal update-with-start).
No session-manager workflow / task queue is involved.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from temporalio.client import Client, WorkflowExecutionStatus
from temporalio.service import RPCError
from temporal_agent_harness.harness.agent_client import AgentClient
from temporal_agent_harness.harness.agent_protocol import AgentConfig

from . import TASK_QUEUE
from .agent import AGENT_NAME, TaskResult, WorkspaceSession
from .config import connect
from .models import EXECUTION_ID_RE, AgentInput, AgentResult, ExecutionContext


class ExecutionRequest(BaseModel):
    task: str
    max_steps: int = 12
    model: str | None = None  # default: AGENT_MODEL env (grok-4.6)
    engine: str = "native"  # "native" (harness session) | "legacy" (needs AGENT_ENABLE_LEGACY=1 workers)


def create_app(client: Client | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.client = client or await connect()
        yield

    app = FastAPI(title="agent execution manager", lifespan=lifespan)

    @app.post("/executions", status_code=201)
    async def create_execution(req: ExecutionRequest) -> dict:
        ctx = ExecutionContext.new()  # fresh id every time; never reuse a session/workspace
        c: Client = app.state.client
        if req.engine == "legacy":
            handle = await c.start_workflow("AgentWorkflow", AgentInput(ctx=ctx, task=req.task, max_steps=req.max_steps),
                                            id=ctx.execution_id, task_queue=TASK_QUEUE)
            return {**ctx.model_dump(), "workflow_id": handle.id, "run_id": handle.result_run_id, "engine": "legacy"}
        if req.engine != "native":
            raise HTTPException(400, "engine must be native|legacy")
        session = WorkspaceSession(execution_id=ctx.execution_id, max_turns=req.max_steps,
                                   model=req.model or os.environ.get("AGENT_MODEL", "grok-4.6"))
        reply = await AgentClient(c, ctx.execution_id).start_and_submit_message(
            "run_task", {"task": req.task},
            workflow_name=AGENT_NAME, task_queue=TASK_QUEUE,
            start_config=AgentConfig(), start_data=session,
            update_id=f"{ctx.execution_id}-task",  # idempotent re-delivery
        )
        desc = await c.get_workflow_handle(ctx.execution_id).describe()
        return {**ctx.model_dump(), "workflow_id": ctx.execution_id, "run_id": desc.run_id, "engine": "native",
                "model": session.model, "message_id": reply.message_id, "turn_number": reply.turn_number}

    @app.get("/executions/{execution_id}")
    async def get_execution(execution_id: str) -> dict:
        if not EXECUTION_ID_RE.fullmatch(execution_id):
            raise HTTPException(400, "bad execution id")
        handle = app.state.client.get_workflow_handle(execution_id)
        try:
            desc = await handle.describe()
        except RPCError:
            raise HTTPException(404, "not found")
        out: dict = {"execution_id": execution_id, "workflow_type": desc.workflow_type,
                     "status": desc.status.name if desc.status else None}
        if desc.workflow_type == AGENT_NAME:
            res = await handle.query("result", result_type=TaskResult | None)
            out["session_status"] = "closed" if desc.status == WorkflowExecutionStatus.COMPLETED else "open"
            out["result"] = res.model_dump() if res else None
        elif desc.status == WorkflowExecutionStatus.COMPLETED:
            legacy = app.state.client.get_workflow_handle(execution_id, result_type=AgentResult)
            out["result"] = (await legacy.result()).model_dump()
        return out

    @app.post("/executions/{execution_id}/close", status_code=202)
    async def close_execution(execution_id: str) -> dict:
        """Manual close (the session normally closes itself when its task is done)."""
        if not EXECUTION_ID_RE.fullmatch(execution_id):
            raise HTTPException(400, "bad execution id")
        try:
            await app.state.client.get_workflow_handle(execution_id).signal("close")
        except RPCError as e:
            raise HTTPException(409, f"cannot close: {e.message}")
        return {"execution_id": execution_id, "close": "requested"}

    @app.get("/healthz")
    async def healthz() -> dict:
        return {"ok": True}

    return app


app = create_app() if os.environ.get("AGENT_API_AUTOCREATE", "1") == "1" else None
