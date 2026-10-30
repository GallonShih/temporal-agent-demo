"""Execution Manager HTTP API.

POST /executions generates execution_id (= workflow_id = session_id = workspace_id) and starts the
agent SESSION directly on task queue `agent`, submitting the task as its first message in the
same call (harness AgentClient.start_and_submit_message = Temporal update-with-start).
No session-manager workflow / task queue is involved.

    uvicorn temporal_agent_demo.api:app --port 8000
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from temporalio.client import Client, WorkflowExecutionStatus
from temporalio.service import RPCError

from .config import connect
from .shared import AGENT_NAME, EXECUTION_ID_RE, TaskResult
from .starter import start_execution


class ExecutionRequest(BaseModel):
    task: str
    max_steps: int = 12  # max model turns
    model: str | None = None  # default: OPENAI_MODEL env (grok-4.6)


def create_app(client: Client | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.client = client or await connect()
        yield

    app = FastAPI(title="agent execution manager", lifespan=lifespan)

    def _check(execution_id: str) -> None:
        if not EXECUTION_ID_RE.fullmatch(execution_id):
            raise HTTPException(400, "bad execution id")

    @app.post("/executions", status_code=201)
    async def create_execution(req: ExecutionRequest) -> dict:
        return await start_execution(app.state.client, req.task, req.model, req.max_steps)

    @app.get("/executions/{execution_id}")
    async def get_execution(execution_id: str) -> dict:
        _check(execution_id)
        handle = app.state.client.get_workflow_handle(execution_id)
        try:
            desc = await handle.describe()
        except RPCError:
            raise HTTPException(404, "not found")
        if desc.workflow_type != AGENT_NAME:
            raise HTTPException(404, "not an agent execution")
        res = await handle.query("result", result_type=TaskResult | None)
        return {"execution_id": execution_id, "workflow_type": desc.workflow_type,
                "status": desc.status.name if desc.status else None,
                "session_status": "closed" if desc.status == WorkflowExecutionStatus.COMPLETED else "open",
                "result": res.model_dump() if res else None}

    @app.post("/executions/{execution_id}/close", status_code=202)
    async def close_execution(execution_id: str) -> dict:
        """Manual close (the session normally closes itself when its task is done)."""
        _check(execution_id)
        try:
            await app.state.client.get_workflow_handle(execution_id).signal("close")
        except RPCError as e:
            raise HTTPException(409, f"cannot close: {e.message}")
        return {"execution_id": execution_id, "close": "requested"}

    @app.get("/healthz")
    async def healthz() -> dict:
        return {"ok": True}

    return app


app = create_app()  # connects lazily in lifespan
