"""Start one execution (= one agent session) and print its result.

    python -m temporal_agent_demo.starter "Write a haiku to /workspace/output/haiku.md"
    python -m temporal_agent_demo.starter --model grok-mock --no-wait "some task"

Env: TEMPORAL_ADDRESS (default localhost:7233), TEMPORAL_NAMESPACE, OPENAI_MODEL (default grok-4.6).
The model call itself happens on the workers, so no LLM key is needed here.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os

from temporalio.client import Client
from temporal_agent_harness.harness.agent_client import AgentClient
from temporal_agent_harness.harness.agent_protocol import AgentConfig

from .config import connect
from .shared import AGENT_NAME, DEFAULT_MODEL, TASK_QUEUE, ExecutionContext, TaskResult, WorkspaceSession


async def start_execution(client: Client, task: str, model: str | None = None, max_turns: int = 12) -> dict:
    """Start a session with execution_id == workflow_id == session_id == workspace_id and submit
    the task as its first message in the same call (update-with-start). Returns the ids."""
    ctx = ExecutionContext.new()  # fresh id every time; never reuse a session/workspace
    session = WorkspaceSession(execution_id=ctx.execution_id, max_turns=max_turns,
                               model=model or os.environ.get("OPENAI_MODEL", DEFAULT_MODEL))
    reply = await AgentClient(client, ctx.execution_id).start_and_submit_message(
        "run_task", {"task": task},
        workflow_name=AGENT_NAME, task_queue=TASK_QUEUE,
        start_config=AgentConfig(), start_data=session,
        update_id=f"{ctx.execution_id}-task",  # idempotent re-delivery
    )
    desc = await client.get_workflow_handle(ctx.execution_id).describe()
    return {**ctx.model_dump(), "workflow_id": ctx.execution_id, "run_id": desc.run_id,
            "model": session.model, "message_id": reply.message_id}


async def wait_result(client: Client, execution_id: str) -> TaskResult:
    handle = client.get_workflow_handle(execution_id)
    await handle.result()  # the session closes itself after its task
    return await handle.query("result", result_type=TaskResult)


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("task")
    ap.add_argument("--model", default=None)
    ap.add_argument("--max-turns", type=int, default=12)
    ap.add_argument("--no-wait", action="store_true")
    args = ap.parse_args()
    client = await connect()
    started = await start_execution(client, args.task, args.model, args.max_turns)
    print(json.dumps(started, indent=2))
    if not args.no_wait:
        res = await wait_result(client, started["execution_id"])
        print(json.dumps(res.model_dump(), indent=2))


if __name__ == "__main__":
    asyncio.run(main())
