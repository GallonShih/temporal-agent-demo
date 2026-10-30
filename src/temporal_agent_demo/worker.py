"""Agent worker: every replica runs this identical process on task queue `agent`.

Registers the native harness agent (WorkspaceAgent). Model activities come from the
OpenAIAgentsPlugin and WorkspaceDisk vfs.* / harness activities from AgentHarnessPlugin, both
on the client (see config.py). AGENT_ENABLE_LEGACY=1 also registers the old hand-written
AgentWorkflow + llm_call (temporal_agent_demo.legacy).
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import socket

from temporalio.client import Client
from temporalio.worker import Worker

from . import TASK_QUEUE
from .agent import WorkspaceAgent
from .config import connect


def worker_name() -> str:
    return os.environ.get("POD_NAME") or socket.gethostname()


def legacy_enabled() -> bool:
    return os.environ.get("AGENT_ENABLE_LEGACY", "0") == "1"


def build_worker(client: Client, identity: str | None = None, legacy: bool | None = None, **kw) -> Worker:
    workflows: list = [WorkspaceAgent]
    activities: list = []
    if legacy if legacy is not None else legacy_enabled():
        from .legacy.llm import llm_call
        from .legacy.workflow import AgentWorkflow

        workflows.append(AgentWorkflow)
        activities.append(llm_call)
    return Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=workflows,
        activities=activities,
        identity=identity or f"{worker_name()}@agent-worker",
        **kw,
    )


async def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)  # no request lines (headers never logged anyway)
    logging.getLogger("openai").setLevel(logging.WARNING)
    client = await connect()
    worker = build_worker(client)
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for s in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(s, stop.set)
    logging.info("worker %s polling task queue %r (workflows: %s)", worker.config()["identity"], TASK_QUEUE,
                 [getattr(w, "__temporal_workflow_definition").name for w in worker.config()["workflows"]])
    async with worker:
        await stop.wait()
    logging.info("worker stopped")


if __name__ == "__main__":
    asyncio.run(main())
