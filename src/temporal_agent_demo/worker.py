"""Agent worker: every replica runs this identical process on task queue `agent`.

    python -m temporal_agent_demo.worker

Registers the WorkspaceAgent workflow. Its activities come from the client plugins (config.py):
``invoke_model_activity`` (OpenAIAgentsPlugin) and the harness + WorkspaceDisk ``vfs.*`` activities
(AgentHarnessPlugin).
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import socket

from temporalio.client import Client
from temporalio.worker import Worker

from .config import connect
from .shared import TASK_QUEUE
from .workflows import WorkspaceAgent


def worker_name() -> str:
    return os.environ.get("POD_NAME") or socket.gethostname()


def build_worker(client: Client, identity: str | None = None, **kw) -> Worker:
    return Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[WorkspaceAgent],
        identity=identity or f"{worker_name()}@agent-worker",
        **kw,
    )


async def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)  # no request lines
    logging.getLogger("openai").setLevel(logging.WARNING)
    client = await connect()
    worker = build_worker(client)
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for s in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(s, stop.set)
    logging.info("worker %s polling task queue %r", worker.config()["identity"], TASK_QUEUE)
    async with worker:
        await stop.wait()
    logging.info("worker stopped")


if __name__ == "__main__":
    asyncio.run(main())
