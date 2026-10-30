"""Client construction shared by worker, API, tests and scripts.

Plugin stack (order matters, harness LAST, as in the harness examples):
  1. OpenAIAgentsPlugin   model activities (invoke_model_activity) + OpenAI-aware payload converter.
                          Its model provider is an OpenAI *chat-completions* client pointed at
                          OPENAI_BASE_URL (host llm_gateway; tests: mock_llm_server), SDK retries off,
                          so Temporal activity retries own 429/5xx.
  2. AgentHarnessPlugin   harness activities, WorkspaceDisk vfs.* activities, large-payload offload.
"""

from __future__ import annotations

import os
from datetime import timedelta
from pathlib import Path

os.environ.setdefault("OPENAI_AGENTS_DISABLE_TRACING", "1")  # no trace export to OpenAI

from temporalio.client import Client
from temporalio.common import RetryPolicy
from temporal_agent_harness.plugin import AgentHarnessPlugin
from temporal_agent_harness.utils.large_payload import local_payload_storage

from .workspace_fs import WorkspaceDisk


def harness_plugin() -> AgentHarnessPlugin:
    # Large-payload offload must be readable by EVERY worker -> put it on the shared PVC
    # (in prod: s3_payload_storage). Separate dir from workspaces so no workspace id can reach it.
    payload_dir = Path(os.environ.get("PAYLOAD_DIR", "/data/payloads"))
    return AgentHarnessPlugin(
        filesystems=[WorkspaceDisk],
        large_payload_offload=local_payload_storage(base_dir=payload_dir),
    )


# Retry posture of every model call (the harness maps 408/409/429/5xx -> retryable, honouring
# Retry-After; other 4xx -> non-retryable).
MODEL_RETRY = RetryPolicy(
    initial_interval=timedelta(seconds=1), backoff_coefficient=2.0,
    maximum_interval=timedelta(seconds=20), maximum_attempts=6,
)


def model_plugin(base_url: str | None = None, api_key: str | None = None):
    from agents.models.openai_provider import OpenAIProvider
    from openai import AsyncOpenAI
    from temporal_agent_harness.ai_sdks.openai_agents import ModelActivityParameters, OpenAIAgentsPlugin

    client = AsyncOpenAI(
        base_url=base_url or os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:8787/v1"),
        # .strip(): k8s Secrets created from files often carry a trailing newline
        api_key=(api_key or os.environ.get("OPENAI_API_KEY", "") or "unset").strip(),
        max_retries=0,  # Temporal retries, not the SDK
        timeout=float(os.environ.get("OPENAI_TIMEOUT_S", "120")),
        default_headers={"X-Client-Id": os.environ.get("POD_NAME", "local")},
    )
    return OpenAIAgentsPlugin(
        model_params=ModelActivityParameters(
            start_to_close_timeout=timedelta(minutes=3),
            heartbeat_timeout=timedelta(seconds=10),  # auto-heartbeat => fast failover on pod death
            retry_policy=MODEL_RETRY,
        ),
        # use_responses=False -> POST /v1/chat/completions (what xAI + our gateway/mock speak)
        model_provider=OpenAIProvider(openai_client=client, use_responses=False),
    )


def plugins(base_url: str | None = None, api_key: str | None = None) -> list:
    return [model_plugin(base_url, api_key), harness_plugin()]


async def connect(address: str | None = None, namespace: str | None = None, **model_kw) -> Client:
    return await Client.connect(
        address or os.environ.get("TEMPORAL_ADDRESS", "localhost:7233"),
        namespace=namespace or os.environ.get("TEMPORAL_NAMESPACE", "default"),
        plugins=plugins(**model_kw),
    )
