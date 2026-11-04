"""ClaudeTaskWorkflow: run one Claude Agent SDK task (activity run_claude_agent) in its workspace.

workflow_id == execution_id == workspace_id, same as WorkspaceAgent. The activity heartbeats;
a lost worker is detected within the heartbeat timeout and the activity retried on another worker
(a retry re-runs the agent from the start; file writes are idempotent replaces).
"""

from __future__ import annotations

from datetime import timedelta

from temporalio import workflow
from temporalio.common import RetryPolicy

with workflow.unsafe.imports_passed_through():
    from .activities import run_claude_agent
    from .shared import CLAUDE_WORKFLOW, ClaudeAgentInput, ClaudeAgentResult


@workflow.defn(name=CLAUDE_WORKFLOW)
class ClaudeTaskWorkflow:
    @workflow.run
    async def run(self, inp: ClaudeAgentInput) -> ClaudeAgentResult:
        if inp.execution_id != workflow.info().workflow_id:
            raise ValueError("execution_id must equal the workflow id")
        return await workflow.execute_activity(
            run_claude_agent, inp,
            start_to_close_timeout=timedelta(minutes=20),
            heartbeat_timeout=timedelta(seconds=60),
            retry_policy=RetryPolicy(maximum_attempts=2, non_retryable_error_types=["ValueError", "ValidationError"]),
        )
