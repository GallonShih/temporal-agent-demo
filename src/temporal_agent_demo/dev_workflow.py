"""DevTaskWorkflow: plan -> implement -> test -> fix (<= N, resumed session) -> review.

Long-term memory lives in the workspace (PLAN.md, NOTES.md, REVIEW.md); each fresh step only gets
a short handoff. The fix loop continues the *implement* session via Claude Code `resume`, so the
model keeps its own context of what it wrote; transcripts live on the PVC so any worker can resume.
Tests are judged by a deterministic activity (run_workspace_tests), not by the model.
"""

from __future__ import annotations

from datetime import timedelta

from temporalio import workflow
from temporalio.common import RetryPolicy

with workflow.unsafe.imports_passed_through():
    from .shared import (ClaudeAgentInput, ClaudeAgentResult, DevTaskInput, DevTaskResult, StepRecord,
                         SystemPromptPreset, PytestRun)

RETRY = RetryPolicy(maximum_attempts=2, non_retryable_error_types=["ValueError", "ValidationError"])
FILE_TOOLS = ["Read", "Write", "Edit", "Glob", "Grep"]
SYSTEM_APPEND = (
    "You work inside a sandboxed workspace (your cwd). Only touch files there, use relative paths. "
    "No shell is available. Workspace files are the shared memory between steps: PLAN.md is the plan, "
    "NOTES.md is the handoff log - append a short dated section for your step (what you did, open issues). "
    "Finish with a 2-4 line summary of what you did.")


def _tail(s: str, n: int = 3000) -> str:
    return s if len(s) <= n else "...\n" + s[-n:]


@workflow.defn(name="DevTaskWorkflow")
class DevTaskWorkflow:
    def __init__(self) -> None:
        self._r: DevTaskResult | None = None

    @workflow.query
    def progress(self) -> DevTaskResult | None:
        return self._snapshot()

    def _snapshot(self) -> DevTaskResult | None:
        # fresh model: the converter may drop fields mutated in place (not in model_fields_set)
        return None if self._r is None else DevTaskResult.model_validate(self._r.model_dump())

    async def _claude(self, inp: DevTaskInput, step: str, prompt: str, *, tools: list[str] | None = None,
                      builtins: list[str] | None = None, resume: str | None = None) -> ClaudeAgentResult:
        rec = StepRecord(name=step, status="running", resumed_from=resume)
        self._r.steps.append(rec)
        res: ClaudeAgentResult = await workflow.execute_activity(
            "run_claude_agent",
            ClaudeAgentInput(execution_id=inp.execution_id, prompt=prompt, step=step,
                             system_prompt=SystemPromptPreset(append=SYSTEM_APPEND),
                             tools=tools or [], builtin_tools=builtins or FILE_TOOLS, model=inp.model,
                             max_turns=inp.max_turns_per_step, persist_session=True, resume_session_id=resume),
            result_type=ClaudeAgentResult,
            start_to_close_timeout=timedelta(minutes=20), heartbeat_timeout=timedelta(seconds=60),
            retry_policy=RETRY)
        rec.status = "error" if res.is_error else "ok"
        rec.session_id, rec.worker, rec.attempt = res.session_id, res.worker, res.attempt
        rec.turns, rec.usage, rec.tool_uses, rec.denied = res.num_turns, res.usage, res.tool_uses, res.denied
        rec.summary = res.result[-800:]
        self._r.files = res.files
        return res

    async def _test(self, inp: DevTaskInput, step: str) -> PytestRun:
        rec = StepRecord(name=step, status="running")
        self._r.steps.append(rec)
        t: PytestRun = await workflow.execute_activity(
            "run_workspace_tests", inp.execution_id, result_type=PytestRun,
            start_to_close_timeout=timedelta(minutes=5), retry_policy=RETRY)
        rec.status, rec.worker = ("ok" if t.passed else "failed"), t.worker
        rec.summary = f"exit_code={t.exit_code}\n" + _tail(t.output, 800)
        return t

    @workflow.run
    async def run(self, inp: DevTaskInput) -> DevTaskResult:
        if inp.execution_id != workflow.info().workflow_id:
            raise ValueError("execution_id must equal the workflow id")
        self._r = DevTaskResult(execution_id=inp.execution_id, status="running")
        task = inp.task

        plan = await self._claude(inp, "plan", (
            f"TASK:\n{task}\n\nStep: PLAN. Do not write code yet. Write PLAN.md: files to create "
            "(Python modules + pytest tests under tests/), function signatures, edge cases, and a numbered "
            "checklist. Create NOTES.md with a '## plan' section."))

        impl = await self._claude(inp, "implement", (
            f"TASK:\n{task}\n\nStep: IMPLEMENT. Read PLAN.md and NOTES.md, then implement the code and the "
            "pytest tests exactly as planned (tests import modules from the workspace root). "
            "Append a '## implement' section to NOTES.md.\n\nPrevious step summary:\n" + plan.result[-1200:]),
            tools=["run_pytest"])

        if inp.acceptance_tests:
            await workflow.execute_activity("write_workspace_files", args=[inp.execution_id, inp.acceptance_tests],
                                            start_to_close_timeout=timedelta(minutes=1), retry_policy=RETRY)

        session = impl.session_id
        t = await self._test(inp, "test-0")
        i = 0
        while not t.passed and i < inp.max_fix_iterations:
            i += 1
            fix = await self._claude(inp, f"fix-{i}", (
                f"Step: FIX (iteration {i}/{inp.max_fix_iterations}). The test suite (python -m pytest -q) "
                f"failed with exit code {t.exit_code}:\n\n{_tail(t.output)}\n\nFix the code (not the tests, "
                "unless a test you wrote is clearly wrong). Never modify, skip, deselect or work around tests you did not write (e.g. acceptance tests) - no conftest/pytest.ini tricks; if a test cannot run, report it. Use the run_pytest tool "
                f"to verify. Append a '## fix-{i}' section to NOTES.md."),
                tools=["run_pytest"], resume=session)
            session = fix.session_id or session  # continue the newest point of the same conversation
            t = await self._test(inp, f"test-{i}")
        self._r.fix_iterations = i

        await self._claude(inp, "review", (
            f"TASK:\n{task}\n\nStep: REVIEW (fresh reviewer, read-only except REVIEW.md). Read PLAN.md, NOTES.md "
            f"and the code/tests. Final test run: exit_code={t.exit_code}, passed={t.passed}.\n"
            f"{_tail(t.output, 1500)}\n\nWrite REVIEW.md: verdict (APPROVE / CHANGES_REQUESTED), correctness "
            "issues, test-coverage gaps, code quality, and whether PLAN.md's checklist is complete."),
            builtins=["Read", "Glob", "Grep", "Write"])

        self._r.status = "passed" if t.passed else "failed"
        return self._snapshot()
