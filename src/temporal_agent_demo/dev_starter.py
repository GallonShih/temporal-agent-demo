"""Start a DevTaskWorkflow (plan -> implement -> test -> fix loop -> review) and print its result.

    python -m temporal_agent_demo.dev_starter "Write roman.py with to_roman/from_roman + pytest tests" \
        --acceptance tests/test_acceptance.py=./hidden_test.py --model grok-4.6

Progress while running:  temporal workflow query --workflow-id <id> --type progress
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from .config import connect
from .shared import DEFAULT_MODEL, DEV_WORKFLOW, TASK_QUEUE, DevTaskInput, DevTaskResult, new_execution_id


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("task")
    ap.add_argument("--acceptance", action="append", default=[], metavar="WS_PATH=LOCAL_FILE",
                    help="hidden test file written into the workspace after implement")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--max-fix", type=int, default=3)
    ap.add_argument("--max-turns", type=int, default=16)
    ap.add_argument("--id", default=None, help="execution id (default: generated)")
    args = ap.parse_args()
    acc = {}
    for a in args.acceptance:
        ws, _, local = a.partition("=")
        acc[ws] = Path(local).read_text(encoding="utf-8")
    inp = DevTaskInput(execution_id=args.id or new_execution_id(), task=args.task, model=args.model,
                       max_fix_iterations=args.max_fix, max_turns_per_step=args.max_turns, acceptance_tests=acc)
    client = await connect()
    print(f"workflow id: {inp.execution_id}", flush=True)
    res = await client.execute_workflow(DEV_WORKFLOW, inp, id=inp.execution_id, task_queue=TASK_QUEUE,
                                        result_type=DevTaskResult)
    print(json.dumps(res.model_dump(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())
