"""Start a ClaudeTaskWorkflow (Claude Agent SDK activity) and print its result.

    python -m temporal_agent_demo.claude_starter "Summarize notes.md into output/summary.md" \
        --skill haiku=skills/haiku/SKILL.md --tool word_count --model grok-4.6

--skill name=PATH reads a local SKILL.md; --system TEXT sets a custom system prompt
(--append-system TEXT appends to Claude Code's preset instead).
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from .config import connect
from .shared import (CLAUDE_WORKFLOW, DEFAULT_MODEL, TASK_QUEUE, ClaudeAgentInput, ClaudeAgentResult,
                     SystemPromptPreset, new_execution_id)


def build_input(args) -> ClaudeAgentInput:
    skills = {}
    for s in args.skill:
        name, _, path = s.partition("=")
        skills[name] = Path(path).read_text(encoding="utf-8")
    sp = args.system if args.system else (SystemPromptPreset(append=args.append_system) if args.append_system else None)
    return ClaudeAgentInput(execution_id=new_execution_id(), prompt=args.prompt, system_prompt=sp,
                            skills=skills, tools=args.tool, model=args.model, max_turns=args.max_turns)


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("prompt")
    ap.add_argument("--skill", action="append", default=[], metavar="NAME=PATH")
    ap.add_argument("--tool", action="append", default=[])
    ap.add_argument("--system", default=None)
    ap.add_argument("--append-system", default=None)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--max-turns", type=int, default=10)
    args = ap.parse_args()
    inp = build_input(args)
    client = await connect()
    res = await client.execute_workflow(CLAUDE_WORKFLOW, inp, id=inp.execution_id, task_queue=TASK_QUEUE,
                                        result_type=ClaudeAgentResult)
    print(json.dumps(res.model_dump(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())
