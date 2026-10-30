"""Real-LLM e2e: a few short executions through Temporal -> agent-worker pods -> host llm_gateway -> Grok.

Needs the same port-forwards as e2e.py (API :18000, Temporal :17233) and scripts/start-gateway.sh running.
Writes E2E_REPORT (default e2e-out/report-real.json). Keep tasks tiny: they spend SuperGrok quota.
"""

import asyncio
import json
import os
import sys
from collections import Counter

import httpx
from temporalio.client import Client

sys.path.insert(0, os.path.dirname(__file__))
from e2e import API, TEMPORAL, history_by_activity, in_pod, post, worker_pods  # noqa: E402

from temporal_agent_demo.config import harness_plugin
from temporal_agent_demo.models import AgentResult

TASKS = [
    "Write a haiku about Kubernetes pods to output/haiku.md.",
    "Create output/primes.txt listing the first 10 prime numbers, one per line, then read it back to verify.",
    "Write a 3-bullet checklist for safely restarting a Temporal worker to output/checklist.md, then list the files in output/.",
]
TRAVERSAL = ("Security test: call write_file with path '../escape.txt' and content 'x' exactly once. "
             "Report whether it succeeded. Then write output/result.md describing the tool result.")


async def main() -> int:
    client = await Client.connect(TEMPORAL, plugins=[harness_plugin()])
    http = httpx.Client(timeout=30)
    pods = worker_pods()
    execs = [post(http, t) for t in TASKS + [TRAVERSAL]]
    results = await asyncio.gather(*(client.get_workflow_handle(e["workflow_id"], result_type=AgentResult).result()
                                     for e in execs))
    rows, checks = [], {}
    for e, task, res in zip(execs, TASKS + [TRAVERSAL], results):
        wid = e["workflow_id"]
        acts = await history_by_activity(client, wid)
        files = in_pod(pods[0], f"cd /data/workspaces/{wid} && find . -type f | sort").split()
        tools = [(c.name, c.args.get("path"), r.ok, r.output[:160]) for s in res.steps
                 for c, r in zip(s.tool_calls, s.tool_results)]
        outputs = {f: in_pod(pods[-1], f"cat /data/workspaces/{wid}/{f}")[:400] for f in files if f.startswith("./output/")}
        rows.append({"execution_id": wid, "task": task, "session": res.session_status, "steps": len(res.steps),
                     "tool_calls": tools, "activity_pods": dict(Counter(p for _, p in acts)),
                     "files": files, "outputs": outputs, "final": res.final[:500]})
    real = rows[:3]
    checks["all_sessions_closed"] = all(r["session"] == "closed" for r in rows)
    checks["grok_used_tools"] = all(any(t[0] == "write_file" and t[2] for t in r["tool_calls"]) for r in real)
    checks["outputs_in_own_workspace"] = all(r["outputs"] for r in real)
    trav = rows[3]
    escaped = in_pod(pods[0], "ls /data/workspaces/escape.txt /data/escape.txt 2>/dev/null || true").strip()
    checks["traversal_rejected"] = (any(t[1] and ".." in t[1] and not t[2] and "PermissionError" in t[3]
                                        for t in trav["tool_calls"]) and not escaped)
    checks["pods_used"] = sorted(set().union(*(r["activity_pods"] for r in rows)))
    report = {"checks": checks, "executions": rows}
    out = os.environ.get("E2E_REPORT", "e2e-out/report-real.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(report, open(out, "w"), indent=2, default=str)
    print(json.dumps(report, indent=2, default=str))
    return 0 if all(v for k, v in checks.items() if k != "pods_used") else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
