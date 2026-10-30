"""End-to-end check of the WorkspaceAgent deployment on the kind cluster.

Needs (see deploy.sh):  kubectl port-forward -n agent-demo svc/agent-api 18000:8000
                        kubectl port-forward -n <temporal-ns> svc/<frontend> 17233:7233
and the host llm_gateway running; model `grok-mock` is always the scripted mock. Start it with mock latency
so the SIGKILL lands mid-activity (faults optional):  MOCK_LATENCY_S=2 MOCK_FAULT_EVERY=4 scripts/start-gateway.sh

    uv run python scripts/e2e.py                 # mock suite: 10 runs, traversal probe, SIGKILL a worker
    uv run python scripts/e2e.py --real 3        # + 3 short real tasks on grok-4.6 (spends quota)
    uv run python scripts/e2e.py --skip-mock --real 1

Env: E2E_API (default http://localhost:18000), E2E_TEMPORAL (default localhost:17233).
Writes e2e-out/report.json.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import httpx
from temporalio.api.enums.v1 import EventType
from temporalio.client import Client

from temporal_agent_demo.config import harness_plugin
from temporal_agent_demo.shared import TaskResult

API = os.environ.get("E2E_API", "http://localhost:18000")
TEMPORAL = os.environ.get("E2E_TEMPORAL", "localhost:17233")
NS = "agent-demo"
MODEL_ACTIVITY = "invoke_model_activity"
EXPECTED = {"./session.json", "./input/task.md", "./output/plan.md", "./output/result.md", "./temp/steps.json"}
REAL_TASKS = [
    "Write a three-line haiku about Temporal workflows to /workspace/output/haiku.md, then read it back.",
    "List the files in /workspace, then write a one-paragraph summary of what you found to /workspace/output/summary.md.",
    "Write the first 10 Fibonacci numbers, comma separated, to /workspace/output/fib.txt.",
]


def sh(*args: str) -> str:
    return subprocess.run(args, check=True, capture_output=True, text=True).stdout


def worker_pods() -> list[str]:
    out = sh("kubectl", "get", "pods", "-n", NS, "-l", "app=agent-worker", "--field-selector=status.phase=Running",
             "-o", "jsonpath={.items[*].metadata.name}")
    return out.split()


def in_pod(pod: str, cmd: str) -> str:
    return sh("kubectl", "exec", "-n", NS, pod, "--", "sh", "-c", cmd)


def post(http: httpx.Client, task: str, model: str) -> dict:
    r = http.post(f"{API}/executions", json={"task": task, "model": model})
    r.raise_for_status()
    return r.json()


async def result(client: Client, wid: str, timeout: float = 300) -> TaskResult:
    h = client.get_workflow_handle(wid)
    await asyncio.wait_for(h.result(), timeout)
    return await h.query("result", result_type=TaskResult)


async def activity_starts(client: Client, wid: str) -> list[dict]:
    """Every activity start in schedule order: type, worker pod, attempt."""
    sched, out = {}, []
    async for ev in client.get_workflow_handle(wid).fetch_history_events():
        if ev.event_type == EventType.EVENT_TYPE_ACTIVITY_TASK_SCHEDULED:
            sched[ev.event_id] = ev.activity_task_scheduled_event_attributes.activity_type.name
        elif ev.event_type == EventType.EVENT_TYPE_ACTIVITY_TASK_STARTED:
            a = ev.activity_task_started_event_attributes
            out.append({"type": sched[a.scheduled_event_id], "pod": a.identity.split("@")[0], "attempt": a.attempt})
    return out


async def mock_suite(client: Client, http: httpx.Client, report: dict) -> None:
    checks = report["checks"]
    pods = worker_pods()
    report["pods_before"] = pods
    print("worker pods:", pods)

    # ---- 1. ten executions ---------------------------------------------------------------
    execs = [post(http, f"Task {i:03d}: analyse something #{i}", "grok-mock") for i in range(1, 11)]
    for e in execs:
        assert e["execution_id"] == e["workflow_id"] == e["session_id"] == e["workspace_id"] != e["run_id"]
    results = await asyncio.gather(*(result(client, e["workflow_id"]) for e in execs))
    rows, used, retries = [], set(), 0
    for e, res in zip(execs, results):
        acts = await activity_starts(client, e["workflow_id"])
        used |= {a["pod"] for a in acts}
        retries += sum(1 for a in acts if a["type"] == MODEL_ACTIVITY and a["attempt"] > 1)
        rows.append({"execution_id": e["execution_id"], "outcome": res.outcome, "session": res.session_status,
                     "model_calls": res.model_calls, "activity_pods": dict(Counter(a["pod"] for a in acts))})
    report["ten_executions"] = rows
    report["model_activity_retries"] = retries
    checks["ten_completed_and_closed"] = all(r.outcome == "completed" and r.session_status == "closed" for r in results)
    checks["both_pods_used"] = len(used) >= 2

    pod_a, pod_b = pods[0], pods[-1]
    ok = True
    for e in execs:
        wid = e["workspace_id"]
        la = in_pod(pod_a, f"cd /data/workspaces/{wid} && find . -type f | sort")
        lb = in_pod(pod_b, f"cd /data/workspaces/{wid} && find . -type f | sort")
        status = json.loads(in_pod(pod_b, f"cat /data/workspaces/{wid}/session.json"))["status"]
        ok &= EXPECTED <= set(la.split()) and la == lb and status == "closed"
    checks["workspaces_complete_and_shared"] = ok
    report["sample_tree"] = in_pod(pod_a, f"cd /data/workspaces && find {execs[0]['workspace_id']} | sort")

    # ---- 2. traversal probe ----------------------------------------------------------------
    victim = execs[0]["workspace_id"]
    before = in_pod(pod_a, f"cd /data/workspaces/{victim} && find . -type f | sort")
    probe = post(http, f"[probe-traversal][probe-target={victim}] try to escape", "grok-mock")
    pres = await result(client, probe["workflow_id"])
    psteps = [s for s in pres.tool_steps if s.call_id.startswith("p")]
    after = in_pod(pod_a, f"cd /data/workspaces/{victim} && find . -type f | sort")
    report["traversal_probe"] = [s.model_dump() for s in psteps]
    checks["traversal_rejected"] = (len(psteps) == 6 and not any(s.ok for s in psteps)
                                    and "pwned" not in after and before == after)

    # ---- 3. SIGKILL a worker while it runs a model activity --------------------------------
    crash = post(http, "Task CRASH: long running job", "grok-mock")
    h = client.get_workflow_handle(crash["workflow_id"])
    victim_pod, deadline = None, time.time() + 90
    while time.time() < deadline and not victim_pod:
        desc = await h.describe()
        pend = [p for p in desc.raw_description.pending_activities if p.last_worker_identity]
        if pend:
            victim_pod = pend[0].last_worker_identity.split("@")[0]
        else:
            await asyncio.sleep(0.2)
    assert victim_pod, "never caught an in-flight activity"
    print(f"SIGKILL worker process in {victim_pod} ({crash['execution_id']})")
    cid = sh("kubectl", "get", "pod", "-n", NS, victim_pod, "-o",
             "jsonpath={.status.containerStatuses[0].containerID}").split("//")[-1]
    node = sh("kubectl", "get", "pod", "-n", NS, victim_pod, "-o", "jsonpath={.spec.nodeName}")
    pid = sh("docker", "exec", node, "crictl", "inspect", "-o", "go-template", "--template", "{{.info.pid}}", cid).strip()
    killed_at = time.time()
    sh("docker", "exec", node, "kill", "-9", pid)
    cres = await result(client, crash["workflow_id"], 300)
    acts = await activity_starts(client, crash["workflow_id"])
    survivor = next(p for p in pods if p != victim_pod)
    files = in_pod(survivor, f"cd /data/workspaces/{crash['workspace_id']} && find . -type f | sort").split()
    report["crash"] = {"execution_id": crash["execution_id"], "killed_pod": victim_pod, "survivor": survivor,
                       "activity_starts": acts, "outcome": cres.outcome, "session": cres.session_status,
                       "files_seen_by_survivor": files,
                       "seconds_to_complete_after_kill": round(time.time() - killed_at, 1)}
    checks["crash_recovered_same_workspace"] = (cres.outcome == "completed" and cres.session_status == "closed"
                                                and survivor in {a["pod"] for a in acts} and EXPECTED <= set(files))
    report["pods_after"] = worker_pods()


async def real_suite(client: Client, http: httpx.Client, report: dict, n: int, model: str) -> None:
    rows = []
    for task in (REAL_TASKS * n)[:n]:
        e = post(http, task, model)
        res = await result(client, e["workflow_id"], 600)
        rows.append({"execution_id": e["execution_id"], "model": res.model, "outcome": res.outcome,
                     "session": res.session_status, "model_calls": res.model_calls,
                     "tools": [s.name for s in res.tool_steps], "final": res.final[:500]})
        print(f"real {e['execution_id']}: {res.outcome} ({len(res.tool_steps)} tool calls)")
    report["real"] = rows
    report["checks"]["real_runs_completed"] = all(r["outcome"] == "completed" and r["session"] == "closed" for r in rows)


async def main() -> int:
    ap = argparse.ArgumentParser(description="e2e for the WorkspaceAgent deployment")
    ap.add_argument("--real", type=int, default=0, help="number of short real-model executions")
    ap.add_argument("--real-model", default="grok-4.6")
    ap.add_argument("--skip-mock", action="store_true")
    args = ap.parse_args()

    client = await Client.connect(TEMPORAL, plugins=[harness_plugin()])
    http = httpx.Client(timeout=60)
    report: dict = {"checks": {}}
    if not args.skip_mock:
        await mock_suite(client, http, report)
    if args.real:
        await real_suite(client, http, report, args.real, args.real_model)

    Path("e2e-out").mkdir(exist_ok=True)
    Path("e2e-out/report.json").write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report["checks"], indent=2))
    return 0 if all(report["checks"].values()) else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
