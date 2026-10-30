"""End-to-end demo against the kind cluster.

Needs (see deploy.sh):  kubectl port-forward -n agent-demo svc/agent-api 18000:8000
                        kubectl port-forward -n <temporal-ns> svc/<frontend> 17233:7233
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import time
from collections import Counter

import httpx
from temporalio.api.enums.v1 import EventType
from temporalio.client import Client

from temporal_agent_demo.config import harness_plugin
from temporal_agent_demo.models import AgentResult

API = "http://localhost:18000"
TEMPORAL = "localhost:17233"
NS = "agent-demo"


def sh(*args: str) -> str:
    return subprocess.run(args, check=True, capture_output=True, text=True).stdout


def worker_pods() -> list[str]:
    out = sh("kubectl", "get", "pods", "-n", NS, "-l", "app=agent-worker", "--field-selector=status.phase=Running",
             "-o", "jsonpath={.items[*].metadata.name}")
    return out.split()


def in_pod(pod: str, cmd: str) -> str:
    return sh("kubectl", "exec", "-n", NS, pod, "--", "sh", "-c", cmd)


async def history_by_activity(client: Client, wid: str) -> list[tuple[str, str]]:
    """[(activity_type, identity_that_started_it)] in schedule order."""
    sched, out = {}, []
    async for ev in client.get_workflow_handle(wid).fetch_history_events():
        if ev.event_type == EventType.EVENT_TYPE_ACTIVITY_TASK_SCHEDULED:
            sched[ev.event_id] = ev.activity_task_scheduled_event_attributes.activity_type.name
        elif ev.event_type == EventType.EVENT_TYPE_ACTIVITY_TASK_STARTED:
            a = ev.activity_task_started_event_attributes
            out.append((sched[a.scheduled_event_id], a.identity.split("@")[0]))
    return out


async def wf_task_identities(client: Client, wid: str) -> list[str]:
    ids = []
    async for ev in client.get_workflow_handle(wid).fetch_history_events():
        if ev.event_type == EventType.EVENT_TYPE_WORKFLOW_TASK_STARTED:
            ids.append(ev.workflow_task_started_event_attributes.identity.split("@")[0])
    return ids


def post(http: httpx.Client, task: str) -> dict:
    r = http.post(f"{API}/executions", json={"task": task})
    r.raise_for_status()
    return r.json()


async def main() -> int:
    client = await Client.connect(TEMPORAL, plugins=[harness_plugin()])
    http = httpx.Client(timeout=30)
    report: dict = {"checks": {}}
    pods = worker_pods()
    print("worker pods:", pods)
    report["pods_before"] = pods

    # ---- 1. ten executions --------------------------------------------------------------
    execs = [post(http, f"Task {i:03d}: analyse something #{i}") for i in range(1, 11)]
    for e in execs:
        assert e["execution_id"] == e["workflow_id"] == e["session_id"] == e["workspace_id"] != e["run_id"]
    print(f"submitted {len(execs)} executions")
    results = await asyncio.gather(
        *(client.get_workflow_handle(e["workflow_id"], result_type=AgentResult).result() for e in execs)
    )
    rows = []
    for e, res in zip(execs, results):
        acts = await history_by_activity(client, e["workflow_id"])
        llm_pods = [p for t, p in acts if t == "llm_call"]
        all_pods = Counter(p for _, p in acts)
        rows.append({"execution_id": e["execution_id"], "llm_call_pods": llm_pods, "activity_pods": dict(all_pods),
                     "workflow_task_pods": dict(Counter(await wf_task_identities(client, e["workflow_id"]))),
                     "final": res.final, "session": res.session_status})
    report["ten_executions"] = rows
    used = set().union(*(r["activity_pods"].keys() for r in rows))
    report["checks"]["both_pods_used"] = len(used) >= 2
    print("pods that executed activities across the 10 runs:", sorted(used))

    # workspace contents, seen from BOTH pods (shared PVC)
    pod_a, pod_b = pods[0], pods[-1]
    ok_files = True
    for e in execs:
        wid = e["workspace_id"]
        listing_a = in_pod(pod_a, f"cd /data/workspaces/{wid} && find . -type f | sort")
        listing_b = in_pod(pod_b, f"cd /data/workspaces/{wid} && find . -type f | sort")
        expected = {"./session.json", "./input/task.md", "./output/plan.md", "./output/result.md",
                    "./temp/step-000.json", "./temp/step-001.json", "./temp/step-002.json"}
        files = set(listing_a.split())
        status = json.loads(in_pod(pod_b, f"cat /data/workspaces/{wid}/session.json"))["status"]
        task = in_pod(pod_a, f"cat /data/workspaces/{wid}/input/task.md")
        good = files == expected and listing_a == listing_b and status == "closed" and wid in in_pod(
            pod_b, f"cat /data/workspaces/{wid}/output/plan.md")
        ok_files &= good and task.startswith("Task ")
    report["checks"]["workspaces_complete_and_shared"] = ok_files
    report["workspace_dirs"] = in_pod(pod_a, "ls /data/workspaces").split()
    report["sample_tree"] = in_pod(pod_a, f"cd /data/workspaces && find {execs[0]['workspace_id']} | sort")

    # ---- 2. traversal probe ---------------------------------------------------------------
    victim = execs[0]["workspace_id"]
    probe = post(http, f"[probe-traversal][probe-target={victim}] try to escape")
    pres = await client.get_workflow_handle(probe["workflow_id"], result_type=AgentResult).result()
    step = next(s for s in pres.steps if any(c.id == "p1" for c in s.tool_calls))
    report["traversal_probe"] = [
        {"call": c.model_dump(), "ok": r.ok, "output": r.output} for c, r in zip(step.tool_calls, step.tool_results)
    ]
    pwned = in_pod(pod_a, f"ls /data/workspaces/{victim}/output").split()
    report["checks"]["traversal_rejected"] = (not any(r.ok for r in step.tool_results)) and "pwned.md" not in pwned
    print("traversal probe results:", [(c.args["path"], r.output[:60]) for c, r in zip(step.tool_calls, step.tool_results)])

    # ---- 3. crash recovery ------------------------------------------------------------------
    crash = post(http, "Task CRASH: long running job")
    h = client.get_workflow_handle(crash["workflow_id"], result_type=AgentResult)
    victim_pod = None
    deadline = time.time() + 60
    while time.time() < deadline:
        desc = await h.describe()
        done_llm = [t for t, _ in await history_by_activity(client, crash["workflow_id"]) if t == "llm_call"]
        pend = [p for p in desc.raw_description.pending_activities
                if p.activity_type.name == "llm_call" and p.last_worker_identity]
        if len(done_llm) >= 2 and pend:  # mid-run: >=1 llm step done, another in flight
            victim_pod = pend[0].last_worker_identity.split("@")[0]
            break
        await asyncio.sleep(0.3)
    assert victim_pod, "never caught an in-flight llm_call"
    print(f"killing pod {victim_pod} while it runs llm_call of {crash['execution_id']}")
    killed_at = time.time()
    if os.environ.get("CRASH_MODE", "delete") == "sigkill":
        # hard crash: SIGKILL the worker process from the kind node (no SIGTERM, no graceful shutdown)
        cid = sh("kubectl", "get", "pod", "-n", NS, victim_pod, "-o",
                 "jsonpath={.status.containerStatuses[0].containerID}").split("//")[-1]
        node = sh("kubectl", "get", "pod", "-n", NS, victim_pod, "-o", "jsonpath={.spec.nodeName}")
        pid = sh("docker", "exec", node, "crictl", "inspect", "-o", "go-template", "--template", "{{.info.pid}}", cid).strip()
        sh("docker", "exec", node, "kill", "-9", pid)
    else:
        sh("kubectl", "delete", "pod", "-n", NS, victim_pod, "--grace-period=0", "--force", "--wait=false")
    cres = await asyncio.wait_for(h.result(), 180)
    recovered_in = time.time() - killed_at
    acts = await history_by_activity(client, crash["workflow_id"])
    survivor = [p for p in pods if p != victim_pod][0]
    retries = []
    async for ev in h.fetch_history_events():
        if ev.event_type == EventType.EVENT_TYPE_ACTIVITY_TASK_STARTED:
            a = ev.activity_task_started_event_attributes
            if a.attempt > 1:
                retries.append({"attempt": a.attempt, "by": a.identity.split("@")[0], "last_failure": a.last_failure.message})
    files = in_pod(survivor, f"cd /data/workspaces/{crash['workspace_id']} && find . -type f | sort").split()
    steps_by_pod = [{"step": s.step, "llm_worker": s.llm_worker} for s in cres.steps]
    report["crash"] = {
        "execution_id": crash["execution_id"], "killed_pod": victim_pod, "survivor": survivor,
        "activity_starts": acts, "retries": retries, "mode": os.environ.get("CRASH_MODE", "delete"), "steps": steps_by_pod, "final": cres.final, "session": cres.session_status,
        "workspace_files_seen_by_survivor": files, "seconds_to_complete_after_kill": round(recovered_in, 1),
    }
    pods_used = {p for _, p in acts}
    report["checks"]["crash_recovered_same_workspace"] = (
        victim_pod in pods_used and survivor in pods_used and cres.session_status == "closed"
        and "./output/result.md" in files and "./input/task.md" in files
    )
    report["pods_after"] = worker_pods()
    print("crash run activity starts:", acts)

    # ---- 4. mock-llm request log -----------------------------------------------------------
    logs = [json.loads(l) for l in sh("kubectl", "logs", "-n", NS, "deploy/mock-llm").splitlines() if l.startswith("{")]
    chat = [l for l in logs if l.get("event") == "chat.completions"]
    faulted = {l["req_hash"] for l in chat if l["fault"]}
    ok_hashes = {l["req_hash"] for l in chat if l["status"] == 200}
    report["mock_llm"] = {
        "chat_requests": len(chat),
        "by_status": dict(Counter(l["status"] for l in chat)),
        "faults_injected": sum(1 for l in chat if l["fault"]),
        "faulted_requests_later_succeeded": len(faulted & ok_hashes),
        "faulted_requests_never_succeeded": sorted(faulted - ok_hashes),
        "requests_by_client_pod": dict(Counter(l["client"] for l in chat)),
    }
    report["checks"]["llm_faults_retried_ok"] = bool(faulted) and not (faulted - ok_hashes)
    print("mock-llm:", json.dumps(report["mock_llm"]))

    json.dump(report, open(os.environ.get("E2E_REPORT", "e2e-out/report.json"), "w"), indent=2, default=str)
    print(json.dumps(report["checks"], indent=2))
    return 0 if all(report["checks"].values()) else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
