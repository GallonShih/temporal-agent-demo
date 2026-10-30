# temporal-agent-demo

2 identical Temporal agent workers (k8s Deployment `agent-worker`, replicas=2) on one task queue `agent`,
one shared PVC `agent-workspaces`, and one isolated workspace per execution. The model is Grok (via a host-side
LLM gateway) or a scripted mock (`grok-mock`).

## Runtime
* **temporal-agent-harness 0.8.0** session runtime: `WorkspaceAgent` is an `@agent.defn` agent with a VFS mount
  `/workspace` -> `WorkspaceDisk`, the Code Mode tool `run_code`, and the tools write_file / read_file / list_files.
* `AgentHarnessPlugin(filesystems=[WorkspaceDisk], large_payload_offload=local_payload_storage(/data/payloads))`
  provides the Pydantic converter, payload offload on the shared PVC and the `vfs.workspace-disk.*` activities;
  `OpenAIAgentsPlugin` (chat completions -> `OPENAI_BASE_URL`) provides `invoke_model_activity`.
  There are no hand-written activities.
* `POST /executions` (or `starter.py`) starts the session with update-with-start
  (`AgentClient.start_and_submit_message`); the session closes itself after its task.
* `execution_id == workflow_id == session_id == workspace_id`, fresh per execution.

## Layout
```
src/temporal_agent_demo/
  shared.py        constants, ExecutionContext, models (WorkspaceSession, TaskMessage, ToolStep, TaskResult)
  workflows.py     WorkspaceAgent (@agent.defn): bind workspace -> run task -> write outputs -> close session
  tools.py         workflow-side tools (write_file / read_file / list_files) confined to /workspace
                   (named tools.py, not activities.py: they are @agent.tool_defn tools; their file ops run as
                   the harness's vfs.workspace-disk.* activities)
  workspace_fs.py  WorkspaceDisk: per-workspace root, path confinement, atomic idempotent writes
  worker.py        worker process (task queue `agent`)
  starter.py       CLI: start one execution and print its result
  api.py           FastAPI: POST /executions, GET /executions/{id}, POST /executions/{id}/close, /healthz
  config.py        Temporal client, AgentHarnessPlugin + OpenAIAgentsPlugin (tracing disabled)
llm_gateway/       host-side OpenAI-compatible gateway: real Grok (Hermes xAI OAuth) or the mock
mock_llm_server/   scripted OpenAI-compatible mock (used in-process by the gateway and by tests)
tests/             pytest (local Temporal dev server via temporalio.testing; no k8s)
k8s/               namespace, PVC, llm-api-key Secret, worker Deployment (x2), API
scripts/           start-gateway.sh, deploy.sh (build -> kind load -> apply), e2e.py
```

## Run
```bash
uv sync
uv run pytest -q tests                                  # unit + integration (downloads temporal dev server once)

scripts/start-gateway.sh                                # host gateway on :8787 (GATEWAY_MODE=mock for the mock)
TEMPORAL_ADDRESS=temporal-frontend.temporal.svc.cluster.local:7233 scripts/deploy.sh
kubectl port-forward -n agent-demo svc/agent-api 18000:8000 &
kubectl port-forward -n temporal svc/temporal-frontend 17233:7233 &

curl -XPOST localhost:18000/executions -H 'content-type: application/json' -d '{"task":"hello","model":"grok-4.6"}'
TEMPORAL_ADDRESS=localhost:17233 uv run python -m temporal_agent_demo.starter "Write a haiku to /workspace/output/haiku.md"
MOCK_LATENCY_S=2 MOCK_FAULT_EVERY=4 scripts/start-gateway.sh   # mock latency + faults for the e2e
uv run python scripts/e2e.py [--real 2]                  # see the script docstring; writes e2e-out/report.json
```

## LLM: host-side gateway → real Grok via SuperGrok (Hermes xAI OAuth)

The LLM endpoint now lives **outside kind**, on the box host, and pods reach it through the kind docker
bridge gateway (`docker network inspect kind` → `172.18.0.1`).

```
agent-worker pod --(OPENAI_BASE_URL=http://172.18.0.1:8787/v1, Bearer <local gateway key>)--> llm_gateway (host, 0.0.0.0:8787)
    GATEWAY_MODE=mock       -> scripted mock (mock_llm_server, in-process)
    GATEWAY_MODE=xai-oauth  -> https://api.x.ai/v1/chat/completions with the SuperGrok OAuth bearer from Hermes
```

* `llm_gateway/server.py`: FastAPI with `POST /v1/chat/completions` (tools / tool_calls) and `GET /v1/models`
  (the upstream list plus `grok-mock`), and also `/healthz` and `/v1/stats`.
* Upstream credential: it calls Hermes Agent's own helper `hermes_cli.auth_xai.resolve_xai_oauth_runtime_credentials()`
  **per request**, so Hermes refreshes the token when it nears expiry. On an upstream 401 it force-refreshes once and retries.
  The token is never logged, returned, written to disk or put in k8s. Only Hermes's `~/.hermes/auth.json` holds it.
* Workers authenticate to the gateway with a **local** key: `llm_gateway/.gateway-key`, chmod 600, generated once.
  `start-gateway.sh` syncs it into the Secret `agent-demo/llm-api-key`.
* The model `grok-mock` is always served by the mock, even in xai-oauth mode. The default model is `grok-4.6`.
* Errors: upstream 429/5xx (with `Retry-After`) are passed through, and the worker turns them into retryable
  activity errors. Other 4xx are non-retryable. Credential failures become 503. `stream=true` gets a 400.
* Log: `llm_gateway/gateway.log`, one JSON line per call with model, route, status, ms, tool-call names and token usage.

```bash
hermes auth list xai-oauth                        # needs a logged-in xai-oauth credential
scripts/start-gateway.sh                          # xai-oauth on :8787; prints PID (llm_gateway/gateway.pid)
GATEWAY_MODE=mock scripts/start-gateway.sh        # switch to the mock (restarts the gateway)
TEMPORAL_ADDRESS=temporal-frontend.temporal.svc.cluster.local:7233 scripts/deploy.sh
#   -> workers get OPENAI_BASE_URL=http://<kind gateway>:8787/v1, OPENAI_MODEL=grok-4.6 (override via env)
uv run python scripts/e2e.py --skip-mock --real 3  # 3 short real tasks -> e2e-out/report.json
```

The gateway must run in Hermes's Python runtime (`hermes --print-runtime-command` plus `hermes_bootstrap`), which
`start-gateway.sh` handles. Mock latency / fault injection: `MOCK_LATENCY_S`, `MOCK_FAULT_*` on the gateway.

## Caveats
* kind is single-node: PVC is RWO (local-path). Multi-node needs RWX (NFS/EFS/CephFS).
* Workspace isolation is path-level inside the FS backend, not process/OS isolation.
* Payload offload on the PVC works for 1 node; for multi-host use `s3_payload_storage`.
* SuperGrok OAuth: the token is issued to **Hermes Agent's** OAuth client. Reusing it through this gateway is a
  personal/dev convenience. Check xAI's terms, and use an `XAI_API_KEY` (`OPENAI_BASE_URL=https://api.x.ai/v1`) for anything shared.
  Some subscription tiers get HTTP 403 on the API (Hermes issue #26847). The gateway passes that through as non-retryable.
* The gateway listens on 0.0.0.0:8787 and is protected only by the local bearer key. Firewall it if the box is exposed.
* Real-model runs spend subscription quota. The 4-execution e2e used about 12k tokens.
