# temporal-agent-demo

2 identical Temporal agent workers (k8s Deployment `agent-worker`, replicas=2) on one task queue `agent`,
one shared PVC `agent-workspaces`, and one isolated workspace per execution. LLM is mocked.

## What is real harness vs. hand-written
* Uses **temporal-agent-harness 0.8.0** (PyPI, temporal-community/temporal-agent-harness):
  * `AgentHarnessPlugin(filesystems=[WorkspaceDisk], large_payload_offload=local_payload_storage(/data/payloads))`
    -> Pydantic data converter, large-payload offload (pointed at the shared PVC), and the
    `vfs.workspace-disk.*` activities.
  * `WorkspaceDisk` is a harness `ActivityFileSystem` (same pattern as the LocalDisk / code-mode memory
    example) whose root is `/data/workspaces/<workspace_id>`; the workflow uses `WorkspaceDisk.backend(cfg)`.
* The agent loop itself is a plain Temporal workflow (`AgentWorkflow`) + `llm_call` activity, NOT the harness's
  `@agent.defn` / `AgentWorkflowRunner` / OpenAIAgentsPlugin session runtime (that needs a real AI-SDK model
  integration + session manager; out of scope while the LLM is mocked).

## Layout
```
src/temporal_agent_demo/
  models.py        ExecutionContext (execution_id == workflow_id == session_id == workspace_id), LLM/tool models
  workspace_fs.py  WorkspaceDisk: path confinement + atomic idempotent writes
  llm.py           LLMClient interface, in-process MockLLM, OpenAICompatibleLLM (httpx), `llm_call` activity
  workflow.py      AgentWorkflow: bind workspace -> loop(llm -> tools) -> explicitly close session
  worker.py        worker process (task queue `agent`)
  api.py           FastAPI: POST /executions (generates id, starts workflow), GET /executions/{id}
  config.py        Temporal client + AgentHarnessPlugin
mock_llm_server/   standalone OpenAI-compatible mock provider (FastAPI) + its Dockerfile (image mock-llm:dev)
tests/             pytest (local Temporal dev server via temporalio.testing; no k8s)
k8s/               namespace, PVC, llm-api-key Secret, mock-llm Deployment+Service, worker Deployment (x2), API
scripts/deploy.sh  build -> kind load -> apply
scripts/e2e.py     10 executions, traversal probe, kill-a-pod crash recovery, mock-llm log audit
```

## Run
```bash
uv sync
uv run pytest -v                      # unit + integration (downloads temporal dev server once)

TEMPORAL_ADDRESS=<frontend-svc>.<ns>.svc.cluster.local:7233 scripts/deploy.sh
kubectl port-forward -n agent-demo svc/agent-api 18000:8000 &
kubectl port-forward -n <temporal-ns> svc/<frontend-svc> 17233:7233 &
mkdir -p e2e-out && uv run python scripts/e2e.py     # writes e2e-out/report.json (crash = kubectl delete --force)
CRASH_MODE=sigkill E2E_REPORT=e2e-out/report-sigkill.json uv run python scripts/e2e.py   # crash = SIGKILL worker process

curl -XPOST localhost:18000/executions -H 'content-type: application/json' -d '{"task":"hello"}'
```

## LLM: mock-llm service (OpenAI-compatible), like an agent talking to xAI / SuperGrok
Workers run `LLM_BACKEND=openai` and call `POST $OPENAI_BASE_URL/chat/completions` with `tools`
(write_file / read_file / list_files) over HTTP, Bearer key from Secret `llm-api-key`.
In k8s: `OPENAI_BASE_URL=http://mock-llm.agent-demo.svc:8000/v1`, `OPENAI_MODEL=grok-mock`.

mock-llm (`mock_llm_server/app.py`):
* `GET /v1/models`, `POST /v1/chat/completions` (function calling; deterministic scripted plan ->
  tool_calls ... then a final message), `GET /v1/stats`, `GET /healthz`.
* Bearer key required (`MOCK_API_KEYS`; empty = any non-empty key). Wrong model -> 404.
* `stream: true` is NOT supported (400).
* Latency `MOCK_LATENCY_S`; fault injection `MOCK_FAULT_EVERY=N` (requests whose hash % N == 0 fail
  `MOCK_FAULT_ATTEMPTS` times with a code from `MOCK_FAULT_CODES`, then succeed), `MOCK_FAULT_RATE`
  (random), or per request header `X-Mock-Fault: 429|500|...`. 429 sends `Retry-After: 1`.
* One JSON log line per request (status, fault, req_hash, client pod, ms).

Error mapping in `OpenAICompatibleLLM`: 429/408/409/5xx/network -> retryable `ApplicationError`
(Temporal activity retry, honours Retry-After); other 4xx (bad key, unknown model) -> non-retryable.

### Point it at a real provider (e.g. xAI)
```bash
kubectl -n agent-demo create secret generic llm-api-key --from-literal=api-key="$XAI_API_KEY" \
  --dry-run=client -o yaml | kubectl apply -f -
kubectl -n agent-demo set env deploy/agent-worker OPENAI_BASE_URL=https://api.x.ai/v1 OPENAI_MODEL=<grok model id>
kubectl -n agent-demo delete deploy/mock-llm svc/mock-llm   # optional
```
(the kind node needs internet egress for that). Locally: `LLM_BACKEND=openai OPENAI_BASE_URL=... OPENAI_API_KEY=... OPENAI_MODEL=...`.
`LLM_BACKEND=mock` still selects the in-process MockLLM.

Run mock-llm locally: `MOCK_API_KEYS=sk-x uv run uvicorn mock_llm_server.app:app --port 8001`

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
uv run python scripts/e2e_real.py                 # 3 short real tasks + 1 traversal probe -> e2e-out/report-real.json
```

The gateway must run in Hermes's Python runtime (`hermes --print-runtime-command` plus `hermes_bootstrap`), which
`start-gateway.sh` handles. The in-cluster `mock-llm` Deployment is kept at `replicas: 0`. To use it again, scale it to 1
and set `OPENAI_BASE_URL=http://mock-llm.agent-demo.svc:8000/v1 OPENAI_MODEL=grok-mock`. `scripts/e2e.py` (the fault-injection
audit) expects that setup.

## Caveats
* kind is single-node: PVC is RWO (local-path). Multi-node needs RWX (NFS/EFS/CephFS).
* Workspace isolation is path-level inside the FS backend, not process/OS isolation.
* Payload offload on the PVC works for 1 node; for multi-host use `s3_payload_storage`.
* SuperGrok OAuth: the token is issued to **Hermes Agent's** OAuth client. Reusing it through this gateway is a
  personal/dev convenience. Check xAI's terms, and use an `XAI_API_KEY` (`OPENAI_BASE_URL=https://api.x.ai/v1`) for anything shared.
  Some subscription tiers get HTTP 403 on the API (Hermes issue #26847). The gateway passes that through as non-retryable.
* The gateway listens on 0.0.0.0:8787 and is protected only by the local bearer key. Firewall it if the box is exposed.
* Real-model runs spend subscription quota. The 4-execution e2e used about 12k tokens.


## Phase 4: native temporal-agent-harness runtime (WIP)

The agent is now `WorkspaceAgent` (`src/temporal_agent_demo/agent.py`), defined with
`@agent.defn`. It has a VFS mount `/workspace` → `WorkspaceDisk`, the Code Mode tool `run_code`,
the inline tools write_file, read_file and list_files, and OpenAIAgentsPlugin chat-completions
pointed at the gateway. `POST /executions` starts the session through update-with-start
(`AgentClient.start_and_submit_message`), and the session closes itself after its task.
The old hand-written AgentWorkflow lives in `temporal_agent_demo/legacy/`. Set
`AGENT_ENABLE_LEGACY=1` to register it, and use `engine=legacy` on the API.
Tests: `uv run pytest -q tests` (38 tests: 8 native and 30 legacy/gateway/fs).
