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
  activities.py    run_claude_agent: Claude Agent SDK (Claude Code) run confined to the workspace
  claude_tools.py  worker-side custom tool registry for the Claude activity (in-process SDK MCP server)
  claude_workflow.py  ClaudeTaskWorkflow: one run_claude_agent activity per execution
  claude_starter.py   CLI: start a ClaudeTaskWorkflow with prompt / skills / tools
  worker.py        worker process (task queue `agent`)
  starter.py       CLI: start one execution and print its result
  api.py           FastAPI: POST /executions, GET /executions/{id}, POST /executions/{id}/close, /healthz
  config.py        Temporal client, AgentHarnessPlugin + OpenAIAgentsPlugin (tracing disabled)
llm_gateway/       host-side gateway (OpenAI chat completions + Anthropic /v1/messages): real Grok (Hermes xAI OAuth) or the mock
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

## Claude Agent SDK activity (PoC)

`ClaudeTaskWorkflow` (`claude_workflow.py`) runs one activity, `run_claude_agent` (`activities.py`),
which drives Claude Code through the Claude Agent SDK **inside a single activity**. Claude Code talks
to the same host gateway through its Anthropic-compatible endpoint (`POST /v1/messages`, plus
`/v1/messages/count_tokens`, translated to chat completions in `llm_gateway/anthropic_compat.py`),
so it runs on grok-4.6 via SuperGrok (or `grok-mock`). No Anthropic key and no Node are needed: the SDK ships a native
Claude Code binary.

Input (`ClaudeAgentInput`, recorded in history, no secrets): `execution_id`, `prompt`,
`system_prompt` (string or `{preset, append}`), `skills` (name -> SKILL.md text, written atomically
to `<workspace>/.claude/skills/<name>/SKILL.md`), `enabled_skills`, `tools` (names from the
worker-side registry in `claude_tools.py`: `word_count`, `current_utc_time`), `builtin_tools`
(default Read/Write/Edit/Glob/Grep; Bash/Web* rejected), `model`, `max_turns`.

Confinement: cwd `/data/workspaces/<execution id>`, `setting_sources=["project"]`,
`strict_mcp_config`, `permission_mode="dontAsk"`, a PreToolUse hook denying any path outside the
workspace (symlinks resolved), throwaway HOME/CLAUDE_CONFIG_DIR. Workers get
`ANTHROPIC_BASE_URL=http://<kind gateway>:8787` and `ANTHROPIC_AUTH_TOKEN` from the `llm-api-key`
Secret (pod env only). Temporal owns retries (`CLAUDE_CODE_MAX_RETRIES=0`, 2 attempts, 60 s heartbeat).

```bash
kubectl -n agent-demo exec -i deploy/agent-worker -- sh -c 'cat > /tmp/SKILL.md' < my-skill/SKILL.md
kubectl -n agent-demo exec deploy/agent-worker -- python -m temporal_agent_demo.claude_starter \
  "Write a poem to poem.txt, count its words, write output/report.md" \
  --model grok-4.6 --tool word_count --skill report-style=/tmp/SKILL.md --append-system "Be brief."
```

Limits: a retry reruns the whole agent; the gateway replays streaming from a non-streamed upstream
call (no token-by-token output); `count_tokens` is a chars/4 estimate; images/thinking/cache_control
are dropped; the grok-mock path just calls the first offered tool with placeholder args.

## DevTaskWorkflow: multi-step coding PoC (plan → implement → test → fix → review)

`dev_workflow.py` chains `run_claude_agent` steps over one workspace:

| step | session | tools |
|---|---|---|
| plan | fresh | Read/Write/Edit/Glob/Grep → writes `PLAN.md`, `NOTES.md` |
| implement | fresh (handoff: task + PLAN.md + plan summary) | file tools + `run_pytest` |
| test-N | no LLM: `run_workspace_tests` activity (`python -m pytest -q`), ground truth | – |
| fix-N (≤ `max_fix_iterations`, default 3) | **resumes the implement session** (`resume=<session_id>`) | file tools + `run_pytest` |
| review | fresh, writes `REVIEW.md` | Read/Glob/Grep/Write |

- **Long-term memory = workspace files** (`PLAN.md`, `NOTES.md` handoff log, `REVIEW.md`); fresh steps get a short handoff.
- **Session persistence**: with `persist_session=True` the activity sets `HOME`/`CLAUDE_CONFIG_DIR` to
  `$SESSIONS_ROOT/<execution id>` (`/data/sessions`, PVC subPath `sessions`, mode 0700, validated, never inside
  the workspaces tree; the file guard denies agent access to it). Claude Code transcripts live there, keyed by
  cwd, so any worker pod can resume. New `ClaudeAgentInput` fields (all optional, backward compatible):
  `persist_session`, `resume_session_id` (UUID), `fork_session`, `step`. Results add `step`, `resumed_from`,
  `worker` (pod), `attempt`.
- **Hidden acceptance tests**: `acceptance_tests={path: content}` are written after implement, so the fix loop has
  to satisfy a spec the implementer did not see.
- **`run_pytest` tool** (registry, bound to the run's workspace): `python -m pytest` via exec (no shell), cwd = workspace,
  path confined, minimal env (no secrets), runs as `nobody` when the worker is root (cannot read the worker's
  `/proc/*/environ`), rlimits (CPU, 1 GiB AS, nproc), wall timeout (`RUN_PYTEST_TIMEOUT_S`, 90s), output truncated.
  General Bash stays disabled.
- **State**: query `progress` → `DevTaskResult` (per step: status, session_id, resumed_from, worker, attempt,
  turns, usage, tool_uses, denied, summary); the same object is the workflow result.
- **Retries**: an activity retry re-runs the step with the *same input*; for a fix step it resumes from the last
  completed session id again (the transcript may already contain the partial turns of the failed attempt; Claude
  Code continues from it). Workspace edits of a failed attempt are not rolled back.

```bash
python -m temporal_agent_demo.dev_starter "Create roman.py with to_roman/from_roman + pytest tests" \
    --acceptance tests/test_acceptance.py=./hidden_test.py --model grok-4.6 --max-fix 3
```

## Caveats
* kind is single-node: PVC is RWO (local-path). Multi-node needs RWX (NFS/EFS/CephFS).
* Workspace isolation is path-level inside the FS backend, not process/OS isolation.
* Payload offload on the PVC works for 1 node; for multi-host use `s3_payload_storage`.
* SuperGrok OAuth: the token is issued to **Hermes Agent's** OAuth client. Reusing it through this gateway is a
  personal/dev convenience. Check xAI's terms, and use an `XAI_API_KEY` (`OPENAI_BASE_URL=https://api.x.ai/v1`) for anything shared.
  Some subscription tiers get HTTP 403 on the API (Hermes issue #26847). The gateway passes that through as non-retryable.
* The gateway listens on 0.0.0.0:8787 and is protected only by the local bearer key. Firewall it if the box is exposed.
* Real-model runs spend subscription quota. The 4-execution e2e used about 12k tokens.
