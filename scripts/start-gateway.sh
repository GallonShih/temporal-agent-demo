#!/usr/bin/env bash
# Start the host-side LLM gateway (outside kind) inside Hermes Agent's Python runtime.
#   GATEWAY_MODE=xai-oauth|mock (default xai-oauth)  GATEWAY_PORT (default 8787)
#   Local worker key: llm_gateway/.gateway-key (generated once, chmod 600). Put the SAME value in the
#   k8s Secret llm-api-key (this script does it if kubectl is available).
#   Logs: llm_gateway/gateway.log   PID: llm_gateway/gateway.pid
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT=$PWD
HERMES_ROOT="${HERMES_ROOT:-$HOME/.hermes/hermes-agent}"
MODE="${GATEWAY_MODE:-xai-oauth}"
PORT="${GATEWAY_PORT:-8787}"
KEYFILE=llm_gateway/.gateway-key
if [ ! -s "$KEYFILE" ]; then
  umask 077; python3 -c 'import secrets,sys; sys.stdout.write("sk-gw-" + secrets.token_hex(24))' > "$KEYFILE"
fi
if [ -f llm_gateway/gateway.pid ] && kill -0 "$(cat llm_gateway/gateway.pid)" 2>/dev/null; then
  kill "$(cat llm_gateway/gateway.pid)"; sleep 1
fi
# Hermes's own runtime command (its bundled Python + hermes_bootstrap), then run our module.
HPY=$("$HOME/.local/bin/hermes" --print-runtime-command | python3 -c 'import json,sys; print(json.load(sys.stdin)[0])')
GATEWAY_MODE="$MODE" GATEWAY_PORT="$PORT" GATEWAY_API_KEYS="$(tr -d "[:space:]" < "$KEYFILE")" \
nohup "$HPY" -I -c "
import os, sys, runpy
os.environ.pop('PYTHONHOME', None); os.environ.pop('PYTHONPATH', None); os.environ.pop('VIRTUAL_ENV', None)
sys.path.insert(0, '$HERMES_ROOT'); sys._hermes_pin_default_home = True
import hermes_bootstrap
os.environ.get('HERMES_HOME') or os.environ.__setitem__('HERMES_HOME', str(__import__('hermes_constants').get_default_hermes_root()))
sys.path.insert(0, '$ROOT')
runpy.run_module('llm_gateway.server', run_name='__main__')
" >> llm_gateway/gateway.log 2>&1 &
echo $! > llm_gateway/gateway.pid
for i in $(seq 1 30); do curl -sf "localhost:$PORT/healthz" >/dev/null && break; sleep 0.5; done
echo "gateway pid $(cat llm_gateway/gateway.pid) mode=$MODE listening 0.0.0.0:$PORT  -> $(curl -s localhost:$PORT/healthz)"
if command -v kubectl >/dev/null && kubectl get ns agent-demo >/dev/null 2>&1; then
  kubectl -n agent-demo create secret generic llm-api-key --from-file=api-key="$KEYFILE" \
    --dry-run=client -o yaml | kubectl apply -f - >/dev/null && echo "k8s Secret agent-demo/llm-api-key synced with local gateway key"
fi
