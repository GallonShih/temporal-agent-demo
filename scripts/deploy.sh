#!/usr/bin/env bash
# Build image, load into kind, deploy PVC + worker(x2) + API. Usage: TEMPORAL_ADDRESS=svc.ns:7233 scripts/deploy.sh
set -euo pipefail
cd "$(dirname "$0")/.."
: "${TEMPORAL_ADDRESS:?set TEMPORAL_ADDRESS, e.g. temporal-frontend.temporal.svc.cluster.local:7233}"
CLUSTER="${KIND_CLUSTER:-$(kind get clusters | head -1)}"
docker build -t agent-worker:dev .
docker build -t mock-llm:dev mock_llm_server
for img in agent-worker:dev mock-llm:dev; do
  if ! kind load docker-image "$img" --name "$CLUSTER"; then
    # fallback for "digest not found" on this box: import straight into the node's containerd
    docker save --platform linux/amd64 "$img" | \
      docker exec -i "${CLUSTER}-control-plane" ctr -n k8s.io images import --digests --snapshotter overlayfs -
  fi
done
GW_IP=$(docker network inspect kind -f '{{range .IPAM.Config}}{{if .Gateway}}{{.Gateway}} {{end}}{{end}}' | tr ' ' '\n' | grep -E '^[0-9.]+$' | head -1)
OPENAI_BASE_URL="${OPENAI_BASE_URL:-http://${GW_IP}:${GATEWAY_PORT:-8787}/v1}"
OPENAI_MODEL="${OPENAI_MODEL:-grok-4.6}"
kubectl apply -f k8s/00-namespace.yaml -f k8s/10-pvc.yaml -f k8s/18-mock-llm.yaml
kubectl get secret -n agent-demo llm-api-key >/dev/null 2>&1 || kubectl apply -f k8s/15-llm-secret.yaml
for f in k8s/20-worker.yaml k8s/30-api.yaml; do
  sed -e "s|__TEMPORAL_ADDRESS__|${TEMPORAL_ADDRESS}|" -e "s|__OPENAI_BASE_URL__|${OPENAI_BASE_URL}|" \
      -e "s|__OPENAI_MODEL__|${OPENAI_MODEL}|" "$f" | kubectl apply -f -
done
kubectl rollout restart -n agent-demo deploy/agent-worker deploy/agent-api >/dev/null 2>&1 || true
kubectl rollout status -n agent-demo deploy/agent-worker --timeout=180s
kubectl rollout status -n agent-demo deploy/agent-api --timeout=180s
kubectl get pods -n agent-demo -o wide
