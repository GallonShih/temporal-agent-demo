"""Host-side OpenAI-compatible LLM gateway for the kind workers.

Runs on the box (not in kind), bound to 0.0.0.0, reachable from pods via the kind docker bridge
gateway (172.18.0.1). Workers authenticate with a LOCAL bearer key (GATEWAY_API_KEYS); the gateway
attaches the upstream credential itself, so the upstream token never leaves this process.

Modes (GATEWAY_MODE):
  mock       every request is served by the scripted mock (mock_llm_server, in-process ASGI)
  xai-oauth  forward to xAI (https://api.x.ai/v1/chat/completions) with the SuperGrok OAuth access
             token obtained from Hermes Agent's own credential store/refresh helpers
             (hermes_cli.auth_xai.resolve_xai_oauth_runtime_credentials), read per request so
             refreshes are picked up; on upstream 401 it force-refreshes once and retries.
             Model "grok-mock" is still routed to the mock in this mode.

Must run inside Hermes's Python runtime for xai-oauth (scripts/start-gateway.sh does this).
Token values are never logged or returned.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from mock_llm_server.app import Settings as MockSettings
from mock_llm_server.app import create_app as create_mock_app

log = logging.getLogger("llm_gateway")
MOCK_MODEL = "grok-mock"


def _err(status: int, msg: str, typ: str) -> JSONResponse:
    return JSONResponse({"error": {"message": msg, "type": typ, "code": status}}, status_code=status)


class XaiOAuthUpstream:
    """Bearer from Hermes's xAI OAuth store (refresh handled by Hermes's helper)."""

    def __init__(self) -> None:
        from hermes_cli.auth_xai import resolve_xai_oauth_runtime_credentials  # Hermes runtime only

        self._resolve = resolve_xai_oauth_runtime_credentials
        self._lock = asyncio.Lock()

    async def credential(self, force_refresh: bool = False) -> tuple[str, str]:
        async with self._lock:  # serialize refreshes; the helper also takes Hermes's file lock
            c = await asyncio.to_thread(self._resolve, force_refresh=force_refresh)
        return c["base_url"].rstrip("/"), c["api_key"]

    async def request(self, method: str, path: str, body: dict | None = None) -> httpx.Response:
        base, token = await self.credential()
        async with httpx.AsyncClient(timeout=float(os.environ.get("UPSTREAM_TIMEOUT_S", "180"))) as c:
            r = await c.request(method, base + path, json=body, headers={"Authorization": f"Bearer {token}"})
            if r.status_code == 401:  # token revoked/expired early -> refresh once
                base, token = await self.credential(force_refresh=True)
                r = await c.request(method, base + path, json=body, headers={"Authorization": f"Bearer {token}"})
        return r


def create_app() -> FastAPI:
    mode = os.environ.get("GATEWAY_MODE", "mock")
    keys = {k.strip() for k in os.environ.get("GATEWAY_API_KEYS", "").split(",") if k.strip()}
    if not keys:
        raise SystemExit("GATEWAY_API_KEYS must be set (local bearer key(s) for workers)")
    default_model = os.environ.get("GATEWAY_DEFAULT_MODEL", "grok-4.6")
    mock_app = create_mock_app(MockSettings(api_keys=set(), model=MOCK_MODEL))
    mock = httpx.AsyncClient(transport=httpx.ASGITransport(app=mock_app), base_url="http://mock/v1")
    upstream = XaiOAuthUpstream() if mode == "xai-oauth" else None

    app = FastAPI(title=f"llm-gateway ({mode})")
    app.state.stats = {"requests": 0, "upstream": 0, "mock": 0}

    def authed(request: Request) -> bool:
        h = request.headers.get("authorization", "")
        return h.lower().startswith("bearer ") and h[7:].strip() in keys

    @app.get("/healthz")
    async def healthz():
        return {"ok": True, "mode": mode, "default_model": default_model}

    @app.get("/v1/stats")
    async def stats():
        return app.state.stats

    @app.get("/v1/models")
    async def models(request: Request):
        if not authed(request):
            return _err(401, "missing or invalid gateway API key", "invalid_api_key")
        data = [{"id": MOCK_MODEL, "object": "model", "owned_by": "gateway-mock"}]
        if upstream:
            r = await upstream.request("GET", "/models")
            if r.status_code != 200:
                return JSONResponse(r.json() if r.headers.get("content-type", "").startswith("application/json")
                                    else {"error": {"message": r.text[:300]}}, status_code=r.status_code)
            data += [{"id": m["id"], "object": "model", "owned_by": "xai"} for m in r.json().get("data", [])]
        return {"object": "list", "data": data}

    @app.post("/v1/chat/completions")
    async def chat(request: Request):
        t0, rid = time.monotonic(), uuid.uuid4().hex[:12]
        rec: dict = {"event": "chat.completions", "rid": rid, "client": request.headers.get("x-client-id")}
        status = 500
        try:
            if not authed(request):
                status = 401
                return _err(401, "missing or invalid gateway API key", "invalid_api_key")
            body = await request.json()
            if body.get("stream"):
                status = 400
                return _err(400, "stream=true not supported by this gateway", "invalid_request_error")
            body.setdefault("model", default_model)
            rec.update(model=body["model"], n_messages=len(body.get("messages") or []), n_tools=len(body.get("tools") or []))
            app.state.stats["requests"] += 1
            if upstream is None or body["model"] == MOCK_MODEL:
                rec["route"] = "mock"
                app.state.stats["mock"] += 1
                fwd = {"Authorization": "Bearer local"}
                if f := request.headers.get("x-mock-fault"):
                    fwd["X-Mock-Fault"] = f
                r = await mock.post("/chat/completions", json={**body, "model": MOCK_MODEL}, headers=fwd)
            else:
                rec["route"] = "xai-oauth"
                app.state.stats["upstream"] += 1
                try:
                    r = await upstream.request("POST", "/chat/completions", body)
                except httpx.TransportError as e:
                    status = 502
                    return _err(502, f"upstream transport error: {type(e).__name__}", "upstream_error")
                except Exception as e:  # e.g. Hermes AuthError on refresh (no secrets in message)
                    status = 503
                    rec["error"] = type(e).__name__
                    return _err(503, f"upstream credential error: {type(e).__name__}", "upstream_auth_error")
            status = r.status_code
            try:
                payload = r.json()
            except ValueError:
                return _err(502 if status < 400 else status, f"non-JSON upstream response ({status})", "upstream_error")
            if status == 200:
                msg = payload["choices"][0]["message"]
                rec["tool_calls"] = [tc["function"]["name"] for tc in msg.get("tool_calls") or []]
                rec["finish"] = payload["choices"][0].get("finish_reason")
                u = payload.get("usage") or {}
                rec["usage"] = {k: u.get(k) for k in ("prompt_tokens", "completion_tokens", "total_tokens")}
            else:
                rec["upstream_error"] = str(payload)[:300]
            headers = {k: v for k, v in r.headers.items() if k.lower() == "retry-after"}
            return JSONResponse(payload, status_code=status, headers=headers)
        finally:
            rec.update(status=status, ms=round((time.monotonic() - t0) * 1000))
            log.info(json.dumps(rec))

    return app


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)  # never log request lines/headers
app = create_app()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=os.environ.get("GATEWAY_HOST", "0.0.0.0"), port=int(os.environ.get("GATEWAY_PORT", "8787")),
                log_level="info", access_log=False)
