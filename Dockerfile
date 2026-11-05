FROM python:3.13-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy PYTHONUNBUFFERED=1
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project
COPY src ./src
# dev group included: pytest is needed by the run_pytest tool / run_workspace_tests activity
RUN uv sync --frozen
ENV PATH="/app/.venv/bin:$PATH" WORKSPACES_ROOT=/data/workspaces PAYLOAD_DIR=/data/payloads SESSIONS_ROOT=/data/sessions
# default: worker; the API deployment overrides the command
CMD ["python", "-m", "temporal_agent_demo.worker"]
