import os

os.environ.setdefault("OPENAI_AGENTS_DISABLE_TRACING", "1")
import tempfile

import pytest
import pytest_asyncio

_tmp = tempfile.mkdtemp(prefix="agent-demo-")
os.environ["WORKSPACES_ROOT"] = os.path.join(_tmp, "workspaces")
os.environ["PAYLOAD_DIR"] = os.path.join(_tmp, "payloads")
os.makedirs(os.environ["WORKSPACES_ROOT"], exist_ok=True)

from temporalio.testing import WorkflowEnvironment  # noqa: E402

from temporal_agent_demo.config import harness_plugin  # noqa: E402


@pytest.fixture(scope="session")
def ws_root() -> str:
    return os.environ["WORKSPACES_ROOT"]


@pytest_asyncio.fixture(scope="session")
async def env():
    # Local Temporal dev server (auto-downloaded); no k8s needed.
    async with await WorkflowEnvironment.start_local(plugins=[harness_plugin()]) as e:
        yield e


# ---- shared mock-llm HTTP server (OpenAI chat-completions) -------------------------------------
import socket  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402

import uvicorn  # noqa: E402

MOCK_KEY = "sk-http-test"


@pytest.fixture(scope="session")
def mock_llm():
    from mock_llm_server.app import Settings, create_app

    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    app = create_app(Settings(api_keys={MOCK_KEY}))
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    t = threading.Thread(target=server.run, daemon=True)
    t.start()
    while not server.started:
        time.sleep(0.05)
    yield app, f"http://127.0.0.1:{port}/v1"
    server.should_exit = True
    t.join(5)


@pytest.fixture
def mock(mock_llm):
    """Fresh counters + restored fault settings per test."""
    app, url = mock_llm
    st = app.state.settings
    saved = (st.fault_every, st.fault_attempts, st.fault_codes, st.latency_s)
    app.state.stats.clear()
    app.state.fault_attempts.clear()
    yield app, url
    st.fault_every, st.fault_attempts, st.fault_codes, st.latency_s = saved
