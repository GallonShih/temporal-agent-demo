"""DevTaskWorkflow sequencing + resume wiring (fake activities), session dir and run_pytest confinement."""

import os
from pathlib import Path

import pytest
from pydantic import ValidationError
from temporalio import activity
from temporalio.client import Client
from temporalio.worker import Worker

from temporal_agent_demo import activities as A
from temporal_agent_demo.claude_tools import build_server, run_pytest_in
from temporal_agent_demo.config import plugins
from temporal_agent_demo.dev_workflow import DevTaskWorkflow
from temporal_agent_demo.shared import (DEV_WORKFLOW, ClaudeAgentInput, ClaudeAgentResult, DevTaskInput,
                                        DevTaskResult, PytestRun, new_execution_id)

SID = "11111111-2222-3333-4444-{:012d}"


def test_session_dir_confined(ws_root):
    eid = new_execution_id()
    d = A.session_dir(eid)
    assert d.parent == Path(os.environ["SESSIONS_ROOT"]).resolve() and d.is_dir()
    assert oct(d.stat().st_mode & 0o777) == "0o700"
    assert Path(ws_root).resolve() not in d.parents
    for bad in ("../x", "exec-../../etc", "/tmp"):
        with pytest.raises(ValueError):
            A.session_dir(bad)


def test_session_dir_rejects_inside_workspaces(monkeypatch, ws_root):
    monkeypatch.setenv("SESSIONS_ROOT", os.path.join(ws_root, "sessions"))
    with pytest.raises(ValueError):
        A.session_dir(new_execution_id())


def test_resume_input_validation_and_options():
    eid = new_execution_id()
    with pytest.raises(ValidationError):
        ClaudeAgentInput(execution_id=eid, prompt="x", resume_session_id="../../etc/passwd")
    inp = ClaudeAgentInput(execution_id=eid, prompt="x", persist_session=True, resume_session_id=SID.format(1),
                           fork_session=True)
    o = A.build_options(inp, A.workspace_dir(eid), "/tmp/h", [])
    assert o.resume == SID.format(1) and o.fork_session is True
    o = A.build_options(ClaudeAgentInput(execution_id=eid, prompt="x"), A.workspace_dir(eid), "/tmp/h", [])
    assert o.resume is None and o.fork_session is False  # backward compatible defaults


async def test_resume_requires_persist():
    from temporalio.testing import ActivityEnvironment
    with pytest.raises(ValueError):
        await ActivityEnvironment().run(A.run_claude_agent, ClaudeAgentInput(
            execution_id=new_execution_id(), prompt="x", resume_session_id=SID.format(1)))


def _ws(files: dict[str, str]) -> Path:
    root = A.workspace_dir(new_execution_id())
    for k, v in files.items():
        (root / k).parent.mkdir(parents=True, exist_ok=True)
        (root / k).write_text(v)
    return root


async def test_run_pytest_pass_fail_and_confinement():
    root = _ws({"m.py": "def f():\n    return 1\n", "tests/test_m.py": "from m import f\ndef test_f():\n    assert f() == 1\n"})
    code, out = await run_pytest_in(root, ".")
    assert code == 0 and "1 passed" in out
    (root / "tests/test_bad.py").write_text("def test_x():\n    assert False\n")
    code, out = await run_pytest_in(root)
    assert code == 1 and "1 failed" in out
    assert (await run_pytest_in(root, "../"))[0] == 2
    assert (await run_pytest_in(root, "/etc"))[0] == 2


async def test_run_pytest_no_secrets_timeout_truncation(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "sekrit-should-not-leak")
    root = _ws({"test_env.py": "import os\ndef test_env():\n    assert 'ANTHROPIC_AUTH_TOKEN' not in os.environ\n"
                               "    print('x' * 20000)\n    assert False\n"})
    code, out = await run_pytest_in(root)
    assert code == 1 and "sekrit" not in out and "[truncated]" in out and len(out) < 7000
    root = _ws({"test_slow.py": "import time\ndef test_s():\n    time.sleep(30)\n"})
    code, out = await run_pytest_in(root, timeout_s=2)
    assert code == 124 and "timeout" in out


async def test_run_pytest_tool_bound_to_root():
    root = _ws({"test_a.py": "def test_a():\n    pass\n"})
    servers, ids = build_server(["run_pytest"], root)
    assert ids == ["mcp__demo__run_pytest"] and "demo" in servers
    with pytest.raises(ValueError):
        build_server(["run_pytest"])


def _fakes(test_results: list[bool], calls: list):
    n = {"sid": 0}

    @activity.defn(name="run_claude_agent")
    async def fake_claude(inp: ClaudeAgentInput) -> ClaudeAgentResult:
        calls.append(("claude", inp.step, inp.resume_session_id, inp.persist_session, tuple(inp.tools),
                      tuple(inp.builtin_tools)))
        if inp.resume_session_id:  # resume keeps the same conversation id
            sid = inp.resume_session_id
        else:
            n["sid"] += 1
            sid = SID.format(n["sid"])
        return ClaudeAgentResult(execution_id=inp.execution_id, is_error=False, subtype="success",
                                 result=f"{inp.step} done", num_turns=2, session_id=sid, step=inp.step,
                                 resumed_from=inp.resume_session_id, worker="w1", usage={"input_tokens": 10})

    @activity.defn(name="run_workspace_tests")
    async def fake_tests(eid: str) -> PytestRun:
        ok = test_results.pop(0)
        calls.append(("test", ok))
        return PytestRun(exit_code=0 if ok else 1, passed=ok, output="1 passed" if ok else "1 failed", worker="w2")

    @activity.defn(name="write_workspace_files")
    async def fake_write(eid: str, files: dict[str, str]) -> list[str]:
        calls.append(("write", sorted(files)))
        return sorted(files)

    return [fake_claude, fake_tests, fake_write]


async def _run(env, inp: DevTaskInput, results: list[bool]):
    calls: list = []
    c = await Client.connect(env.client.service_client.config.target_host, plugins=plugins("http://unused/v1", "x"))
    async with Worker(c, task_queue="dev-test", workflows=[DevTaskWorkflow], activities=_fakes(results, calls)):
        res = await c.execute_workflow(DEV_WORKFLOW, inp, id=inp.execution_id, task_queue="dev-test",
                                       result_type=DevTaskResult)
    return res, calls


async def test_dev_workflow_fix_loop_resumes(env):
    eid = new_execution_id()
    inp = DevTaskInput(execution_id=eid, task="t", acceptance_tests={"tests/test_acc.py": "x"})
    res, calls = await _run(env, inp, [False, False, True])
    seq = [c[1] if c[0] == "claude" else c[0] for c in calls]
    assert seq == ["plan", "implement", "write", "test", "fix-1", "test", "fix-2", "test", "review"]
    cl = {c[1]: c for c in calls if c[0] == "claude"}
    impl_sid = SID.format(2)
    assert cl["plan"][2] is None and cl["implement"][2] is None and cl["review"][2] is None  # fresh
    assert cl["fix-1"][2] == impl_sid and cl["fix-2"][2] == impl_sid  # resumed implement session
    assert all(c[3] for c in cl.values())  # persist_session everywhere
    assert "run_pytest" in cl["fix-1"][4] and "Edit" not in cl["review"][5]
    assert res.status == "passed" and res.fix_iterations == 2
    names = [s.name for s in res.steps]
    assert names == ["plan", "implement", "test-0", "fix-1", "test-1", "fix-2", "test-2", "review"]
    fix = res.steps[3]
    assert fix.resumed_from == impl_sid and fix.session_id == impl_sid and fix.turns == 2 and fix.worker == "w1"


async def test_dev_workflow_gives_up_after_max_fix(env):
    eid = new_execution_id()
    res, calls = await _run(env, DevTaskInput(execution_id=eid, task="t", max_fix_iterations=1), [False, False])
    assert res.status == "failed" and res.fix_iterations == 1
    assert [s.name for s in res.steps][-1] == "review" and not any(c[0] == "write" for c in calls)


async def test_dev_workflow_passes_first_time(env):
    eid = new_execution_id()
    res, _ = await _run(env, DevTaskInput(execution_id=eid, task="t"), [True])
    assert res.status == "passed" and [s.name for s in res.steps] == ["plan", "implement", "test-0", "review"]


async def test_write_workspace_files_readable_and_confined():
    from temporalio.testing import ActivityEnvironment
    eid = new_execution_id()
    out = await ActivityEnvironment().run(A.write_workspace_files, eid, {"tests/test_acc.py": "def test_a():\n    pass\n"})
    root = A.workspace_dir(eid)
    assert out == ["tests/test_acc.py"] and oct((root / out[0]).stat().st_mode & 0o777) == "0o644"
    for bad in ("../x.py", "/etc/x.py"):
        with pytest.raises(ValueError):
            await ActivityEnvironment().run(A.write_workspace_files, eid, {bad: "x"})
