from temporal_agent_demo.legacy.llm import MockLLM, make_llm
from temporal_agent_demo.models import LLMRequest, Message


async def test_mock_is_deterministic_and_terminates():
    llm = MockLLM()
    msgs = [Message(role="user", content="analyse repo")]
    outs = []
    for step in range(10):
        r1 = await llm.complete(LLMRequest(session_id="s", step=step, messages=msgs))
        r2 = await llm.complete(LLMRequest(session_id="s", step=step, messages=msgs))
        assert r1 == r2
        outs.append(r1)
        if r1.final:
            break
        msgs.append(Message(role="assistant", tool_calls=r1.tool_calls))
    assert outs[-1].final and outs[-1].final.startswith("DONE s")
    assert len(outs) == 4


def test_factory_default_mock(monkeypatch):
    monkeypatch.delenv("LLM_BACKEND", raising=False)
    assert make_llm().name == "mock"
