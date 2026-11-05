"""OrchestratorAgent: a WorkspaceAgent that can also drive WorkspaceAgent SUBAGENTS (harness subagent_toolset).

Each wired key is a child workflow (ParentClosePolicy.TERMINATE) of type WorkspaceAgent, started with
init data built HERE (developer code) - the model only gets `start_<key>()` (no arguments), so it
can never choose a workspace id, path, role or mode. Workspace mode per key:

* reviewer   - inherit + read-only : mounts the parent's /data/workspaces/<parent id>, cannot write.
* researcher - isolated, writable  : own workspace exec-<sha256(child workflow id)[:32]>.
* helper     - writable, mode taken from the START REQUEST (OrchestratorSession.helper_workspace_mode).

Children get no subagent tools (plain WorkspaceAgent), so there is no recursion.
Concurrency: an inherit-mode WRITER would race the parent on the same files (last atomic write
wins, no locking); the reviewer is read-only to avoid that. Use isolated mode for writers.
"""

from __future__ import annotations

from temporalio import workflow

with workflow.unsafe.imports_passed_through():
    from temporal_agent_harness.ai_sdks.openai_agents_harness import as_openai_agent_tool
    from temporal_agent_harness.harness import agent
    from temporal_agent_harness.harness.agent_protocol import AgentConfig
    from temporal_agent_harness.harness.agent_workflow import tool_defn
    from temporal_agent_harness.harness.subagent_toolset import subagent_toolset

    from .shared import MOUNT, ORCHESTRATOR_NAME, TASK_QUEUE, AgentRole, OrchestratorSession, WorkspaceSession
    from .workflows import WorkspaceAgent

REVIEWER = AgentRole(name="reviewer", workspace_mode="inherit", read_only=True, instructions=(
    "Review the files the parent points you to: correctness, clarity, concrete improvement suggestions. "
    "Quote the exact lines you comment on."))
RESEARCHER = AgentRole(name="researcher", workspace_mode="isolated", read_only=False, instructions=(
    "Produce the requested draft/notes as files in your own workspace and summarise them (include the "
    "full text of short files in your reply, since the parent cannot see your workspace)."))


def _positional_send(t):
    """harness 0.8.0 workaround: the generated `<key>_<fn>(subagent, **kw)` send tool advertises the
    signature (subagent, message) but the OpenAI-Agents adapter calls it positionally
    ("takes 1 positional argument but 2 were given"). Re-wrap with a real positional signature."""
    sig = getattr(t, "__signature__", None)
    params = list(sig.parameters) if sig else []
    if len(params) != 2 or params[0] != "subagent":
        return t  # start_/stop_ tools are fine
    pname = params[1]

    async def send(subagent, payload):
        return await t(subagent, **{pname: payload})

    send.__name__ = send.__qualname__ = t.__name__
    send.__doc__, send.__signature__, send.__annotations__ = t.__doc__, sig, dict(t.__annotations__)
    return tool_defn()(send)


def _toolset(**kw) -> list:
    return [_positional_send(t) for t in subagent_toolset(WorkspaceAgent, task_queue=TASK_QUEUE, **kw)]


def helper_role(mode: str) -> AgentRole:
    return AgentRole(name="helper", workspace_mode=mode, read_only=False,
                     instructions="Carry out the delegated subtask with your file tools.")


ORCH_PROMPT = f"""\
You are an orchestrator agent. Your workspace is mounted at `{MOUNT}` (input/, output/, temp/).
Besides file tools you can delegate to subagents. Start one with start_<key>() (returns a handle), then
send it ONE task with <key>_run_task(subagent=<handle>, message={{"task": "..."}}); it closes itself after replying.
- reviewer: READ-ONLY, sees YOUR workspace (same `{MOUNT}` paths). Use it to review files you wrote.
- researcher: writable but ISOLATED: its own empty workspace; it cannot see your files and you cannot see its
  files, so pass content in the task text and read results from its reply.
- helper: writable; workspace mode chosen by the operator ({{helper_mode}}).
When finished, reply with a summary and no tool calls."""


@agent.defn(name=ORCHESTRATOR_NAME)
class OrchestratorAgent(WorkspaceAgent):
    @agent.init
    def __init__(self, config: AgentConfig, data: OrchestratorSession) -> None:
        super().__init__(config, data)
        model, turns = data.model, data.max_turns

        def make(role: AgentRole):  # init_data callback: deterministic, from code + start params only
            return lambda: WorkspaceSession(role=role, model=model, max_turns=turns)

        self._sub_tools = (
            _toolset(key="reviewer", init_data=make(REVIEWER))
            + _toolset(key="researcher", init_data=make(RESEARCHER))
            + _toolset(key="helper", init_data=make(helper_role(data.helper_workspace_mode))))

    def _instructions(self) -> str:
        return ORCH_PROMPT.replace("{helper_mode}", self._data.helper_workspace_mode) \
            + f"\nsession_id={self._data.execution_id}"

    def _tools(self, injections: dict) -> list:
        return super()._tools(injections) + [as_openai_agent_tool(self._runner, t) for t in self._sub_tools]
