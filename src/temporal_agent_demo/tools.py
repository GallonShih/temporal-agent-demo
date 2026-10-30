"""Agent tools over the session's bound /workspace mount.

These are harness *workflow-side* tools (``@agent.tool_defn``): they run inside the workflow and
every file operation they make is a ``vfs.workspace-disk.*`` ACTIVITY executed by WorkspaceDisk
(workspace_fs.py) on whichever worker picks it up. There are no hand-written activities in this
project: model calls are ``invoke_model_activity`` (OpenAIAgentsPlugin) and file I/O is the
``vfs.*`` activities (AgentHarnessPlugin), both registered by the client plugins in config.py.

``run_code`` (Code Mode) is not defined here: it is built per session from the bound mount with
``agent.code_mode_tool`` in workflows.py.

NB: the harness calls ``user_fn(*model_args, **injected, **model_kwargs)``, so ``Injected``
params must come after the model params (keyword-only here).
"""

from __future__ import annotations

from pathlib import PurePosixPath
from typing import Any

from temporalio.exceptions import ActivityError, ApplicationError
from temporal_agent_harness.harness import agent
from temporal_agent_harness.harness.agent_workflow import Injected

from .shared import MAX_OUTPUT, MOUNT


def to_workspace_path(path: str) -> PurePosixPath:
    """Virtual path -> path relative to the workspace. '..' is NOT collapsed here: WorkspaceDisk
    (inside the activity) is the authority and rejects it."""
    p = PurePosixPath(path)
    if p.is_absolute():
        if p == PurePosixPath(MOUNT):
            return PurePosixPath(".")
        if not str(p).startswith(MOUNT + "/"):
            raise PermissionError(f"PermissionError: path outside {MOUNT}: {path!r}")
        p = PurePosixPath(str(p)[len(MOUNT) + 1 :])
    return p


async def _fs(op: str, fs: Any, *args: Any) -> Any:
    try:
        return await getattr(fs, op)(*args)
    except ActivityError as e:  # surface WorkspaceDisk's PermissionError etc. to the model
        c = e.cause
        kind = getattr(c, "type", None) or type(c).__name__
        raise ApplicationError(f"{kind}: {getattr(c, 'message', c)}", type=kind, non_retryable=True) from None


@agent.tool_defn(inherently_safe=True)
async def write_file(path: str, content: str, *, ws: Injected[Any]) -> str:
    """Write (create or replace) a UTF-8 text file. `path` is under /workspace, e.g. /workspace/output/report.md."""
    await _fs("write", ws, to_workspace_path(path), content.encode())
    return f"wrote {path} ({len(content.encode())} bytes)"


@agent.tool_defn(inherently_safe=True)
async def read_file(path: str, *, ws: Injected[Any]) -> str:
    """Read a UTF-8 text file under /workspace."""
    data: bytes = await _fs("read", ws, to_workspace_path(path))
    return data.decode(errors="replace")[:MAX_OUTPUT]


@agent.tool_defn(inherently_safe=True)
async def list_files(path: str = MOUNT, *, ws: Injected[Any]) -> list[str]:
    """List a directory under /workspace (default: the workspace root)."""
    return await _fs("list", ws, to_workspace_path(path))


FILE_TOOLS = (write_file, read_file, list_files)
