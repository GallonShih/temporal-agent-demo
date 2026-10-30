import os
from pathlib import Path, PurePosixPath as P

import pytest

from temporal_agent_demo.shared import ExecutionContext, new_execution_id
from temporal_agent_demo.workspace_fs import WorkspaceDisk, WorkspaceDiskConfig


def disk(eid=None):
    eid = eid or new_execution_id()
    return eid, WorkspaceDisk(WorkspaceDiskConfig(workspace_id=eid))


async def test_write_read_list_confined(ws_root):
    eid, fs = disk()
    await fs.mkdir(P("."))
    await fs.mkdir(P("output"))
    await fs.write(P("output/a.txt"), b"hello")
    assert await fs.read(P("output/a.txt")) == b"hello"
    assert await fs.list(P("output")) == ["a.txt"]
    assert (Path(ws_root) / eid / "output/a.txt").read_bytes() == b"hello"


@pytest.mark.parametrize("bad", ["../x", "a/../../x", "/etc/passwd", "..", "output/../../exec-x"])
async def test_traversal_rejected(bad):
    _, fs = disk()
    await fs.mkdir(P("."))
    for op in (fs.read(P(bad)), fs.list(P(bad)), fs.write(P(bad), b"x"), fs.stat(P(bad))):
        with pytest.raises(PermissionError):
            await op


async def test_symlink_escape_rejected(ws_root):
    victim, vfs = disk()
    await vfs.mkdir(P("."))
    await vfs.write(P("secret.txt"), b"s3cret")
    eid, fs = disk()
    await fs.mkdir(P("."))
    os.symlink(Path(ws_root) / victim, Path(ws_root) / eid / "link")
    with pytest.raises(PermissionError):
        await fs.read(P("link/secret.txt"))


@pytest.mark.parametrize("bad_id", ["../etc", "exec-123", "", "exec-" + "g" * 32, "/data", "exec-" + "a" * 32 + "/.."])
def test_bad_workspace_id_rejected(bad_id):
    with pytest.raises(ValueError):
        WorkspaceDisk(WorkspaceDiskConfig.model_construct(workspace_id=bad_id)) if False else WorkspaceDiskConfig(
            workspace_id=bad_id
        )


async def test_write_is_idempotent_and_atomic(ws_root):
    eid, fs = disk()
    await fs.mkdir(P("."))
    for _ in range(3):  # simulated activity retries
        await fs.write(P("f.txt"), b"same")
    assert (Path(ws_root) / eid / "f.txt").read_bytes() == b"same"
    assert sorted(os.listdir(Path(ws_root) / eid)) == ["f.txt"]  # no temp leftovers, no duplication
    await fs.mkdir(P("."))  # retry-safe mkdir
    await fs.rename(P("f.txt"), P("g.txt"))
    await fs.rename(P("f.txt"), P("g.txt"))  # retried rename is a no-op
    await fs.delete(P("g.txt"))
    await fs.delete(P("g.txt"))
    assert await fs.list(P(".")) == []


def test_execution_context_ids_equal():
    ctx = ExecutionContext.new()
    assert ctx.execution_id == ctx.session_id == ctx.workspace_id
    assert ExecutionContext.new().execution_id != ctx.execution_id
