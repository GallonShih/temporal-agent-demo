"""WorkspaceDisk: an Agent-Harness ``ActivityFileSystem`` confined to /data/workspaces/<workspace_id>.

Same shape as the harness's LocalDisk / code-mode memory example (``LocalDiskConfig(directory=...)``)
but the root is the shared PVC mount and the per-execution directory is chosen by the
Execution Manager (the workflow input), never by the agent/model.

Guarantees:
* every path is resolved and must stay inside the workspace root (``..``, absolute paths,
  symlinks pointing outside -> PermissionError, which the harness turns into a non-retryable
  ApplicationError of type "PermissionError");
* the config itself is untrusted (the harness says so) -> workspace_id must match exec-<hex32>;
* writes are idempotent: temp file in the same dir + fsync + os.replace (atomic rename),
  so an activity retry after a crash just rewrites the same final content. No append API.
"""

from __future__ import annotations

import errno
import os
import shutil
import tempfile
from pathlib import Path, PurePosixPath

from pydantic import BaseModel, field_validator
from temporal_agent_harness.harness.code_mode.activity_fs import ActivityFileSystem
from temporal_agent_harness.harness.code_mode.vfs import FileStat

from .models import EXECUTION_ID_RE


def workspaces_root() -> Path:
    return Path(os.environ.get("WORKSPACES_ROOT", "/data/workspaces"))


class WorkspaceDiskConfig(BaseModel):
    workspace_id: str

    @field_validator("workspace_id")
    @classmethod
    def _shape(cls, v: str) -> str:
        if not EXECUTION_ID_RE.fullmatch(v):
            raise ValueError(f"invalid workspace id {v!r}")
        return v


def _deny(path: object) -> PermissionError:
    return PermissionError(errno.EACCES, "path escapes workspace", str(path))


class WorkspaceDisk(ActivityFileSystem[WorkspaceDiskConfig], name="workspace-disk"):
    def __init__(self, config: WorkspaceDiskConfig) -> None:
        super().__init__(config)
        base = workspaces_root().resolve()
        self._root = (base / config.workspace_id).resolve()
        if self._root.parent != base:  # belt and braces on top of the regex
            raise ValueError("workspace root outside workspaces base")

    # ---- confinement -------------------------------------------------------------------
    def _resolve(self, path: PurePosixPath | str) -> Path:
        p = PurePosixPath(path)
        if p.is_absolute() or ".." in p.parts:
            raise _deny(path)
        full = (self._root / p).resolve()  # follows symlinks
        if full != self._root and self._root not in full.parents:
            raise _deny(path)
        return full

    # ---- read side ---------------------------------------------------------------------
    async def stat(self, path: PurePosixPath) -> FileStat | None:
        full = self._resolve(path)
        if not full.exists():
            return None
        st = full.stat()
        return FileStat(is_dir=full.is_dir(), size=st.st_size, mtime=st.st_mtime)

    async def read(self, path: PurePosixPath, offset: int = 0, length: int | None = None) -> bytes:
        full = self._resolve(path)
        with open(full, "rb") as f:
            f.seek(offset)
            return f.read() if length is None else f.read(length)

    async def list(self, path: PurePosixPath) -> list[str]:
        full = self._resolve(path)
        return sorted(e for e in os.listdir(full) if not e.startswith(".tmp-"))

    # ---- write side (idempotent) -------------------------------------------------------
    async def write(self, path: PurePosixPath, data: bytes) -> None:
        full = self._resolve(path)
        if full == self._root:
            raise IsADirectoryError(errno.EISDIR, "is the workspace root", str(path))
        full.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=".tmp-", dir=full.parent)
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(data)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, full)  # atomic on POSIX; a retry just replaces again
        except BaseException:
            try:
                os.unlink(tmp)
            except FileNotFoundError:
                pass
            raise

    async def mkdir(self, path: PurePosixPath) -> None:
        # "." creates the workspace root itself. exist_ok => safe to retry.
        self._resolve(path).mkdir(parents=True, exist_ok=True)

    async def delete(self, path: PurePosixPath) -> None:
        full = self._resolve(path)
        if full == self._root:
            raise _deny(path)
        if full.is_dir():
            full.rmdir()
        elif full.exists():
            full.unlink()
        # missing => already deleted (idempotent)

    async def rename(self, path: PurePosixPath, target: PurePosixPath) -> None:
        src, dst = self._resolve(path), self._resolve(target)
        if not src.exists() and dst.exists():
            return  # retry after a successful rename
        dst.parent.mkdir(parents=True, exist_ok=True)
        os.replace(src, dst)


def remove_workspace(workspace_id: str) -> None:
    """Ops helper (not exposed to the agent)."""
    cfg = WorkspaceDiskConfig(workspace_id=workspace_id)
    shutil.rmtree(workspaces_root() / cfg.workspace_id, ignore_errors=True)
