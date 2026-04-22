from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class IsolationContext:
    task_id: str
    worktree_path: Path
    original_path: Path
    branch_name: str
    created_at: float = field(default_factory=time.time)
    max_age_seconds: float = 3600.0

    def is_expired(self) -> bool:
        return (time.time() - self.created_at) > self.max_age_seconds


class IsolationError(RuntimeError):
    pass


class WorktreeIsolator:
    """Create/reap temporary git worktrees for risky command execution."""

    def __init__(self, base_dir: Path | None = None, prefix: str = "isolated") -> None:
        self.base_dir = base_dir or Path(tempfile.gettempdir()) / "minicode-isolation"
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.prefix = prefix
        self.active: dict[str, IsolationContext] = {}

    def create(
        self,
        source_path: Path,
        task_id: str | None = None,
        max_age_seconds: float = 3600.0,
    ) -> IsolationContext:
        tid = task_id or str(uuid.uuid4())[:8]
        branch = f"{self.prefix}_{tid}"
        worktree = self.base_dir / f"{self.prefix}_{tid}"
        if not (source_path / ".git").exists():
            raise IsolationError(f"Not a git repository: {source_path}")
        try:
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(source_path),
                    "worktree",
                    "add",
                    "-b",
                    branch,
                    str(worktree),
                    "HEAD",
                ],
                capture_output=True,
                text=True,
                check=True,
            )
        except subprocess.CalledProcessError as exc:
            raise IsolationError(f"worktree add failed: {exc.stderr}") from exc
        ctx = IsolationContext(
            task_id=tid,
            worktree_path=worktree,
            original_path=source_path,
            branch_name=branch,
            max_age_seconds=max_age_seconds,
        )
        self.active[tid] = ctx
        return ctx

    def execute(
        self,
        task_id: str,
        command: str,
        args: list[str],
        cwd: Path | None = None,
        timeout: int = 300,
    ) -> dict[str, Any]:
        ctx = self.active.get(task_id)
        if ctx is None:
            return {"ok": False, "output": f"Isolation context not found: {task_id}"}
        if ctx.is_expired():
            self.cleanup(task_id)
            return {"ok": False, "output": "Isolation context expired."}
        exec_cwd = cwd or ctx.worktree_path
        if not exec_cwd.exists():
            return {"ok": False, "output": f"Working directory not found: {exec_cwd}"}
        try:
            proc = subprocess.run(
                [command, *args],
                cwd=str(exec_cwd),
                env=os.environ.copy(),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
            )
            output = "\n".join(p for p in (proc.stdout.strip(), proc.stderr.strip()) if p)
            return {"ok": proc.returncode == 0, "output": output[:10_000]}
        except subprocess.TimeoutExpired:
            return {"ok": False, "output": f"Command timed out after {timeout}s"}
        except BaseException as exc:
            return {"ok": False, "output": f"Isolated execution failed: {exc}"}

    def cleanup(self, task_id: str) -> bool:
        ctx = self.active.pop(task_id, None)
        if ctx is None:
            return False
        try:
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(ctx.original_path),
                    "worktree",
                    "remove",
                    "-f",
                    str(ctx.worktree_path),
                ],
                capture_output=True,
                text=True,
            )
        except BaseException:
            pass
        if ctx.worktree_path.exists():
            try:
                shutil.rmtree(ctx.worktree_path)
            except BaseException:
                pass
        return True

    def cleanup_expired(self) -> list[str]:
        expired = [tid for tid, ctx in self.active.items() if ctx.is_expired()]
        for tid in expired:
            self.cleanup(tid)
        return expired

    def cleanup_all(self) -> list[str]:
        for tid in list(self.active):
            self.cleanup(tid)
        return []

    def status(self) -> dict[str, Any]:
        return {
            "active_isolations": len(self.active),
            "base_dir": str(self.base_dir),
            "isolations": [
                {
                    "task_id": tid,
                    "branch": ctx.branch_name,
                    "age_seconds": time.time() - ctx.created_at,
                    "expired": ctx.is_expired(),
                }
                for tid, ctx in self.active.items()
            ],
        }
