from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

from .types import HookContext, HookHandler


def logging_hook(log_file: Path | None = None) -> HookHandler:
    """Factory returning a hook that appends a compact line per event."""

    def handler(ctx: HookContext) -> None:
        ts = time.strftime("%H:%M:%S", time.localtime(ctx.timestamp))
        parts = [f"[{ts}]", ctx.event.value]
        if ctx.tool_name:
            parts.append(f"tool={ctx.tool_name}")
        if ctx.session_id:
            parts.append(f"session={ctx.session_id[:8]}")
        msg = " ".join(parts)
        if log_file is None:
            return
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    return handler


def script_hook(script_path: Path, timeout_s: float = 10.0) -> HookHandler:
    """Factory returning a hook that runs an external script synchronously.

    Uses subprocess.run with a timeout; does NOT use asyncio so the hook stays
    synchronous and can be composed with the timeout-bounded HookService.
    """

    def handler(ctx: HookContext) -> str:
        script_str = str(script_path)
        suffix = script_path.suffix.lower()
        if sys.platform == "win32" and suffix in (".py", ".sh", ".bat", ".cmd", ".ps1"):
            if suffix == ".py":
                cmd = [sys.executable, script_str]
            elif suffix in (".bat", ".cmd"):
                cmd = ["cmd", "/c", script_str]
            elif suffix == ".ps1":
                cmd = ["powershell", "-ExecutionPolicy", "Bypass", "-File", script_str]
            else:
                cmd = ["bash", script_str]
        else:
            cmd = [script_str]
        cmd = cmd + [ctx.event.value] + [str(v) for v in ctx.data.values()]
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout_s,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return f"script timeout after {timeout_s}s"
        if proc.returncode == 0:
            return proc.stdout
        return f"script failed (rc={proc.returncode}): {proc.stderr.strip()}"

    return handler
