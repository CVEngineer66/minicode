from __future__ import annotations

import subprocess
import time
import uuid
from pathlib import Path

from minicode.core.types import BackgroundTaskRecord


def run_command_sync(command: str, cwd: str, timeout: int = 60) -> tuple[int, str, str]:
    completed = subprocess.run(
        command,
        cwd=cwd,
        shell=True,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return completed.returncode, completed.stdout, completed.stderr


def start_background_command(command: str, cwd: str, log_dir: str | Path) -> tuple[subprocess.Popen[str], BackgroundTaskRecord]:
    task_id = uuid.uuid4().hex[:12]
    log_path = Path(log_dir) / f"background-{task_id}.log"
    handle = open(log_path, "w", encoding="utf-8")
    process = subprocess.Popen(command, cwd=cwd, shell=True, stdout=handle, stderr=subprocess.STDOUT, text=True)
    now = time.time()
    record = BackgroundTaskRecord(
        task_id=task_id,
        command=command,
        cwd=cwd,
        status="running",
        created_at=now,
        updated_at=now,
        output_path=str(log_path),
    )
    return process, record
