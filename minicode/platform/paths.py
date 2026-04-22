from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class AppPaths:
    global_dir: Path
    project_dir: Path
    local_dir: Path
    db_path: Path
    config_path: Path
    mcp_config_path: Path
    skills_dir: Path
    export_dir: Path
    logs_dir: Path


def resolve_paths(cwd: str | Path) -> AppPaths:
    workspace = Path(cwd).resolve()
    global_root = Path(os.environ.get("MINICODE_HOME", Path.home() / ".minicode")).resolve()
    project_root = workspace / ".minicode"
    local_root = workspace / ".minicode-local"
    global_root.mkdir(parents=True, exist_ok=True)
    project_root.mkdir(parents=True, exist_ok=True)
    local_root.mkdir(parents=True, exist_ok=True)
    skills_dir = global_root / "skills"
    export_dir = global_root / "exports"
    logs_dir = global_root / "logs"
    skills_dir.mkdir(parents=True, exist_ok=True)
    export_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    return AppPaths(
        global_dir=global_root,
        project_dir=project_root,
        local_dir=local_root,
        db_path=global_root / "runtime.sqlite",
        config_path=global_root / "config.json",
        mcp_config_path=global_root / "mcp_servers.json",
        skills_dir=skills_dir,
        export_dir=export_dir,
        logs_dir=logs_dir,
    )
