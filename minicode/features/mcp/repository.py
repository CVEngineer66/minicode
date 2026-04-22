from __future__ import annotations

import json
from typing import Any


class McpServerRepository:
    def __init__(self, db: Any) -> None:
        self.db = db

    def add(self, name: str, command: str, args: list[str], env: dict[str, str], cwd: str | None) -> None:
        with self.db.connection() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO mcp_servers(name, command, args_json, env_json, cwd) VALUES(?, ?, ?, ?, ?)",
                (name, command, json.dumps(args), json.dumps(env), cwd),
            )

    def list(self) -> list[dict[str, Any]]:
        with self.db.connection() as conn:
            rows = conn.execute("SELECT name, command, args_json, env_json, cwd FROM mcp_servers ORDER BY name").fetchall()
        return [
            {"name": row[0], "command": row[1], "args": json.loads(row[2] or "[]"), "env": json.loads(row[3] or "{}"), "cwd": row[4]}
            for row in rows
        ]
