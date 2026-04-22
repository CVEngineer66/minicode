from __future__ import annotations

import json
import time
from typing import Any


class CollaborationRepository:
    def __init__(self, db: Any) -> None:
        self.db = db

    def register_agent(self, name: str, description: str, skills: list[str], status: str = "idle") -> None:
        with self.db.connection() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO collaboration_agents(name, description, skills_json, status) VALUES(?, ?, ?, ?)",
                (name, description, json.dumps(skills), status),
            )

    def list_agents(self) -> list[dict[str, Any]]:
        with self.db.connection() as conn:
            rows = conn.execute("SELECT name, description, skills_json, status FROM collaboration_agents ORDER BY name").fetchall()
        return [{"name": row[0], "description": row[1], "skills": json.loads(row[2] or "[]"), "status": row[3]} for row in rows]

    def post_message(self, channel: str, sender: str, recipient: str, content: str) -> None:
        with self.db.connection() as conn:
            conn.execute(
                "INSERT INTO collaboration_messages(channel, sender, recipient, content, created_at, status) VALUES(?, ?, ?, ?, ?, ?)",
                (channel, sender, recipient, content, time.time(), "posted"),
            )

    def list_messages(self, channel: str) -> list[dict[str, Any]]:
        with self.db.connection() as conn:
            rows = conn.execute(
                "SELECT sender, recipient, content, created_at, status FROM collaboration_messages WHERE channel = ? ORDER BY created_at",
                (channel,),
            ).fetchall()
        return [
            {"sender": row[0], "recipient": row[1], "content": row[2], "created_at": row[3], "status": row[4]}
            for row in rows
        ]
