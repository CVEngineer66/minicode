from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import Any

from langchain_core.messages import BaseMessage

from minicode.core.types import SessionMeta


def checkpoint_from_messages(messages: list[BaseMessage]) -> dict[str, Any]:
    return {
        "v": 2,
        "id": uuid.uuid4().hex,
        "ts": str(time.time()),
        "channel_values": {"messages": messages},
        "channel_versions": {"messages": 1},
        "versions_seen": {},
        "pending_sends": [],
        "updated_channels": ["messages"],
    }


class SessionRepository:
    def __init__(self, db: Any) -> None:
        self.db = db

    def upsert(self, meta: SessionMeta) -> None:
        with self.db.connection() as conn:
            conn.execute(
                """
                INSERT INTO session_meta(thread_id, workspace, created_at, updated_at, model, title)
                VALUES(?, ?, ?, ?, ?, ?)
                ON CONFLICT(thread_id) DO UPDATE SET
                    workspace = excluded.workspace,
                    updated_at = excluded.updated_at,
                    model = excluded.model,
                    title = CASE
                        WHEN excluded.title != '' THEN excluded.title
                        ELSE session_meta.title
                    END
                """,
                (
                    meta.thread_id,
                    meta.workspace,
                    meta.created_at,
                    meta.updated_at,
                    meta.model,
                    meta.title,
                ),
            )

    def get(self, thread_id: str) -> SessionMeta | None:
        with self.db.connection() as conn:
            row = conn.execute(
                "SELECT thread_id, workspace, created_at, updated_at, model, title FROM session_meta WHERE thread_id = ?",
                (thread_id,),
            ).fetchone()
        return SessionMeta(*row) if row else None

    def list(self, workspace: str | None = None) -> list[SessionMeta]:
        query = "SELECT thread_id, workspace, created_at, updated_at, model, title FROM session_meta"
        params: tuple[object, ...] = ()
        if workspace:
            query += " WHERE workspace = ?"
            params = (workspace,)
        query += " ORDER BY updated_at DESC"
        with self.db.connection() as conn:
            rows = conn.execute(query, params).fetchall()
        return [SessionMeta(*row) for row in rows]

    def latest(self, workspace: str | None = None) -> SessionMeta | None:
        sessions = self.list(workspace)
        return sessions[0] if sessions else None

    def load_messages(self, thread_id: str) -> list[BaseMessage]:
        with self.db.checkpointer() as saver:
            checkpoint = saver.get_tuple({"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}})
        if checkpoint is None:
            return []
        values = checkpoint.checkpoint.get("channel_values", {})
        messages = values.get("messages", [])
        return list(messages) if isinstance(messages, list) else []

