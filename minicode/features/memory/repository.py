from __future__ import annotations

import json
import time
from typing import Any


class MemoryRepository:
    def __init__(self, db: Any) -> None:
        self.db = db

    def put(self, scope: str, workspace: str, entry_key: str, content: str, tags: list[str] | None = None) -> None:
        now = time.time()
        with self.db.connection() as conn:
            conn.execute(
                """
                INSERT INTO memory_entries(scope, workspace, entry_key, content, tags_json, created_at, updated_at)
                VALUES(?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(scope, workspace, entry_key) DO UPDATE SET
                    content = excluded.content,
                    tags_json = excluded.tags_json,
                    updated_at = excluded.updated_at
                """,
                (scope, workspace, entry_key, content, json.dumps(tags or []), now, now),
            )

    def list(self, scope: str | None = None, workspace: str | None = None) -> list[dict[str, Any]]:
        query = "SELECT scope, workspace, entry_key, content, tags_json, created_at, updated_at FROM memory_entries WHERE 1=1"
        params: list[object] = []
        if scope:
            query += " AND scope = ?"
            params.append(scope)
        if workspace:
            query += " AND workspace = ?"
            params.append(workspace)
        query += " ORDER BY updated_at DESC"
        with self.db.connection() as conn:
            rows = conn.execute(query, tuple(params)).fetchall()
        return [
            {
                "scope": row[0],
                "workspace": row[1],
                "entry_key": row[2],
                "content": row[3],
                "tags": json.loads(row[4] or "[]"),
                "created_at": row[5],
                "updated_at": row[6],
            }
            for row in rows
        ]

    def search(self, query: str, workspace: str, limit: int = 5) -> list[dict[str, Any]]:
        pattern = f"%{query.strip()}%"
        with self.db.connection() as conn:
            rows = conn.execute(
                """
                SELECT scope, workspace, entry_key, content, tags_json, created_at, updated_at
                FROM memory_entries
                WHERE workspace IN (?, '')
                  AND (entry_key LIKE ? OR content LIKE ? OR tags_json LIKE ?)
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (workspace, pattern, pattern, pattern, limit),
            ).fetchall()
        return [
            {
                "scope": row[0],
                "workspace": row[1],
                "entry_key": row[2],
                "content": row[3],
                "tags": json.loads(row[4] or "[]"),
                "created_at": row[5],
                "updated_at": row[6],
            }
            for row in rows
        ]
