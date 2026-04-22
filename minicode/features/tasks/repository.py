from __future__ import annotations

import json
import time
from typing import Any


# ---------------------------------------------------------------------------
# TaskTrackerRepository — flat workspace-scoped TODO items
# ---------------------------------------------------------------------------


class TaskTrackerRepository:
    def __init__(self, db: Any) -> None:
        self.db = db

    def add(self, workspace: str, title: str, note: str = "") -> None:
        now = time.time()
        with self.db.connection() as conn:
            conn.execute(
                "INSERT INTO task_items(workspace, title, status, note, created_at, updated_at) VALUES(?, ?, ?, ?, ?, ?)",
                (workspace, title, "open", note, now, now),
            )

    def list(self, workspace: str) -> list[dict[str, Any]]:
        with self.db.connection() as conn:
            rows = conn.execute(
                "SELECT id, title, status, note, created_at, updated_at FROM task_items WHERE workspace = ? ORDER BY updated_at DESC",
                (workspace,),
            ).fetchall()
        return [
            {
                "id": row[0],
                "title": row[1],
                "status": row[2],
                "note": row[3],
                "created_at": row[4],
                "updated_at": row[5],
            }
            for row in rows
        ]

    def set_status(self, workspace: str, task_id: int, status: str) -> bool:
        with self.db.connection() as conn:
            cur = conn.execute(
                "UPDATE task_items SET status = ?, updated_at = ? WHERE workspace = ? AND id = ?",
                (status, time.time(), workspace, task_id),
            )
            return cur.rowcount > 0


# ---------------------------------------------------------------------------
# TaskGraphRepository — persistent DAG of task nodes + edges
# ---------------------------------------------------------------------------


class TaskGraphRepository:
    def __init__(self, db: Any) -> None:
        self.db = db

    def upsert_node(self, node_id: str, workspace: str, title: str, status: str, metadata: dict[str, Any] | None = None) -> None:
        now = time.time()
        with self.db.connection() as conn:
            conn.execute(
                """
                INSERT INTO task_graph_nodes(node_id, workspace, title, status, metadata_json, created_at, updated_at)
                VALUES(?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(node_id) DO UPDATE SET
                    title = excluded.title,
                    status = excluded.status,
                    metadata_json = excluded.metadata_json,
                    updated_at = excluded.updated_at
                """,
                (node_id, workspace, title, status, json.dumps(metadata or {}), now, now),
            )

    def add_edge(self, parent_id: str, child_id: str) -> None:
        with self.db.connection() as conn:
            conn.execute("INSERT OR IGNORE INTO task_graph_edges(parent_id, child_id) VALUES(?, ?)", (parent_id, child_id))

    def list_graph(self, workspace: str) -> dict[str, Any]:
        with self.db.connection() as conn:
            nodes = conn.execute(
                "SELECT node_id, title, status, metadata_json FROM task_graph_nodes WHERE workspace = ? ORDER BY updated_at DESC",
                (workspace,),
            ).fetchall()
            edges = conn.execute("SELECT parent_id, child_id FROM task_graph_edges").fetchall()
        return {
            "nodes": [
                {"node_id": row[0], "title": row[1], "status": row[2], "metadata_json": row[3]}
                for row in nodes
            ],
            "edges": [{"parent_id": row[0], "child_id": row[1]} for row in edges],
        }


# ---------------------------------------------------------------------------
# BackgroundTaskRepository — records for long-running shell tasks
# ---------------------------------------------------------------------------


class BackgroundTaskRepository:
    def __init__(self, db: Any) -> None:
        self.db = db

    def upsert(self, record: Any) -> None:
        with self.db.connection() as conn:
            conn.execute(
                """
                INSERT INTO background_tasks(task_id, command, cwd, status, created_at, updated_at, return_code, output_path)
                VALUES(?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(task_id) DO UPDATE SET
                    status = excluded.status,
                    updated_at = excluded.updated_at,
                    return_code = excluded.return_code,
                    output_path = excluded.output_path
                """,
                (
                    record.task_id,
                    record.command,
                    record.cwd,
                    record.status,
                    record.created_at,
                    record.updated_at,
                    record.return_code,
                    record.output_path,
                ),
            )

    def list(self) -> list[dict[str, Any]]:
        with self.db.connection() as conn:
            rows = conn.execute(
                "SELECT task_id, command, cwd, status, created_at, updated_at, return_code, output_path FROM background_tasks ORDER BY updated_at DESC"
            ).fetchall()
        return [
            {
                "task_id": row[0],
                "command": row[1],
                "cwd": row[2],
                "status": row[3],
                "created_at": row[4],
                "updated_at": row[5],
                "return_code": row[6],
                "output_path": row[7],
            }
            for row in rows
        ]
