from __future__ import annotations

import json
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from langgraph.checkpoint.sqlite import SqliteSaver


class DatabaseManager:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._setup()

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        conn = self.connect()
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    @contextmanager
    def checkpointer(self) -> Iterator[SqliteSaver]:
        with SqliteSaver.from_conn_string(str(self.db_path)) as saver:
            saver.setup()
            yield saver

    def _setup(self) -> None:
        with self.connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS session_meta (
                    thread_id TEXT PRIMARY KEY,
                    workspace TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    model TEXT NOT NULL,
                    title TEXT NOT NULL DEFAULT ''
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS migration_ledger (
                    source_key TEXT PRIMARY KEY,
                    migrated_at REAL NOT NULL,
                    detail_json TEXT NOT NULL DEFAULT '{}'
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memory_entries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    scope TEXT NOT NULL,
                    workspace TEXT NOT NULL,
                    entry_key TEXT NOT NULL,
                    content TEXT NOT NULL,
                    tags_json TEXT NOT NULL DEFAULT '[]',
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    UNIQUE(scope, workspace, entry_key)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS permission_decisions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    decision_key TEXT UNIQUE NOT NULL,
                    decision TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    detail_json TEXT NOT NULL DEFAULT '{}'
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS task_items (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    workspace TEXT NOT NULL,
                    title TEXT NOT NULL,
                    status TEXT NOT NULL,
                    note TEXT NOT NULL DEFAULT '',
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS task_graph_nodes (
                    node_id TEXT PRIMARY KEY,
                    workspace TEXT NOT NULL,
                    title TEXT NOT NULL,
                    status TEXT NOT NULL,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS task_graph_edges (
                    parent_id TEXT NOT NULL,
                    child_id TEXT NOT NULL,
                    PRIMARY KEY(parent_id, child_id)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS background_tasks (
                    task_id TEXT PRIMARY KEY,
                    command TEXT NOT NULL,
                    cwd TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    return_code INTEGER,
                    output_path TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS mcp_servers (
                    name TEXT PRIMARY KEY,
                    command TEXT NOT NULL,
                    args_json TEXT NOT NULL DEFAULT '[]',
                    env_json TEXT NOT NULL DEFAULT '{}',
                    cwd TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS skills (
                    name TEXT PRIMARY KEY,
                    source TEXT NOT NULL,
                    body TEXT NOT NULL,
                    installed_at REAL NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS collaboration_agents (
                    name TEXT PRIMARY KEY,
                    description TEXT NOT NULL,
                    skills_json TEXT NOT NULL DEFAULT '[]',
                    status TEXT NOT NULL DEFAULT 'idle'
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS collaboration_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    channel TEXT NOT NULL,
                    sender TEXT NOT NULL,
                    recipient TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    status TEXT NOT NULL DEFAULT 'posted'
                )
                """
            )

    def write_ledger(self, source_key: str, detail: dict[str, Any] | None = None) -> None:
        with self.connection() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO migration_ledger(source_key, migrated_at, detail_json) VALUES(?, ?, ?)",
                (source_key, time.time(), json.dumps(detail or {}, ensure_ascii=False)),
            )

    def has_ledger(self, source_key: str) -> bool:
        with self.connection() as conn:
            row = conn.execute("SELECT 1 FROM migration_ledger WHERE source_key = ?", (source_key,)).fetchone()
        return row is not None
