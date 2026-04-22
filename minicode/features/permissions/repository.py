from __future__ import annotations

import json
import time
from typing import Any

from .types import PatternSet


class DecisionStore:
    """Per-key decision cache backed by SQLite (permission_decisions table).

    Boundary: the `scope` field attached to each decision records whether it was
    persisted session-wide or forever; turn-scoped decisions are kept in-memory.
    """

    def __init__(self, db: Any) -> None:
        self.db = db

    def set(self, key: str, decision: str, detail: dict[str, Any] | None = None) -> None:
        with self.db.connection() as conn:
            conn.execute(
                """
                INSERT INTO permission_decisions(decision_key, decision, created_at, detail_json)
                VALUES(?, ?, ?, ?)
                ON CONFLICT(decision_key) DO UPDATE SET
                    decision = excluded.decision,
                    created_at = excluded.created_at,
                    detail_json = excluded.detail_json
                """,
                (key, decision, time.time(), json.dumps(detail or {}, ensure_ascii=False)),
            )

    def get(self, key: str) -> str | None:
        with self.db.connection() as conn:
            row = conn.execute(
                "SELECT decision FROM permission_decisions WHERE decision_key = ?",
                (key,),
            ).fetchone()
        return str(row[0]) if row else None

    def clear(self, key: str) -> None:
        with self.db.connection() as conn:
            conn.execute("DELETE FROM permission_decisions WHERE decision_key = ?", (key,))


class PatternRepository:
    """Persistent allow/deny pattern set, stored as a single JSON row keyed by `default`."""

    KEY = "default"

    def __init__(self, db: Any) -> None:
        self.db = db
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        with self.db.connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS permission_patterns(
                    pattern_key TEXT PRIMARY KEY,
                    data_json TEXT NOT NULL,
                    updated_at REAL NOT NULL
                )
                """
            )

    def load(self) -> PatternSet:
        with self.db.connection() as conn:
            row = conn.execute(
                "SELECT data_json FROM permission_patterns WHERE pattern_key = ?",
                (self.KEY,),
            ).fetchone()
        if not row:
            return PatternSet()
        try:
            return PatternSet.from_json(json.loads(row[0]))
        except (json.JSONDecodeError, TypeError):
            return PatternSet()

    def save(self, patterns: PatternSet) -> None:
        with self.db.connection() as conn:
            conn.execute(
                """
                INSERT INTO permission_patterns(pattern_key, data_json, updated_at)
                VALUES(?, ?, ?)
                ON CONFLICT(pattern_key) DO UPDATE SET
                    data_json = excluded.data_json,
                    updated_at = excluded.updated_at
                """,
                (self.KEY, json.dumps(patterns.as_json(), ensure_ascii=False), time.time()),
            )
