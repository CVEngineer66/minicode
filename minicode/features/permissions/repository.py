from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from .types import PatternSet


class _PermissionJsonFile:
    """Workspace-scoped JSON storage for permission decisions and patterns."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def read(self) -> dict[str, Any]:
        if not self.path.exists():
            return self._default_payload()
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8") or "{}")
        except (OSError, json.JSONDecodeError):
            return self._default_payload()
        if not isinstance(payload, dict):
            return self._default_payload()
        data = self._default_payload()
        decisions = payload.get("decisions")
        if isinstance(decisions, dict):
            data["decisions"] = decisions
        patterns = payload.get("patterns")
        if isinstance(patterns, dict):
            data["patterns"] = patterns
        return data

    def write(self, payload: dict[str, Any]) -> None:
        merged = self._default_payload()
        merged["decisions"] = payload.get("decisions") or {}
        merged["patterns"] = payload.get("patterns") or {}
        tmp_path = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp_path.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp_path.replace(self.path)

    @staticmethod
    def _default_payload() -> dict[str, Any]:
        return {
            "version": 1,
            "decisions": {},
            "patterns": PatternSet().as_json(),
        }


class DecisionStore:
    """Per-key decision cache backed by workspace-local ``permissions.json``."""

    def __init__(self, path: str | Path) -> None:
        self._store = _PermissionJsonFile(path)

    def set(self, key: str, decision: str, detail: dict[str, Any] | None = None) -> None:
        payload = self._store.read()
        decisions = payload.setdefault("decisions", {})
        decisions[key] = {
            "decision": decision,
            "updated_at": time.time(),
            "detail": detail or {},
        }
        self._store.write(payload)

    def get(self, key: str) -> str | None:
        payload = self._store.read()
        entry = payload.get("decisions", {}).get(key)
        if isinstance(entry, dict):
            decision = entry.get("decision")
            return str(decision) if decision else None
        if isinstance(entry, str):
            return entry
        return None

    def clear(self, key: str) -> None:
        payload = self._store.read()
        decisions = payload.get("decisions", {})
        if isinstance(decisions, dict) and key in decisions:
            decisions.pop(key, None)
            self._store.write(payload)


class PatternRepository:
    """Workspace-scoped allow/deny patterns stored in ``permissions.json``."""

    def __init__(self, path: str | Path) -> None:
        self._store = _PermissionJsonFile(path)

    def load(self) -> PatternSet:
        payload = self._store.read()
        raw = payload.get("patterns")
        if not isinstance(raw, dict):
            return PatternSet()
        try:
            return PatternSet.from_json(raw)
        except (TypeError, ValueError):
            return PatternSet()

    def save(self, patterns: PatternSet) -> None:
        payload = self._store.read()
        payload["patterns"] = patterns.as_json()
        self._store.write(payload)
