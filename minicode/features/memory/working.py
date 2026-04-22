from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from minicode.features.context.token_estimator import estimate_tokens


@dataclass
class WorkingMemoryEntry:
    content: str
    entry_type: str = "active_task"
    created_at: float = field(default_factory=time.time)
    expires_at: float | None = None
    importance: float = 1.0

    def is_expired(self) -> bool:
        return self.expires_at is not None and time.time() > self.expires_at

    def token_count(self) -> int:
        return estimate_tokens(self.content)


class WorkingMemoryTracker:
    """Runtime-only store of high-priority content protected from compaction."""

    def __init__(self, max_entries: int = 15, max_tokens: int = 4000) -> None:
        self._entries: list[WorkingMemoryEntry] = []
        self.max_entries = max_entries
        self.max_tokens = max_tokens

    def add(
        self,
        content: str,
        entry_type: str = "active_task",
        ttl_seconds: float | None = None,
        importance: float = 1.0,
    ) -> WorkingMemoryEntry:
        expires = None if ttl_seconds is None else time.time() + ttl_seconds
        entry = WorkingMemoryEntry(content, entry_type, expires_at=expires, importance=importance)
        self._entries.append(entry)
        self._enforce()
        return entry

    def clear_expired(self) -> int:
        before = len(self._entries)
        self._entries = [e for e in self._entries if not e.is_expired()]
        return before - len(self._entries)

    def protected_content(self) -> list[str]:
        self.clear_expired()
        return [e.content for e in self._entries]

    def protected_tokens(self) -> int:
        return sum(e.token_count() for e in self._entries if not e.is_expired())

    def stats(self) -> dict[str, Any]:
        self.clear_expired()
        return {
            "entries": len(self._entries),
            "max_entries": self.max_entries,
            "protected_tokens": self.protected_tokens(),
            "max_tokens": self.max_tokens,
        }

    def _enforce(self) -> None:
        self.clear_expired()
        while self.protected_tokens() > self.max_tokens and self._entries:
            self._entries.sort(key=lambda e: e.importance)
            self._entries.pop(0)
        while len(self._entries) > self.max_entries and self._entries:
            self._entries.sort(key=lambda e: e.importance)
            self._entries.pop(0)


@dataclass
class ContinuityMarker:
    marker_type: str
    description: str
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)


class ContinuityManager:
    """Preserves key conversation flow points across compaction boundaries."""

    def __init__(self, max_markers: int = 20) -> None:
        self._markers: list[ContinuityMarker] = []
        self.max_markers = max_markers

    def add(
        self,
        marker_type: str,
        description: str,
        metadata: dict[str, Any] | None = None,
    ) -> ContinuityMarker:
        marker = ContinuityMarker(marker_type, description, metadata=metadata or {})
        self._markers.append(marker)
        if len(self._markers) > self.max_markers:
            self._markers = self._markers[-self.max_markers :]
        return marker

    def recent(self, limit: int = 10) -> list[ContinuityMarker]:
        return self._markers[-limit:]

    def since(self, timestamp: float) -> list[ContinuityMarker]:
        return [m for m in self._markers if m.timestamp > timestamp]
