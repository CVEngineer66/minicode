from __future__ import annotations

import json
import time
from pathlib import Path


class InputHistoryRepository:
    """File-backed rolling history of user input strings.

    Boundary: bounded to the last `max_entries` entries (default 200) to avoid
    unbounded growth across sessions.
    """

    def __init__(self, history_path: Path, max_entries: int = 200) -> None:
        self.history_path = history_path
        self.max_entries = max_entries

    def load(self) -> list[str]:
        if not self.history_path.exists():
            return []
        try:
            data = json.loads(self.history_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return []
        entries = data.get("entries", [])
        if not isinstance(entries, list):
            return []
        return [str(e) for e in entries]

    def save(self, entries: list[str]) -> None:
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"entries": entries[-self.max_entries :], "updated_at": time.time()}
        self.history_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )

    def append(self, entry: str) -> None:
        entry = entry.strip()
        if not entry:
            return
        entries = self.load()
        if entries and entries[-1] == entry:
            return
        entries.append(entry)
        self.save(entries)
