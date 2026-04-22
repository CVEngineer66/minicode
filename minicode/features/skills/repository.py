from __future__ import annotations

import time
from pathlib import Path
from typing import Any


class SkillRepository:
    def __init__(self, db: Any, skills_dir: str | Path) -> None:
        self.db = db
        self.skills_dir = Path(skills_dir)
        self.skills_dir.mkdir(parents=True, exist_ok=True)

    def install(self, name: str, source: str, body: str) -> None:
        with self.db.connection() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO skills(name, source, body, installed_at) VALUES(?, ?, ?, ?)",
                (name, source, body, time.time()),
            )
        target_dir = self.skills_dir / name
        target_dir.mkdir(parents=True, exist_ok=True)
        (target_dir / "SKILL.md").write_text(body, encoding="utf-8")

    def list(self) -> list[dict[str, Any]]:
        with self.db.connection() as conn:
            rows = conn.execute("SELECT name, source, installed_at FROM skills ORDER BY name").fetchall()
        return [{"name": row[0], "source": row[1], "installed_at": row[2]} for row in rows]

    def get(self, name: str) -> dict[str, Any] | None:
        with self.db.connection() as conn:
            row = conn.execute("SELECT name, source, body, installed_at FROM skills WHERE name = ?", (name,)).fetchone()
        if not row:
            return None
        return {"name": row[0], "source": row[1], "body": row[2], "installed_at": row[3]}
