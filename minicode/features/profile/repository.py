from __future__ import annotations

from pathlib import Path

from .parser import parse_user_md, serialize_user_md
from .types import UserProfile


class ProfileRepository:
    """File-based USER.md loader/writer for global + project scopes."""

    def __init__(self, global_path: Path, project_path: Path) -> None:
        self.global_path = global_path
        self.project_path = project_path

    def load(self, path: Path) -> UserProfile | None:
        if not path.exists() or not path.is_file():
            return None
        try:
            content = path.read_text(encoding="utf-8")
        except OSError:
            return None
        profile = parse_user_md(content)
        profile.source_path = str(path)
        return profile

    def load_global(self) -> UserProfile | None:
        return self.load(self.global_path)

    def load_project(self) -> UserProfile | None:
        return self.load(self.project_path)

    def save(self, path: Path, profile: UserProfile) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(serialize_user_md(profile), encoding="utf-8")

    def save_global(self, profile: UserProfile) -> None:
        self.save(self.global_path, profile)

    def save_project(self, profile: UserProfile) -> None:
        self.save(self.project_path, profile)

    def delete_global(self) -> bool:
        if self.global_path.exists():
            self.global_path.unlink()
            return True
        return False

    def delete_project(self) -> bool:
        if self.project_path.exists():
            self.project_path.unlink()
            return True
        return False
