from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from .metadata import (
    LoadedSkill,
    SkillMetadata,
    content_hash,
    extract_description,
    extract_metadata,
)


class SkillConflictError(RuntimeError):
    pass


class SkillService:
    """SQLite-backed skill catalog plus filesystem discovery across multiple roots.

    Boundaries (external integration tier):
    - `trusted_hashes` — if non-empty, only matching content hashes may be
      installed. This lets the operator restrict to vetted skills.
    - `install_from_path` detects version/content conflicts against the catalog
      and refuses silent overwrites unless `overwrite=True`.
    - Failures on a single skill never affect others (discover is tolerant).
    """

    def __init__(
        self,
        repository: Any,
        *,
        discovery_roots: list[tuple[Path, str]] | None = None,
        trusted_hashes: set[str] | None = None,
    ) -> None:
        self.repository = repository
        self.discovery_roots: list[tuple[Path, str]] = list(discovery_roots or [])
        self.trusted_hashes: set[str] = set(trusted_hashes or [])

    # --- low-level catalog ---
    def install_skill(self, name: str, body: str, source: str, *, overwrite: bool = False) -> SkillMetadata:
        if self.trusted_hashes and content_hash(body) not in self.trusted_hashes:
            raise SkillConflictError(
                f"Skill '{name}' not in trusted-hash allowlist"
            )
        existing = self.repository.get(name)
        if existing and not overwrite and existing.get("body") != body:
            raise SkillConflictError(
                f"Skill '{name}' already installed with different content; pass overwrite=True to replace"
            )
        self.repository.install(name, source, body)
        return extract_metadata(name, body, source=source)

    def list_skills(self) -> list[dict[str, Any]]:
        return self.repository.list()

    def load_skill(self, name: str) -> str:
        skill = self.repository.get(name)
        if not skill:
            raise KeyError(f"Unknown skill: {name}")
        return str(skill["body"])

    def metadata(self, name: str) -> SkillMetadata | None:
        skill = self.repository.get(name)
        if not skill:
            return None
        body = str(skill["body"])
        return extract_metadata(name, body, source=str(skill.get("source", "")))

    # --- filesystem discovery ---
    def discover(self) -> list[LoadedSkill]:
        """Discover SKILL.md files across configured roots, deduping by name."""
        seen: dict[str, LoadedSkill] = {}
        for root, source in self.discovery_roots:
            if not root.exists():
                continue
            try:
                entries = list(root.iterdir())
            except OSError:
                continue
            for entry in entries:
                try:
                    if not entry.is_dir():
                        continue
                except OSError:
                    continue
                skill_path = entry / "SKILL.md"
                if not skill_path.exists():
                    continue
                try:
                    body = skill_path.read_text(encoding="utf-8")
                except OSError:
                    continue
                meta = extract_metadata(entry.name, body, source=source, path=str(skill_path))
                loaded = LoadedSkill(
                    name=meta.name,
                    description=meta.description,
                    path=meta.path,
                    source=source,
                    content=body,
                    version=meta.version,
                    content_hash=meta.content_hash,
                )
                seen.setdefault(meta.name, loaded)
        return list(seen.values())

    def install_from_path(
        self,
        path: str | Path,
        *,
        name: str | None = None,
        overwrite: bool = False,
    ) -> SkillMetadata:
        skill_path = Path(path)
        if skill_path.is_dir():
            body_path = skill_path / "SKILL.md"
            inferred = skill_path.name
        elif skill_path.name.upper() == "SKILL.MD":
            body_path = skill_path
            inferred = skill_path.parent.name
        else:
            raise FileNotFoundError(f"No SKILL.md in {skill_path}")
        if not body_path.exists():
            raise FileNotFoundError(f"SKILL.md not found: {body_path}")
        body = body_path.read_text(encoding="utf-8")
        skill_name = (name or inferred).strip()
        if not skill_name:
            raise ValueError("Skill name cannot be empty")
        return self.install_skill(skill_name, body, str(body_path), overwrite=overwrite)

    def remove(self, name: str) -> bool:
        """Physically remove the SKILL.md from the managed skills_dir, if present."""
        skills_dir = getattr(self.repository, "skills_dir", None)
        if skills_dir is None:
            return False
        target = Path(skills_dir) / name
        if not target.exists():
            return False
        try:
            shutil.rmtree(target)
            return True
        except OSError:
            return False

    # --- boundary helpers ---
    def is_trusted(self, body: str) -> bool:
        if not self.trusted_hashes:
            return True
        return content_hash(body) in self.trusted_hashes

    def check_conflict(self, name: str, body: str) -> str | None:
        """Return a reason if installing would conflict; None otherwise."""
        existing = self.repository.get(name)
        if existing is None:
            return None
        if existing.get("body") == body:
            return None
        return (
            f"skill '{name}' already installed at {existing.get('source', 'unknown')} with different content"
        )
