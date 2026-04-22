from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class SkillMetadata:
    name: str
    description: str
    version: str = ""
    source: str = ""
    path: str = ""
    content_hash: str = ""


@dataclass
class LoadedSkill:
    name: str
    description: str
    path: str
    source: str
    content: str
    version: str = ""
    content_hash: str = ""


_FRONTMATTER_RE = re.compile(r"\A---\n(.*?)\n---\n", re.DOTALL)


def extract_description(markdown: str) -> str:
    normalized = markdown.replace("\r\n", "\n")
    stripped = _FRONTMATTER_RE.sub("", normalized, count=1)
    for block in (b.strip() for b in stripped.split("\n\n") if b.strip()):
        if block.startswith("#"):
            continue
        for line in (p.strip() for p in block.split("\n")):
            if line and not line.startswith("#"):
                return line.replace("`", "")
    return "No description provided."


def parse_frontmatter(markdown: str) -> dict[str, str]:
    """Extract `key: value` entries from a leading YAML-like block."""
    match = _FRONTMATTER_RE.match(markdown.replace("\r\n", "\n"))
    if not match:
        return {}
    result: dict[str, str] = {}
    for line in match.group(1).splitlines():
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        result[key.strip().lower()] = value.strip()
    return result


def content_hash(markdown: str) -> str:
    """Stable short hash for trust lists and conflict detection."""
    return hashlib.sha256(markdown.encode("utf-8", errors="replace")).hexdigest()[:16]


def extract_metadata(name: str, markdown: str, *, source: str = "", path: str = "") -> SkillMetadata:
    fm = parse_frontmatter(markdown)
    return SkillMetadata(
        name=name,
        description=fm.get("description") or extract_description(markdown),
        version=fm.get("version", ""),
        source=source,
        path=path,
        content_hash=content_hash(markdown),
    )
