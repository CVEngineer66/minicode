from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

PermissionDecision = Literal[
    "allow_once",
    "allow_always",
    "allow_turn",
    "allow_all_turn",
    "allow_command_pattern",
    "allow_directory_pattern",
    "deny_once",
    "deny_always",
    "deny_with_feedback",
]


@dataclass(slots=True)
class ApprovalRequest:
    kind: str
    summary: str
    details: list[str] = field(default_factory=list)
    scope: str = ""
    choices: list[dict[str, Any]] = field(default_factory=list)


@dataclass(slots=True)
class PatternSet:
    """Persistent allow/deny patterns shared across sessions."""

    allowed_directories: set[str] = field(default_factory=set)
    denied_directories: set[str] = field(default_factory=set)
    allowed_commands: set[str] = field(default_factory=set)
    denied_commands: set[str] = field(default_factory=set)
    allowed_edits: set[str] = field(default_factory=set)
    denied_edits: set[str] = field(default_factory=set)

    def as_json(self) -> dict[str, list[str]]:
        return {
            "allowedDirectoryPrefixes": sorted(self.allowed_directories),
            "deniedDirectoryPrefixes": sorted(self.denied_directories),
            "allowedCommandPatterns": sorted(self.allowed_commands),
            "deniedCommandPatterns": sorted(self.denied_commands),
            "allowedEditPatterns": sorted(self.allowed_edits),
            "deniedEditPatterns": sorted(self.denied_edits),
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "PatternSet":
        return cls(
            allowed_directories=set(data.get("allowedDirectoryPrefixes", [])),
            denied_directories=set(data.get("deniedDirectoryPrefixes", [])),
            allowed_commands=set(data.get("allowedCommandPatterns", [])),
            denied_commands=set(data.get("deniedCommandPatterns", [])),
            allowed_edits=set(data.get("allowedEditPatterns", [])),
            denied_edits=set(data.get("deniedEditPatterns", [])),
        )
