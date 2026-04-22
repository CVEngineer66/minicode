from __future__ import annotations

from enum import Enum


class RiskLevel(str, Enum):
    SAFE = "safe"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


CRITICAL_COMMANDS = frozenset(
    {
        "rm",
        "shred",
        "dd",
        "mkfs",
        "fdisk",
        "format",
        "dropdb",
        "drop",
        "truncate",
    }
)

HIGH_COMMANDS = frozenset(
    {
        "sudo",
        "su",
        "chmod",
        "chown",
        "mount",
        "umount",
        "systemctl",
        "service",
        "brew",
        "apt",
        "yum",
        "dnf",
    }
)

MEDIUM_COMMANDS = frozenset(
    {
        "git",
        "npm",
        "pip",
        "cargo",
        "go",
        "make",
        "cmake",
        "docker",
        "docker-compose",
        "kubectl",
    }
)

LOW_COMMANDS = frozenset({"echo", "cat", "tee", "cp", "mv", "mkdir", "touch"})
SAFE_COMMANDS = frozenset(
    {
        "ls",
        "pwd",
        "cat",
        "head",
        "tail",
        "wc",
        "grep",
        "find",
        "which",
        "whoami",
        "date",
        "echo",
        "df",
        "du",
        "uname",
    }
)

DESTRUCTIVE_FLAGS = frozenset({"-rf", "-fr", "--force", "--recursive", "--no-preserve-root"})


def assess_command_risk(command: str, args: list[str] | None = None) -> RiskLevel:
    args = args or []
    cmd_base = command.lower().replace("\\", "/").split("/")[-1]
    if cmd_base in CRITICAL_COMMANDS:
        return RiskLevel.CRITICAL
    if any(flag in args for flag in DESTRUCTIVE_FLAGS):
        return RiskLevel.CRITICAL
    if cmd_base in HIGH_COMMANDS:
        return RiskLevel.HIGH
    if cmd_base in MEDIUM_COMMANDS:
        return RiskLevel.MEDIUM
    if cmd_base in LOW_COMMANDS:
        return RiskLevel.LOW
    if cmd_base in SAFE_COMMANDS:
        return RiskLevel.SAFE
    return RiskLevel.MEDIUM


RISK_DESCRIPTIONS: dict[RiskLevel, str] = {
    RiskLevel.SAFE: "Read-only operation, no side effects",
    RiskLevel.LOW: "Minor writes, low risk of data loss",
    RiskLevel.MEDIUM: "Development operation, isolated execution recommended",
    RiskLevel.HIGH: "System operation, requires approval",
    RiskLevel.CRITICAL: "Destructive operation, requires strict isolation",
}
