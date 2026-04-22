from __future__ import annotations

import re
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any


# ---------------------------------------------------------------------------
# Modes, risk levels, and pattern tables
# ---------------------------------------------------------------------------


class PermissionMode(str, Enum):
    DEFAULT = "default"
    AUTO = "auto"
    BYPASS = "bypass"
    PLAN = "plan"


class AutoRiskLevel(str, Enum):
    SAFE = "safe"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    DANGEROUS = "dangerous"


@dataclass
class RiskAssessment:
    level: AutoRiskLevel
    tool_name: str
    action: str  # approve | prompt | block
    reason: str
    safe_alternative: str | None = None


SAFE_TOOLS: frozenset[str] = frozenset({"read_file", "list_files", "grep_files", "load_skill", "file_tree"})
LOW_RISK_TOOLS: frozenset[str] = frozenset({"run_command"})
MEDIUM_RISK_TOOLS: frozenset[str] = frozenset({"write_file", "edit_file", "patch_file", "modify_file"})

HIGH_RISK_COMMANDS: tuple[str, ...] = (
    "rm -rf",
    "rm -r",
    "git reset --hard",
    "git clean",
    "git push --force",
    "sudo",
    "chmod -R",
    "chown -R",
    "del /s",
    "del /q",
    "rmdir /s",
    "rd /s",
    "icacls",
    "takeown",
    "reg delete",
    "format",
)

DANGEROUS_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"rm\s+-rf\s+/",
        r"chmod\s+777",
        r"curl.*\|\s*sh",
        r"wget.*\|\s*sh",
        r"mkfs",
        r"dd\s+if=",
        r"del\s+/[sfq].*[\\]",
        r"rmdir\s+/s\s+/q",
        r"rd\s+/s\s+/q",
        r"format\s+[a-zA-Z]:",
        r"powershell.*\biex\b",
        r"powershell.*Invoke-Expression",
        r"iwr.*\|\s*iex",
        r"reg\s+delete\s+HKLM",
    )
)

INJECTION_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"ignore\s+(all\s+)?(previous|prior)\s+(instructions|rules|prompts)",
        r"(system|developer)\s*:\s*",
        r"\[?ignore\s+security\]?",
        r"(bypass|skip|override)\s+(permissions|safety|restrictions)",
        r"(execute|run)\s+(this|following)\s+code\s*:",
        r"ignore\s+(all|your)\s+instructions",
    )
)

UNSAFE_OUTPUT_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"rm\s+-rf",
        r"sudo\s+",
        r"chmod\s+777",
        r"del\s+/[sfq]",
        r"rmdir\s+/s",
        r"rd\s+/s",
        r"format\s+[a-zA-Z]:",
        r"DROP\s+TABLE",
        r"DELETE\s+FROM.*WHERE\s+1\s*=\s*1",
    )
)

SENSITIVE_FILE_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p)
    for p in (
        r"\.env",
        r"\.git[/\\]",
        r"node_modules[/\\]",
        r"__pycache__[/\\]",
        r"\.pyc$",
    )
)


# ---------------------------------------------------------------------------
# Risk classifier
# ---------------------------------------------------------------------------


class AutoModeChecker:
    """Risk classifier used by `auto` permission mode.

    Boundaries:
    - PLAN mode allows only SAFE_TOOLS.
    - BYPASS mode approves everything (operator-acknowledged).
    - DEFAULT mode always prompts.
    - AUTO mode dispatches per-tool + command/pattern checks; DANGEROUS hits are
      hard-blocked.
    """

    def __init__(self, mode: PermissionMode = PermissionMode.DEFAULT) -> None:
        self.mode = mode

    def set_mode(self, mode: PermissionMode) -> None:
        self.mode = mode

    def assess(self, tool_name: str, tool_input: dict[str, Any]) -> RiskAssessment:
        if self.mode == PermissionMode.BYPASS:
            return RiskAssessment(
                AutoRiskLevel.DANGEROUS,
                tool_name,
                "approve",
                "Bypass mode: all permissions skipped",
            )
        if self.mode == PermissionMode.PLAN:
            if tool_name in SAFE_TOOLS:
                return RiskAssessment(
                    AutoRiskLevel.SAFE, tool_name, "approve", "Plan mode: read-only tool"
                )
            return RiskAssessment(
                AutoRiskLevel.HIGH, tool_name, "block", "Plan mode: execution not allowed"
            )
        if self.mode == PermissionMode.DEFAULT:
            return RiskAssessment(
                AutoRiskLevel.MEDIUM, tool_name, "prompt", "Default mode: approval required"
            )
        return self._auto(tool_name, tool_input)

    # --- auto-mode internals ---
    def _auto(self, tool_name: str, tool_input: dict[str, Any]) -> RiskAssessment:
        if tool_name in SAFE_TOOLS:
            return RiskAssessment(
                AutoRiskLevel.SAFE, tool_name, "approve", f"{tool_name} is read-only"
            )
        if tool_name == "run_command":
            return self._assess_command(tool_input)
        if tool_name in MEDIUM_RISK_TOOLS:
            return self._assess_file_edit(tool_name, tool_input)
        return RiskAssessment(
            AutoRiskLevel.MEDIUM, tool_name, "prompt", f"Unknown tool '{tool_name}'"
        )

    def _assess_command(self, tool_input: dict[str, Any]) -> RiskAssessment:
        command = tool_input.get("command", "")
        if isinstance(command, list):
            command = " ".join(str(p) for p in command)
        command_str = str(command)
        for pattern in DANGEROUS_PATTERNS:
            if pattern.search(command_str):
                return RiskAssessment(
                    AutoRiskLevel.DANGEROUS,
                    "run_command",
                    "block",
                    f"Dangerous pattern: {pattern.pattern}",
                )
        for risky in HIGH_RISK_COMMANDS:
            if risky in command_str:
                return RiskAssessment(
                    AutoRiskLevel.HIGH,
                    "run_command",
                    "prompt",
                    f"High-risk command: '{risky}'",
                    safe_alternative=f"Consider a safer alternative to '{risky}'",
                )
        return RiskAssessment(
            AutoRiskLevel.LOW, "run_command", "approve", "Command appears safe"
        )

    def _assess_file_edit(self, tool_name: str, tool_input: dict[str, Any]) -> RiskAssessment:
        path = str(tool_input.get("path", ""))
        for pattern in SENSITIVE_FILE_PATTERNS:
            if pattern.search(path):
                return RiskAssessment(
                    AutoRiskLevel.HIGH,
                    tool_name,
                    "prompt",
                    f"Modifying sensitive file: {path}",
                )
        return RiskAssessment(
            AutoRiskLevel.MEDIUM, tool_name, "prompt", "File modification requires approval"
        )

    # --- input / output layer helpers ---
    @staticmethod
    def detect_prompt_injection(user_input: str) -> tuple[bool, str]:
        for pattern in INJECTION_PATTERNS:
            if pattern.search(user_input):
                return True, f"Potential prompt injection: {pattern.pattern}"
        return False, ""

    @staticmethod
    def classify_output_safety(output: str) -> tuple[bool, str]:
        for pattern in UNSAFE_OUTPUT_PATTERNS:
            if pattern.search(output):
                return True, f"Unsafe operation in output: {pattern.pattern}"
        return False, ""


# ---------------------------------------------------------------------------
# Stateful facade
# ---------------------------------------------------------------------------


@dataclass
class ModeState:
    mode: PermissionMode = PermissionMode.DEFAULT
    mode_changed_at: float = 0.0
    mode_changed_by: str = "user"
    auto_approve_count: int = 0
    prompt_count: int = 0
    block_count: int = 0

    def record(self, action: str) -> None:
        if action == "approve":
            self.auto_approve_count += 1
        elif action == "prompt":
            self.prompt_count += 1
        elif action == "block":
            self.block_count += 1

    def stats(self) -> dict[str, Any]:
        total = self.auto_approve_count + self.prompt_count + self.block_count
        return {
            "mode": self.mode.value,
            "auto_approve": self.auto_approve_count,
            "prompt": self.prompt_count,
            "block": self.block_count,
            "auto_approve_rate": (self.auto_approve_count / total) if total else 0.0,
        }


_MODE_MESSAGES = {
    PermissionMode.DEFAULT: "Default mode: all actions require approval",
    PermissionMode.AUTO: "Auto mode: safe operations auto-approved",
    PermissionMode.BYPASS: "BYPASS MODE: all permissions skipped (dangerous)",
    PermissionMode.PLAN: "Plan mode: read-only operations only",
}


class AutoModeService:
    """Stateful facade combining AutoModeChecker + ModeState.

    Exposed to tools as the gate before real execution. Callers should:
      1) inspect `detect_prompt_injection(user_input)` at ingress
      2) call `assess(tool_name, args)` and respect the `action` field
      3) call `record(action)` for stats
      4) optionally `classify_output_safety(text)` at egress
    """

    def __init__(self, mode: PermissionMode = PermissionMode.DEFAULT) -> None:
        self.state = ModeState(mode=mode, mode_changed_at=time.time())
        self.checker = AutoModeChecker(mode=mode)

    def set_mode(self, mode: PermissionMode | str, *, changed_by: str = "user") -> str:
        m = mode if isinstance(mode, PermissionMode) else PermissionMode(str(mode))
        self.state.mode = m
        self.state.mode_changed_at = time.time()
        self.state.mode_changed_by = changed_by
        self.checker.set_mode(m)
        return _MODE_MESSAGES.get(m, f"Mode changed to {m.value}")

    def get_mode(self) -> PermissionMode:
        return self.state.mode

    def assess(self, tool_name: str, tool_input: dict[str, Any]) -> RiskAssessment:
        return self.checker.assess(tool_name, tool_input)

    def record(self, action: str) -> None:
        self.state.record(action)

    def stats(self) -> dict[str, Any]:
        return self.state.stats()

    @staticmethod
    def detect_prompt_injection(user_input: str) -> tuple[bool, str]:
        return AutoModeChecker.detect_prompt_injection(user_input)

    @staticmethod
    def classify_output_safety(output: str) -> tuple[bool, str]:
        return AutoModeChecker.classify_output_safety(output)
