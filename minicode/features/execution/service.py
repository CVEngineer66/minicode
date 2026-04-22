from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from .isolation import IsolationError, WorktreeIsolator
from .risk import RISK_DESCRIPTIONS, RiskLevel, assess_command_risk
from .sandbox import PathEscapeError, is_sensitive_path, resolve_path_within


class Decision(str, Enum):
    ALLOW = "allow"
    REQUIRE_APPROVAL = "require_approval"
    DENY = "deny"


@dataclass
class ExecutionDecision:
    decision: Decision
    risk: RiskLevel
    reason: str = ""
    details: dict[str, Any] = field(default_factory=dict)


class ExecutionService:
    """Centralized safety gate for tools that touch files or the shell.

    Boundaries (mapped to the plan's Security/Permission tier):
    - `check_path_access` enforces path whitelist against allowed roots.
    - `check_sensitive_file` escalates to REQUIRE_APPROVAL when a sensitive
      filename is involved (e.g. `.env`, `id_rsa`, `*.pem`).
    - `check_command` maps risk level to ALLOW / REQUIRE_APPROVAL / DENY:
        * SAFE, LOW  -> ALLOW
        * MEDIUM     -> ALLOW (runner may isolate instead of approving)
        * HIGH       -> REQUIRE_APPROVAL
        * CRITICAL   -> REQUIRE_APPROVAL (strict)
    - `run_isolated` executes risky commands in an ephemeral git worktree.
    """

    def __init__(
        self,
        allowed_roots: list[Path] | None = None,
        isolator: WorktreeIsolator | None = None,
    ) -> None:
        self.allowed_roots: list[Path] = [Path(r).resolve() for r in (allowed_roots or [])]
        self.isolator = isolator or WorktreeIsolator()

    # --- path/file boundary ---
    def check_path_access(
        self,
        target: str | Path,
        *,
        write: bool = False,
    ) -> ExecutionDecision:
        if not self.allowed_roots:
            return ExecutionDecision(
                Decision.ALLOW, RiskLevel.SAFE, reason="no path whitelist configured"
            )
        try:
            resolved = resolve_path_within(target, self.allowed_roots)
        except PathEscapeError as exc:
            return ExecutionDecision(
                Decision.DENY,
                RiskLevel.HIGH,
                reason=str(exc),
                details={"target": str(target)},
            )
        if is_sensitive_path(resolved):
            risk = RiskLevel.HIGH if write else RiskLevel.MEDIUM
            return ExecutionDecision(
                Decision.REQUIRE_APPROVAL,
                risk,
                reason="sensitive file",
                details={"path": str(resolved)},
            )
        return ExecutionDecision(
            Decision.ALLOW, RiskLevel.LOW if write else RiskLevel.SAFE,
            details={"path": str(resolved)},
        )

    # --- command boundary ---
    def check_command(self, command: str, args: list[str] | None = None) -> ExecutionDecision:
        args = args or []
        risk = assess_command_risk(command, args)
        if risk in (RiskLevel.SAFE, RiskLevel.LOW, RiskLevel.MEDIUM):
            decision = Decision.ALLOW
        else:
            decision = Decision.REQUIRE_APPROVAL
        return ExecutionDecision(
            decision,
            risk,
            reason=RISK_DESCRIPTIONS[risk],
            details={"command": command, "args": list(args)},
        )

    def format_risk_info(self, command: str, args: list[str] | None = None) -> str:
        decision = self.check_command(command, args)
        return (
            "Risk Assessment\n"
            f"{'=' * 50}\n"
            f"Command: {command} {' '.join(args or [])}\n"
            f"Level: {decision.risk.value.upper()}\n"
            f"Decision: {decision.decision.value}\n"
            f"Description: {decision.reason}\n"
        )

    # --- isolated execution ---
    def run_isolated(
        self,
        source_path: Path,
        command: str,
        args: list[str] | None = None,
        timeout: int = 300,
        max_age_seconds: float = 600.0,
    ) -> dict[str, Any]:
        args = args or []
        try:
            ctx = self.isolator.create(source_path, max_age_seconds=max_age_seconds)
        except IsolationError as exc:
            return {"ok": False, "output": f"Isolation unavailable: {exc}"}
        try:
            return self.isolator.execute(ctx.task_id, command, args, timeout=timeout)
        finally:
            self.isolator.cleanup(ctx.task_id)

    # --- lifecycle ---
    def cleanup_expired(self) -> list[str]:
        return self.isolator.cleanup_expired()

    def shutdown(self) -> None:
        self.isolator.cleanup_all()
