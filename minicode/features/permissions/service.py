from __future__ import annotations

from typing import Any

from minicode.core.types import PermissionPolicy, ToolCapability

from .patterns import (
    classify_dangerous_command,
    match_any_path,
    match_command,
)
from .repository import DecisionStore, PatternRepository
from .types import ApprovalRequest, PatternSet, PermissionDecision


class PolicyEngine:
    """Decide whether a tool call needs user approval.

    Integrates with PatternSet (persisted allow/deny globs) and dangerous
    command classification. Callers supply tool capability + policy (from the
    ToolSpec) plus the current mode (`default` / `plan` / `bypass`).
    """

    def __init__(self, patterns: PatternSet | None = None) -> None:
        self.patterns: PatternSet = patterns or PatternSet()

    def build_request(
        self,
        *,
        tool_name: str,
        arguments: dict[str, Any],
        capability: ToolCapability,
        policy: PermissionPolicy,
        mode: str,
    ) -> ApprovalRequest | None:
        decision_key = self.decision_key(tool_name, arguments)
        if mode == "bypass":
            return None

        # Deny-first: hard deny patterns short-circuit to deny_once via broker.
        if self._is_hard_denied(tool_name, arguments):
            return ApprovalRequest(
                kind=policy.kind,
                summary=f"Denied by pattern: {tool_name}",
                details=[f"Arguments: {arguments}"],
                scope=decision_key,
                choices=[{"key": "n", "label": "Acknowledged", "decision": "deny_once"}],
            )

        # Pre-approved by pattern allowlist
        if self._is_pattern_allowed(tool_name, arguments):
            return None

        dangerous_reason = self._dangerous_reason(tool_name, arguments)
        details: list[str] = [f"Arguments: {arguments}"]
        if dangerous_reason:
            details.insert(0, dangerous_reason)

        if mode == "plan" and (capability.shell or capability.writes_files or capability.network):
            return ApprovalRequest(
                kind=policy.kind,
                summary=f"Plan mode requires approval for {tool_name}",
                details=details,
                scope=decision_key,
                choices=self.default_choices(),
            )

        requires = (
            policy.always_require_approval
            or capability.writes_files
            or capability.shell
            or capability.network
            or capability.task
            or dangerous_reason is not None
        )
        if requires:
            return ApprovalRequest(
                kind=policy.kind,
                summary=f"Approve {tool_name}",
                details=details,
                scope=decision_key,
                choices=self.default_choices(),
            )
        return None

    def default_choices(self) -> list[dict[str, str]]:
        return [
            {"key": "y", "label": "Allow once", "decision": "allow_once"},
            {"key": "t", "label": "Allow for this turn", "decision": "allow_turn"},
            {"key": "a", "label": "Always allow", "decision": "allow_always"},
            {"key": "n", "label": "Deny", "decision": "deny_once"},
        ]

    def decision_key(self, tool_name: str, arguments: dict[str, Any]) -> str:
        stable = "|".join(f"{k}={arguments[k]}" for k in sorted(arguments))
        return f"{tool_name}|{stable}"

    # --- pattern helpers ---
    def _paths_in_args(self, arguments: dict[str, Any]) -> list[str]:
        paths: list[str] = []
        for key in ("path", "filePath", "file_path", "target", "source", "destination"):
            value = arguments.get(key)
            if isinstance(value, str) and value:
                paths.append(value)
        return paths

    def _command_in_args(self, arguments: dict[str, Any]) -> tuple[str | None, list[str]]:
        command = arguments.get("command")
        if isinstance(command, str) and command:
            args = arguments.get("args") or []
            if isinstance(args, list):
                return command, [str(a) for a in args]
            parts = command.split()
            return parts[0], parts[1:]
        return None, []

    def _is_hard_denied(self, tool_name: str, arguments: dict[str, Any]) -> bool:
        for p in self._paths_in_args(arguments):
            if match_any_path(p, self.patterns.denied_directories) or match_any_path(
                p, self.patterns.denied_edits
            ):
                return True
        command, args = self._command_in_args(arguments)
        if command and match_command(command, args, self.patterns.denied_commands):
            return True
        return False

    def _is_pattern_allowed(self, tool_name: str, arguments: dict[str, Any]) -> bool:
        paths = self._paths_in_args(arguments)
        if paths and all(
            match_any_path(p, self.patterns.allowed_directories)
            or match_any_path(p, self.patterns.allowed_edits)
            for p in paths
        ):
            return True
        command, args = self._command_in_args(arguments)
        if command and match_command(command, args, self.patterns.allowed_commands):
            return True
        return False

    def _dangerous_reason(self, tool_name: str, arguments: dict[str, Any]) -> str | None:
        command, args = self._command_in_args(arguments)
        if command:
            return classify_dangerous_command(command, args)
        return None


class ApprovalBroker:
    """Applies persisted and in-memory decisions on top of PolicyEngine requests.

    Scope rules:
    - allow_always / deny_always → persisted via DecisionStore
    - allow_turn / deny_turn → in-memory per-turn cache (cleared by begin_turn)
    - allow_all_turn → blanket allow for the rest of the turn
    """

    def __init__(
        self,
        policy_engine: PolicyEngine,
        store: DecisionStore,
        pattern_repository: PatternRepository | None = None,
    ) -> None:
        self.policy_engine = policy_engine
        self.store = store
        self.pattern_repository = pattern_repository
        self._turn_allow: set[str] = set()
        self._turn_deny: set[str] = set()
        self._allow_all_turn = False
        if pattern_repository is not None:
            self.policy_engine.patterns = pattern_repository.load()

    # --- turn lifecycle ---
    def begin_turn(self) -> None:
        self._turn_allow.clear()
        self._turn_deny.clear()
        self._allow_all_turn = False

    def end_turn(self) -> None:
        self.begin_turn()

    # --- decision cache ---
    def cached_decision(self, key: str) -> PermissionDecision | None:
        if self._allow_all_turn:
            return "allow_all_turn"
        if key in self._turn_deny:
            return "deny_once"
        if key in self._turn_allow:
            return "allow_turn"
        raw = self.store.get(key)
        return raw  # type: ignore[return-value]

    def store_decision(
        self,
        key: str,
        decision: PermissionDecision | str,
        detail: dict[str, Any] | None = None,
    ) -> None:
        if decision == "allow_always" or decision == "deny_always":
            self.store.set(key, decision, detail)
        elif decision == "allow_turn":
            self._turn_allow.add(key)
        elif decision == "deny_turn" or decision == "deny_once":
            self._turn_deny.add(key)
        elif decision == "allow_all_turn":
            self._allow_all_turn = True

    # --- pattern mutation ---
    def add_allowed_directory(self, path: str) -> None:
        self.policy_engine.patterns.allowed_directories.add(path)
        self._persist_patterns()

    def add_allowed_command(self, pattern: str) -> None:
        self.policy_engine.patterns.allowed_commands.add(pattern)
        self._persist_patterns()

    def add_denied_command(self, pattern: str) -> None:
        self.policy_engine.patterns.denied_commands.add(pattern)
        self._persist_patterns()

    def _persist_patterns(self) -> None:
        if self.pattern_repository is not None:
            self.pattern_repository.save(self.policy_engine.patterns)

    # --- introspection ---
    def summary(self) -> dict[str, Any]:
        return {
            "turn_allow_count": len(self._turn_allow),
            "turn_deny_count": len(self._turn_deny),
            "allow_all_turn": self._allow_all_turn,
            "patterns": self.policy_engine.patterns.as_json(),
        }
