from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

from minicode.core.messages import make_tool_message
from minicode.core.types import ToolContext, ToolResult
from minicode.features.permissions.graph_adapter import ensure_tool_allowed


@dataclass(slots=True)
class ToolBatchResult:
    tool_messages: list[Any]
    saw_tool_result: bool
    tool_error_count: int
    await_user: bool = False
    final_text: str | None = None


_PATH_KEYS = ("path", "filePath", "file_path", "target", "source", "destination")


def _extract_path(args: dict[str, Any]) -> str | None:
    for key in _PATH_KEYS:
        v = args.get(key)
        if isinstance(v, str) and v:
            return v
    return None


class ToolGraphAdapter:
    """Executes LangGraph tool calls with concurrency, permission, and execution-boundary gates.

    Boundaries (before running any tool):
    1. Schema validation (spec.validator)
    2. ExecutionService path/command boundary (DENY short-circuits; REQUIRE_APPROVAL
       feeds into PolicyEngine-driven approval)
    3. AutoModeService tool classification (action: approve/prompt/block)
    4. PolicyEngine approval flow (including dangerous command escalation)
    """

    def __init__(self, registry: object, permissions: object) -> None:
        self.registry = registry
        self.permissions = permissions

    def _should_run_serially(self, tool_calls: list[dict[str, Any]], mode: str) -> bool:
        if mode == "plan":
            return True
        for call in tool_calls:
            spec = self.registry.get(str(call.get("name", "")))
            if spec.capability.requires_serial_execution:
                return True
            request = self.permissions.policy_engine.build_request(
                tool_name=spec.name,
                arguments=dict(call.get("args", {}) or {}),
                capability=spec.capability,
                policy=spec.permission_policy,
                mode=mode,
            )
            if request is not None:
                return True
        return not all(
            self.registry.get(str(call.get("name", ""))).capability.concurrency_safe
            for call in tool_calls
        )

    def execute(
        self, tool_calls: list[dict[str, Any]], context: ToolContext
    ) -> ToolBatchResult:
        if not tool_calls:
            return ToolBatchResult(tool_messages=[], saw_tool_result=False, tool_error_count=0)
        serial = self._should_run_serially(tool_calls, context.mode)
        if serial:
            outputs = [self._run_one(call, context) for call in tool_calls]
        else:
            with ThreadPoolExecutor(max_workers=min(4, len(tool_calls))) as pool:
                outputs = list(
                    pool.map(lambda call: self._run_one(call, context), tool_calls)
                )
        tool_messages = []
        error_count = 0
        await_user = False
        final_text = None
        for call, result in zip(tool_calls, outputs):
            if result.await_user:
                await_user = True
                final_text = result.as_text()
            if not result.ok:
                error_count += 1
            tool_messages.append(
                make_tool_message(
                    result.as_text() or (result.error or ""),
                    tool_call_id=str(call.get("id", "")),
                    name=str(call.get("name", "")),
                    is_error=not result.ok,
                )
            )
        return ToolBatchResult(
            tool_messages=tool_messages,
            saw_tool_result=bool(tool_messages),
            tool_error_count=error_count,
            await_user=await_user,
            final_text=final_text,
        )

    def _run_one(self, tool_call: dict[str, Any], context: ToolContext) -> ToolResult:
        tool_call_id = str(tool_call.get("id", "") or "")
        name = str(tool_call.get("name", ""))
        args = dict(tool_call.get("args", {}) or {})
        spec = self.registry.get(name)
        validated = spec.validator(args)

        # Execution-layer boundary: path + command checks.
        execution = getattr(context.services, "execution", None)
        if execution is not None:
            path = _extract_path(validated)
            if path is not None:
                decision = execution.check_path_access(
                    path, write=spec.capability.writes_files
                )
                if decision.decision.value == "deny":
                    return ToolResult(
                        ok=False,
                        content=f"Path access denied: {decision.reason}",
                        error="path_denied",
                    )
            cmd = validated.get("command")
            if isinstance(cmd, str) and cmd:
                parts = cmd.split()
                command = parts[0] if parts else cmd
                cmd_args = parts[1:]
                decision = execution.check_command(command, cmd_args)
                if decision.decision.value == "deny":
                    return ToolResult(
                        ok=False,
                        content=f"Command denied: {decision.reason}",
                        error="command_denied",
                    )

        # AutoMode classifier (only in auto/plan modes — default/bypass rely on PolicyEngine).
        auto = getattr(context.services, "auto", None)
        if auto is not None and context.mode in {"auto", "plan"}:
            assessment = auto.assess(name, validated)
            auto.record(assessment.action)
            if assessment.action == "block":
                return ToolResult(
                    ok=False,
                    content=f"Blocked by auto mode: {assessment.reason}",
                    error="auto_blocked",
                )

        # PolicyEngine approval gate (patterns / risk / capability).
        decision_key = self.permissions.policy_engine.decision_key(name, validated)
        request = self.permissions.policy_engine.build_request(
            tool_name=name,
            arguments=validated,
            capability=spec.capability,
            policy=spec.permission_policy,
            mode=context.mode,
        )
        if request is not None:
            decision = ensure_tool_allowed(self.permissions, request, decision_key)
            if decision.get("decision") in {"deny", "deny_once", "deny_always"}:
                return ToolResult(
                    ok=False,
                    content="Permission denied by user.",
                    error="Permission denied",
                )

        context.emit_event(
            "tool_start",
            {
                "tool_call_id": tool_call_id,
                "tool_name": name,
                "arguments": validated,
            },
        )
        try:
            result = spec.executor(validated, context)
        except BaseException as exc:
            context.emit_event(
                "tool_result",
                {
                    "tool_call_id": tool_call_id,
                    "tool_name": name,
                    "ok": False,
                    "content": f"exception: {exc}",
                    "content_length": 0,
                    "line_count": 0,
                },
            )
            return ToolResult(ok=False, content=f"Tool raised: {exc}", error=str(exc))
        full_content = result.as_text()
        line_count = full_content.count("\n") + (
            1 if full_content and not full_content.endswith("\n") else 0
        )
        context.emit_event(
            "tool_result",
            {
                "tool_call_id": tool_call_id,
                "tool_name": name,
                "ok": result.ok,
                "content": full_content[:2000],
                "content_length": len(full_content),
                "line_count": line_count,
            },
        )
        return result
