"""Session / message / history bridging utilities.

Pure functions that translate LangChain messages into TUI entry kwargs,
plus the thread-id resolver used by ``/resume``. Kept out of ``app.py``
so the UI class doesn't own parsing.
"""

from __future__ import annotations

from typing import Any

from minicode.core.messages import extract_text, marker_kind
from minicode.features.sessions import InputHistoryRepository


_THINKING_SUFFIXES = ("\n\n/think", "\n\n/no_think")


def _strip_thinking_suffix(text: str) -> str:
    """Remove legacy qwen hybrid-thinking control tokens from persisted user text.

    Older MiniCode runners appended ``\\n\\n/think`` / ``\\n\\n/no_think`` to
    the user's prompt before saving. That suffix was dropped (Qwen thinking is
    now controlled via ``extra_body.enable_thinking``), but checkpoints written
    by old runners still carry it. On replay we strip it so the TUI Input
    widget doesn't receive a multi-line string and the slash-command parser
    doesn't misread the suffix as a fake ``/think`` command.
    """
    for suffix in _THINKING_SUFFIXES:
        if text.endswith(suffix):
            return text[: -len(suffix)]
    return text


def _make_history_repo(services: Any) -> InputHistoryRepository | None:
    paths = getattr(services, "paths", None)
    global_dir = getattr(paths, "global_dir", None)
    if global_dir is None:
        return None
    return InputHistoryRepository(global_dir / "history.json")


def _history_from_messages(messages: list[Any]) -> list[str]:
    from langchain_core.messages import HumanMessage

    history: list[str] = []
    for message in messages:
        if not isinstance(message, HumanMessage):
            continue
        text = _strip_thinking_suffix(extract_text(message)).strip()
        if not text:
            continue
        if history and history[-1] == text:
            continue
        history.append(text)
    return history


def _resolve_thread_id(services: Any, target: str) -> tuple[str | None, str | None]:
    resolver = getattr(services.sessions, "resolve_thread_id", None)
    if callable(resolver):
        return resolver(target, workspace=services.settings.workspace)

    sessions = list(services.sessions.list_sessions(workspace=services.settings.workspace))
    exact = next((s.thread_id for s in sessions if s.thread_id == target), None)
    if exact is not None:
        return exact, None

    matches = [s.thread_id for s in sessions if s.thread_id.startswith(target)]
    if not matches:
        return None, f"Unknown session: {target}"
    if len(matches) > 1:
        preview = ", ".join(matches[:5])
        return None, f"Ambiguous session prefix: {target} ({preview})"
    return matches[0], None


def _tool_kwargs_from_call(call: Any) -> dict[str, str]:
    return {
        "tool_call_id": str(call.get("id", "") or ""),
        "tool_name": str(call.get("name", "tool") or "tool"),
        "tool_args": str(call.get("args", {}) or {}),
        "tool_status": "running",
    }


def _entries_from_messages(messages: list[Any]) -> list[tuple[str, dict[str, str]]]:
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

    entries: list[tuple[str, dict[str, str]]] = []
    pending: dict[str, dict[str, str]] = {}
    for message in messages:
        if isinstance(message, HumanMessage):
            entries.append(("user", {"body": _strip_thinking_suffix(extract_text(message))}))
            continue
        if isinstance(message, ToolMessage):
            text = extract_text(message)
            status = "error" if getattr(message, "status", "success") == "error" else "success"
            tool_call_id = str(getattr(message, "tool_call_id", "") or "")
            slot = pending.pop(tool_call_id, None) if tool_call_id else None
            if slot is not None:
                slot["tool_status"] = status
                slot["tool_output"] = text
                continue
            entries.append(
                (
                    "tool",
                    {
                        "tool_call_id": tool_call_id,
                        "tool_name": message.name or "tool",
                        "tool_args": "",
                        "tool_output": text,
                        "tool_status": status,
                    },
                )
            )
            continue
        if isinstance(message, AIMessage):
            tool_calls = list(message.tool_calls or [])
            if tool_calls:
                for call in tool_calls:
                    kwargs = _tool_kwargs_from_call(call)
                    entries.append(("tool", kwargs))
                    if kwargs["tool_call_id"]:
                        pending[kwargs["tool_call_id"]] = kwargs
                continue
            text, kind = marker_kind(extract_text(message))
            entries.append(("progress" if kind == "progress" else "assistant", {"body": text}))
            continue
        entries.append(("assistant", {"body": extract_text(message)}))
    return entries
