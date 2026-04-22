from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage


def extract_text(message: BaseMessage | None) -> str:
    if message is None:
        return ""
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
        return "".join(parts)
    return str(content)


def extract_thinking_delta(message: BaseMessage | None) -> str:
    """Pull thinking/reasoning delta from a streaming chunk or message.

    Supports these streaming conventions:
    - Claude extended thinking: content is a list; items with
      ``type == "thinking"`` carry the delta in ``thinking``.
    - OpenAI Responses API reasoning summaries: content contains
      ``{"type": "reasoning", "summary": [{"text": ...}]}`` blocks.
    - OpenAI o-series / DeepSeek-R1 / Qwen reasoning: ``additional_kwargs``
      may contain ``reasoning_content`` or ``reasoning`` metadata.
    """
    if message is None:
        return ""
    parts: list[str] = []
    content = getattr(message, "content", None)
    if isinstance(content, list):
        for item in content:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "thinking":
                thinking = item.get("thinking", "")
                if isinstance(thinking, str) and thinking:
                    parts.append(thinking)
                text = item.get("text", "")
                if isinstance(text, str) and text:
                    parts.append(text)
                continue
            if item.get("type") in {"reasoning", "reasoning_content"}:
                parts.extend(_extract_reasoning_parts(item))
    extra = getattr(message, "additional_kwargs", None) or {}
    if isinstance(extra, dict):
        reasoning = extra.get("reasoning_content")
        if isinstance(reasoning, str) and reasoning:
            parts.append(reasoning)
        parts.extend(_extract_reasoning_parts(extra.get("reasoning")))
        parts.extend(_extract_reasoning_parts(extra.get("reasoning_details")))
    return "".join(parts)


def _extract_reasoning_parts(value: Any) -> list[str]:
    parts: list[str] = []
    if value is None:
        return parts
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, list):
        for item in value:
            parts.extend(_extract_reasoning_parts(item))
        return parts
    if not isinstance(value, dict):
        return parts

    for key in ("thinking", "text", "reasoning_content", "content"):
        text = value.get(key)
        if isinstance(text, str) and text:
            parts.append(text)

    summary = value.get("summary")
    if isinstance(summary, list):
        for item in summary:
            parts.extend(_extract_reasoning_parts(item))
    elif isinstance(summary, str) and summary:
        parts.append(summary)

    details = value.get("details")
    if isinstance(details, list):
        for item in details:
            parts.extend(_extract_reasoning_parts(item))

    return parts


def make_system_message(text: str) -> SystemMessage:
    return SystemMessage(content=text)


def make_human_message(text: str) -> HumanMessage:
    return HumanMessage(content=text)


def make_ai_message(text: str, **kwargs: Any) -> AIMessage:
    return AIMessage(content=text, **kwargs)


def make_tool_message(
    text: str,
    *,
    tool_call_id: str,
    name: str | None = None,
    is_error: bool = False,
) -> ToolMessage:
    return ToolMessage(
        content=text,
        tool_call_id=tool_call_id,
        name=name,
        status="error" if is_error else "success",
    )


def marker_kind(text: str) -> tuple[str, str]:
    stripped = text.strip()
    lowered = stripped.lower()
    for prefix, kind in (
        ("<progress>", "progress"),
        ("[progress]", "progress"),
        ("<final>", "final"),
        ("[final]", "final"),
    ):
        if lowered.startswith(prefix):
            return stripped[len(prefix) :].strip(), kind
    return stripped, "plain"
