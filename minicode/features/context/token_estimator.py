from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage

_CJK_PATTERN = re.compile(r"[\u4E00-\u9FFF\u3040-\u309F\u30A0-\u30FF\uAC00-\uD7AF]")
_TOKEN_CACHE: dict[Any, int] = {}
_TOKEN_CACHE_MAX = 1024


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    key = text if len(text) < 256 else hash(text)
    cached = _TOKEN_CACHE.get(key)
    if cached is not None:
        return cached
    cjk = len(_CJK_PATTERN.findall(text))
    ascii_chars = len(text) - cjk
    result = max(1, int(cjk / 1.5 + ascii_chars / 4.0))
    if len(_TOKEN_CACHE) < _TOKEN_CACHE_MAX:
        _TOKEN_CACHE[key] = result
    return result


def message_to_dict(msg: BaseMessage) -> dict[str, Any]:
    """Normalize a LangChain message into the dict shape used internally."""
    content = msg.content if isinstance(msg.content, str) else str(msg.content)
    if isinstance(msg, SystemMessage):
        return {"role": "system", "content": content}
    if isinstance(msg, HumanMessage):
        return {"role": "user", "content": content}
    if isinstance(msg, ToolMessage):
        return {
            "role": "tool_result",
            "content": content,
            "toolName": getattr(msg, "name", "") or "",
            "isError": bool(getattr(msg, "status", "") == "error"),
        }
    if isinstance(msg, AIMessage):
        tool_calls = getattr(msg, "tool_calls", None) or []
        if tool_calls:
            call = tool_calls[0]
            return {
                "role": "assistant_tool_call",
                "content": content,
                "toolName": call.get("name", "") if isinstance(call, dict) else getattr(call, "name", ""),
                "input": call.get("args", {}) if isinstance(call, dict) else getattr(call, "args", {}),
            }
        return {"role": "assistant", "content": content}
    return {"role": getattr(msg, "type", "unknown"), "content": content}


def estimate_message_tokens(msg: BaseMessage | dict[str, Any]) -> int:
    data = msg if isinstance(msg, dict) else message_to_dict(msg)
    overhead = {
        "system": 3,
        "user": 4,
        "assistant": 3,
        "assistant_tool_call": 7,
        "tool_result": 6,
        "assistant_progress": 3,
    }
    tokens = overhead.get(data.get("role", ""), 3)
    content = data.get("content", "")
    if isinstance(content, str):
        tokens += estimate_tokens(content)
    if "input" in data:
        inp = data["input"]
        text = json.dumps(inp, ensure_ascii=False) if isinstance(inp, dict) else str(inp)
        tokens += estimate_tokens(text)
    return tokens


def estimate_messages_tokens(messages: list[BaseMessage] | list[dict[str, Any]]) -> int:
    return sum(estimate_message_tokens(m) for m in messages)
