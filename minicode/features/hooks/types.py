from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable


class HookEvent(str, Enum):
    PRE_TURN = "pre_turn"
    POST_TURN = "post_turn"
    PRE_TOOL = "pre_tool"
    POST_TOOL = "post_tool"
    ON_ERROR = "on_error"
    AGENT_START = "agent_start"
    AGENT_STOP = "agent_stop"
    SUBAGENT_START = "subagent_start"
    SUBAGENT_STOP = "subagent_stop"
    SESSION_SAVE = "session_save"
    SESSION_RESUME = "session_resume"
    USER_INPUT = "user_input"
    ASSISTANT_OUTPUT = "assistant_output"
    STARTUP = "startup"
    SHUTDOWN = "shutdown"


@dataclass
class HookContext:
    event: HookEvent
    timestamp: float = field(default_factory=time.time)
    data: dict[str, Any] = field(default_factory=dict)

    @property
    def tool_name(self) -> str | None:
        return self.data.get("tool_name")

    @property
    def tool_input(self) -> Any:
        return self.data.get("tool_input")

    @property
    def tool_output(self) -> Any:
        return self.data.get("tool_output")

    @property
    def is_error(self) -> bool:
        return bool(self.data.get("is_error"))

    @property
    def session_id(self) -> str | None:
        return self.data.get("session_id")


HookHandler = Callable[[HookContext], Any]


@dataclass
class HookRegistration:
    event: HookEvent
    handler: HookHandler
    description: str = ""
    enabled: bool = True
    created_at: float = field(default_factory=time.time)
    call_count: int = 0
    last_called: float | None = None
    total_duration_ms: int = 0
    error_count: int = 0
    timeout_count: int = 0
