from __future__ import annotations

import copy
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import BaseMessage

from .token_estimator import estimate_message_tokens


@dataclass
class AgentContext:
    agent_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    agent_type: str = "general"
    messages: list[BaseMessage] = field(default_factory=list)
    allowed_tools: list[str] = field(default_factory=list)
    cwd: str = "."
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    max_tokens: int = 50000

    def add_message(self, message: BaseMessage) -> None:
        self.messages.append(message)
        self.updated_at = time.time()

    def token_count(self) -> int:
        return sum(estimate_message_tokens(m) for m in self.messages)

    def summary(self) -> dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "message_count": len(self.messages),
            "token_count": self.token_count(),
            "allowed_tools": list(self.allowed_tools),
            "cwd": self.cwd,
            "created_at": self.created_at,
        }

    def clone(self) -> "AgentContext":
        return copy.deepcopy(self)


class ContextSandbox:
    """Isolated per-subagent contexts with a shared token budget."""

    def __init__(self, total_token_budget: int = 150_000) -> None:
        self._contexts: dict[str, AgentContext] = {}
        self.total_token_budget = total_token_budget
        self.used_tokens = 0

    def create_context(
        self,
        agent_type: str = "general",
        allowed_tools: list[str] | None = None,
        cwd: str = ".",
        max_tokens: int = 50_000,
    ) -> AgentContext:
        if self.used_tokens + max_tokens > self.total_token_budget:
            raise ValueError(
                f"Sandbox token budget exceeded: {self.used_tokens}/{self.total_token_budget}"
            )
        ctx = AgentContext(
            agent_type=agent_type,
            allowed_tools=list(allowed_tools or []),
            cwd=cwd,
            max_tokens=max_tokens,
        )
        self._contexts[ctx.agent_id] = ctx
        self.used_tokens += max_tokens
        return ctx

    def get(self, agent_id: str) -> AgentContext | None:
        return self._contexts.get(agent_id)

    def release(self, agent_id: str) -> None:
        ctx = self._contexts.pop(agent_id, None)
        if ctx:
            self.used_tokens = max(0, self.used_tokens - ctx.max_tokens)

    def release_all(self) -> None:
        self._contexts.clear()
        self.used_tokens = 0

    def active_count(self) -> int:
        return len(self._contexts)

    def stats(self) -> dict[str, Any]:
        pct = (self.used_tokens / self.total_token_budget * 100) if self.total_token_budget > 0 else 0
        return {
            "active_contexts": len(self._contexts),
            "used_tokens": self.used_tokens,
            "total_budget": self.total_token_budget,
            "budget_percentage": pct,
        }
