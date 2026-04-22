from __future__ import annotations

from typing import Any

from langchain_core.messages import BaseMessage

from .isolation import AgentContext, ContextSandbox
from .manager import CompactionResult, ContextManager, ContextStats
from .token_estimator import estimate_message_tokens, estimate_messages_tokens, estimate_tokens


class ContextService:
    """Facade exposing token estimation, compaction, and subagent sandboxing."""

    def __init__(
        self,
        model: str = "default",
        sandbox_budget: int = 150_000,
    ) -> None:
        self.manager = ContextManager(model=model)
        self.sandbox = ContextSandbox(total_token_budget=sandbox_budget)

    def set_model(self, model: str) -> None:
        self.manager.set_model(model)

    def estimate(self, text: str) -> int:
        return estimate_tokens(text)

    def estimate_messages(self, messages: list[BaseMessage]) -> int:
        return estimate_messages_tokens(messages)

    def stats(self, messages: list[BaseMessage]) -> ContextStats:
        return self.manager.stats(messages)

    def should_compact(self, messages: list[BaseMessage]) -> bool:
        return self.manager.should_compact(messages)

    def compact(self, messages: list[BaseMessage]) -> tuple[list[BaseMessage], CompactionResult]:
        return self.manager.compact_base_messages(messages)

    # --- sandbox ---
    def create_subagent(
        self,
        agent_type: str = "general",
        allowed_tools: list[str] | None = None,
        cwd: str = ".",
        max_tokens: int = 50_000,
    ) -> AgentContext:
        return self.sandbox.create_context(agent_type, allowed_tools, cwd, max_tokens)

    def release_subagent(self, agent_id: str) -> None:
        self.sandbox.release(agent_id)

    def sandbox_stats(self) -> dict[str, Any]:
        return self.sandbox.stats()
