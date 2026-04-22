from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Pricing table
# ---------------------------------------------------------------------------

MODEL_PRICING: dict[str, dict[str, float]] = {
    # Anthropic
    "claude-sonnet-4-20250514": {"input": 3.0, "output": 15.0, "cache_read": 0.30, "cache_write": 3.75},
    "claude-opus-4-20250514": {"input": 15.0, "output": 75.0, "cache_read": 1.50, "cache_write": 18.75},
    "claude-haiku-3-20240307": {"input": 0.25, "output": 1.25, "cache_read": 0.03, "cache_write": 0.30},
    # OpenAI
    "gpt-4o": {"input": 2.50, "output": 10.0, "cache_read": 1.25, "cache_write": 2.50},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60, "cache_read": 0.08, "cache_write": 0.15},
    "gpt-4-turbo": {"input": 10.0, "output": 30.0, "cache_read": 5.0, "cache_write": 10.0},
    "o1": {"input": 15.0, "output": 60.0, "cache_read": 7.50, "cache_write": 15.0},
    "o1-mini": {"input": 3.0, "output": 12.0, "cache_read": 1.50, "cache_write": 3.0},
    "o3-mini": {"input": 1.10, "output": 4.40, "cache_read": 0.55, "cache_write": 1.10},
    # OpenRouter / other
    "anthropic/claude-sonnet-4": {"input": 3.0, "output": 15.0, "cache_read": 0.30, "cache_write": 3.75},
    "anthropic/claude-opus-4": {"input": 15.0, "output": 75.0, "cache_read": 1.50, "cache_write": 18.75},
    "openai/gpt-4o": {"input": 2.50, "output": 10.0, "cache_read": 1.25, "cache_write": 2.50},
    "openai/gpt-4o-mini": {"input": 0.15, "output": 0.60, "cache_read": 0.08, "cache_write": 0.15},
    "google/gemini-2.5-pro": {"input": 1.25, "output": 10.0, "cache_read": 0.63, "cache_write": 1.25},
    "google/gemini-2.5-flash": {"input": 0.15, "output": 0.60, "cache_read": 0.08, "cache_write": 0.15},
    "deepseek/deepseek-r1": {"input": 0.55, "output": 2.19, "cache_read": 0.14, "cache_write": 0.55},
    "deepseek/deepseek-chat": {"input": 0.14, "output": 0.28, "cache_read": 0.07, "cache_write": 0.14},
    "default": {"input": 3.0, "output": 15.0, "cache_read": 0.30, "cache_write": 3.75},
}


def pricing_for(model: str) -> dict[str, float]:
    return MODEL_PRICING.get(model, MODEL_PRICING["default"])


def calculate_cost(
    model: str,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cache_read_tokens: int = 0,
    cache_creation_tokens: int = 0,
) -> float:
    p = pricing_for(model)
    return (
        input_tokens / 1_000_000 * p["input"]
        + output_tokens / 1_000_000 * p["output"]
        + cache_read_tokens / 1_000_000 * p["cache_read"]
        + cache_creation_tokens / 1_000_000 * p["cache_write"]
    )


# ---------------------------------------------------------------------------
# Budget enforcement
# ---------------------------------------------------------------------------


class BudgetExceeded(RuntimeError):
    """Raised when a configured cost cap is about to be exceeded.

    Runner should catch this and surface as a LangGraph interrupt so the user
    can approve continuing or abort.
    """

    def __init__(self, scope: str, current: float, cap: float, projected: float) -> None:
        super().__init__(
            f"{scope} cost cap exceeded: "
            f"${projected:.4f} projected vs ${cap:.4f} cap "
            f"(current ${current:.4f})"
        )
        self.scope = scope
        self.current = current
        self.cap = cap
        self.projected = projected


@dataclass
class BudgetCaps:
    """Cost caps expressed in USD; 0.0 means disabled."""

    per_turn_usd: float = 0.0
    per_session_usd: float = 0.0

    def check_turn(self, current_turn: float, projected_add: float) -> None:
        if self.per_turn_usd and current_turn + projected_add > self.per_turn_usd:
            raise BudgetExceeded("turn", current_turn, self.per_turn_usd, current_turn + projected_add)

    def check_session(self, current_session: float, projected_add: float) -> None:
        if self.per_session_usd and current_session + projected_add > self.per_session_usd:
            raise BudgetExceeded(
                "session", current_session, self.per_session_usd, current_session + projected_add
            )


# ---------------------------------------------------------------------------
# Usage ledger
# ---------------------------------------------------------------------------


@dataclass
class ModelUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    cost_usd: float = 0.0
    call_count: int = 0
    error_count: int = 0
    total_duration_ms: int = 0

    @property
    def total_tokens(self) -> int:
        return (
            self.input_tokens
            + self.output_tokens
            + self.cache_read_tokens
            + self.cache_write_tokens
        )

    @property
    def avg_duration_ms(self) -> float:
        return self.total_duration_ms / self.call_count if self.call_count else 0.0


@dataclass
class CostLedger:
    total_cost_usd: float = 0.0
    total_api_duration_ms: int = 0
    total_lines_added: int = 0
    total_lines_removed: int = 0
    total_lines_modified: int = 0
    model_usage: dict[str, ModelUsage] = field(default_factory=dict)
    session_start: float = field(default_factory=time.time)
    last_updated: float = field(default_factory=time.time)

    def add_usage(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        duration_ms: int = 0,
        cache_read_tokens: int = 0,
        cache_write_tokens: int = 0,
    ) -> float:
        cost = calculate_cost(
            model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_tokens=cache_read_tokens,
            cache_creation_tokens=cache_write_tokens,
        )
        usage = self.model_usage.setdefault(model, ModelUsage())
        usage.input_tokens += input_tokens
        usage.output_tokens += output_tokens
        usage.cache_read_tokens += cache_read_tokens
        usage.cache_write_tokens += cache_write_tokens
        usage.cost_usd += cost
        usage.call_count += 1
        usage.total_duration_ms += duration_ms
        self.total_cost_usd += cost
        self.total_api_duration_ms += duration_ms
        self.last_updated = time.time()
        return cost

    def record_error(self, model: str) -> None:
        self.model_usage.setdefault(model, ModelUsage()).error_count += 1
        self.last_updated = time.time()

    def record_code_changes(self, lines_added: int = 0, lines_removed: int = 0) -> None:
        self.total_lines_added += lines_added
        self.total_lines_removed += lines_removed
        self.total_lines_modified += lines_added + lines_removed
        self.last_updated = time.time()

    def total_tokens(self) -> int:
        return sum(u.total_tokens for u in self.model_usage.values())

    def total_calls(self) -> int:
        return sum(u.call_count for u in self.model_usage.values())

    def total_errors(self) -> int:
        return sum(u.error_count for u in self.model_usage.values())

    def short_summary(self) -> str:
        if self.total_cost_usd == 0:
            return "Cost: $0.0000"
        return (
            f"Cost: ${self.total_cost_usd:.4f} | "
            f"Tokens: {self.total_tokens():,} | "
            f"Calls: {self.total_calls()}"
        )


# ---------------------------------------------------------------------------
# Service facade
# ---------------------------------------------------------------------------


class CostService:
    """Session cost accounting with per-turn and per-session budget caps.

    Boundaries:
    - `caps.per_turn_usd` and `caps.per_session_usd` enforce upper limits.
      Calling `check(...)` with a projected incremental cost raises BudgetExceeded
      so the runner can surface an interrupt to the user.
    - Errors during cost recording never propagate upward.
    """

    def __init__(
        self,
        ledger: CostLedger | None = None,
        caps: BudgetCaps | None = None,
    ) -> None:
        self.ledger = ledger or CostLedger()
        self.caps = caps or BudgetCaps()
        self._turn_started_cost: float = self.ledger.total_cost_usd

    # --- turn lifecycle ---
    def begin_turn(self) -> None:
        self._turn_started_cost = self.ledger.total_cost_usd

    def current_turn_cost(self) -> float:
        return max(0.0, self.ledger.total_cost_usd - self._turn_started_cost)

    def end_turn(self) -> float:
        cost = self.current_turn_cost()
        self._turn_started_cost = self.ledger.total_cost_usd
        return cost

    # --- check (boundary enforcement) ---
    def check(self, model: str, input_tokens: int, output_tokens: int = 0) -> None:
        projected = calculate_cost(
            model, input_tokens=input_tokens, output_tokens=output_tokens
        )
        self.caps.check_turn(self.current_turn_cost(), projected)
        self.caps.check_session(self.ledger.total_cost_usd, projected)

    # --- recording ---
    def record_api_call(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        duration_ms: int = 0,
        cache_read_tokens: int = 0,
        cache_write_tokens: int = 0,
    ) -> float:
        return self.ledger.add_usage(
            model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            duration_ms=duration_ms,
            cache_read_tokens=cache_read_tokens,
            cache_write_tokens=cache_write_tokens,
        )

    def record_error(self, model: str) -> None:
        self.ledger.record_error(model)

    def record_code_changes(self, added: int = 0, removed: int = 0) -> None:
        self.ledger.record_code_changes(added, removed)

    # --- introspection ---
    def stats(self) -> dict[str, Any]:
        return {
            "total_cost_usd": self.ledger.total_cost_usd,
            "total_tokens": self.ledger.total_tokens(),
            "total_calls": self.ledger.total_calls(),
            "total_errors": self.ledger.total_errors(),
            "session_duration_s": time.time() - self.ledger.session_start,
            "turn_cost_usd": self.current_turn_cost(),
            "caps": {
                "per_turn_usd": self.caps.per_turn_usd,
                "per_session_usd": self.caps.per_session_usd,
            },
        }

    def short_summary(self) -> str:
        return self.ledger.short_summary()
