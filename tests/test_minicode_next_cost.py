from __future__ import annotations

import pytest

from minicode.features.cost import (
    BudgetCaps,
    BudgetExceeded,
    CostService,
    calculate_cost,
)


def test_calculate_cost_default_model():
    cost = calculate_cost("unknown-model", input_tokens=1_000_000, output_tokens=0)
    assert cost == pytest.approx(3.0, rel=1e-6)


def test_calculate_cost_known_model():
    cost = calculate_cost("gpt-4o-mini", input_tokens=1_000_000, output_tokens=1_000_000)
    assert cost == pytest.approx(0.15 + 0.60, rel=1e-6)


def test_service_record_and_stats():
    svc = CostService()
    svc.begin_turn()
    svc.record_api_call("gpt-4o-mini", input_tokens=500_000, output_tokens=200_000)
    stats = svc.stats()
    assert stats["total_calls"] == 1
    assert stats["total_cost_usd"] > 0
    assert stats["turn_cost_usd"] > 0


def test_budget_turn_cap_triggers():
    svc = CostService(caps=BudgetCaps(per_turn_usd=0.05))
    svc.begin_turn()
    with pytest.raises(BudgetExceeded) as exc:
        # 1M input tokens at $3/1M = $3 projected, far above $0.05 cap
        svc.check("claude-sonnet-4-20250514", input_tokens=1_000_000)
    assert exc.value.scope == "turn"


def test_budget_session_cap_triggers():
    svc = CostService(caps=BudgetCaps(per_session_usd=0.01))
    svc.record_api_call("gpt-4o-mini", input_tokens=100_000, output_tokens=0)
    with pytest.raises(BudgetExceeded) as exc:
        svc.check("gpt-4o-mini", input_tokens=10_000_000)
    assert exc.value.scope in {"turn", "session"}  # turn cap defaults disabled → session


def test_end_turn_resets_tracking():
    svc = CostService()
    svc.begin_turn()
    svc.record_api_call("gpt-4o-mini", input_tokens=100_000, output_tokens=0)
    assert svc.current_turn_cost() > 0
    svc.end_turn()
    assert svc.current_turn_cost() == 0


def test_error_recording():
    svc = CostService()
    svc.record_error("gpt-4o-mini")
    assert svc.stats()["total_errors"] == 1


def test_code_change_recording():
    svc = CostService()
    svc.record_code_changes(added=10, removed=3)
    assert svc.ledger.total_lines_added == 10
    assert svc.ledger.total_lines_removed == 3
    assert svc.ledger.total_lines_modified == 13
