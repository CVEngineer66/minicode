from __future__ import annotations

import time

import pytest

from minicode.core.events import EventBus
from minicode.features.hooks import (
    HookEvent,
    HookService,
)


def test_register_and_fire_basic():
    svc = HookService(EventBus())
    calls: list[str] = []

    svc.register(HookEvent.PRE_TURN, lambda ctx: calls.append(ctx.data.get("user_input", "")))
    svc.fire(HookEvent.PRE_TURN, user_input="hi")
    assert calls == ["hi"]


def test_handler_error_does_not_break_flow():
    svc = HookService(EventBus())
    err_calls: list[dict] = []

    def bad(ctx):
        raise ValueError("boom")

    svc.register(HookEvent.PRE_TOOL, bad, description="bad")
    svc.register(HookEvent.ON_ERROR, lambda ctx: err_calls.append(ctx.data))
    results = svc.fire(HookEvent.PRE_TOOL, tool_name="t")
    assert any("error" in r for r in results if isinstance(r, dict))
    # ON_ERROR dispatched with a failures payload
    assert err_calls and "failures" in err_calls[0]


def test_timeout_bounded():
    svc = HookService(EventBus(), timeout_s=0.05)
    svc.register(HookEvent.POST_TOOL, lambda ctx: time.sleep(0.2))
    results = svc.fire(HookEvent.POST_TOOL)
    assert any(isinstance(r, dict) and r.get("error") == "timeout" for r in results)
    assert svc.stats()["timeout_count"] == 1


def test_string_event_accepted():
    svc = HookService(EventBus())
    hits = []
    svc.register("pre_turn", lambda ctx: hits.append(1))
    svc.fire("pre_turn")
    assert hits == [1]


def test_disable_blocks_fire():
    svc = HookService(EventBus())
    calls = []
    svc.register(HookEvent.PRE_TURN, lambda ctx: calls.append(1))
    svc.disable()
    svc.fire(HookEvent.PRE_TURN)
    assert calls == []
    svc.enable()
    svc.fire(HookEvent.PRE_TURN)
    assert calls == [1]


def test_bus_emission_side_channel():
    bus = EventBus()
    svc = HookService(bus)
    seen: list[str] = []
    bus.subscribe("hook.pre_turn", lambda evt: seen.append(evt.kind))
    svc.fire(HookEvent.PRE_TURN)
    assert seen == ["hook.pre_turn"]


def test_unregister_returns_fn():
    svc = HookService(EventBus())
    calls = []
    unreg = svc.register(HookEvent.PRE_TURN, lambda ctx: calls.append(1))
    svc.fire(HookEvent.PRE_TURN)
    unreg()
    svc.fire(HookEvent.PRE_TURN)
    assert calls == [1]
