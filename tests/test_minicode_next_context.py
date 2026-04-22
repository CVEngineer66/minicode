from __future__ import annotations

import pytest

pytest.importorskip("langchain_core")

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from minicode.features.context import (
    ContextManager,
    ContextSandbox,
    ContextService,
    estimate_message_tokens,
    estimate_tokens,
    message_to_dict,
)


def test_estimate_tokens_cjk_and_ascii():
    assert estimate_tokens("hello world") > 0
    assert estimate_tokens("你好世界") > 0
    # Cache hit path
    assert estimate_tokens("hello world") == estimate_tokens("hello world")


def test_message_to_dict_variants():
    assert message_to_dict(SystemMessage(content="s"))["role"] == "system"
    assert message_to_dict(HumanMessage(content="u"))["role"] == "user"
    ai = AIMessage(content="", tool_calls=[{"name": "t", "args": {"a": 1}, "id": "x"}])
    d = message_to_dict(ai)
    assert d["role"] == "assistant_tool_call"
    assert d["toolName"] == "t"
    tm = ToolMessage(content="out", tool_call_id="x", name="t")
    assert message_to_dict(tm)["role"] == "tool_result"


def test_manager_stats_empty():
    mgr = ContextManager(model="default")
    s = mgr.stats([])
    assert s.total_tokens == 0
    assert s.context_window > 0


def test_manager_compact_triggers():
    mgr = ContextManager(model="default", context_window=1000)
    # fill with large-ish messages to exceed 95%
    msgs = [HumanMessage(content="x" * 6000) for _ in range(3)]
    assert mgr.should_compact(msgs) is True
    compacted, _ = mgr.compact_base_messages(msgs)
    # system marker always present after compaction
    assert any(isinstance(m, SystemMessage) for m in compacted)


def test_service_facade():
    svc = ContextService(model="default")
    assert svc.estimate("abc") > 0
    assert svc.stats([]).context_window > 0
    ctx = svc.create_subagent(agent_type="explore", max_tokens=100)
    assert ctx.agent_id
    svc.release_subagent(ctx.agent_id)
    assert svc.sandbox_stats()["active_contexts"] == 0


def test_sandbox_budget_exceeded():
    box = ContextSandbox(total_token_budget=50)
    with pytest.raises(ValueError):
        box.create_context(max_tokens=100)


def test_estimate_message_tokens_dict_input():
    assert (
        estimate_message_tokens({"role": "user", "content": "hello"}) > 0
    )
