from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("textual")

from textual.widgets import Input

from minicode.ui.tui.app import MiniCodeApp
from minicode.ui.tui.approval_entry import (
    ApprovalEntry,
    _safe_literal,
    _split_args,
)


_CHOICES = [
    {"key": "y", "label": "Allow once", "decision": "allow_once"},
    {"key": "t", "label": "Allow turn", "decision": "allow_turn"},
    {"key": "a", "label": "Always",     "decision": "allow_always"},
    {"key": "n", "label": "Deny",       "decision": "deny_once"},
]

_LONG_CONTENT = "\n".join(f"line{i}" for i in range(7))


def _payload(details: list[str] | None = None, choices: list[dict] | None = None) -> dict:
    return {
        "summary": "Approve write_file",
        "details": details
        if details is not None
        else [f"Arguments: {{'path': 'a.txt', 'content': {_LONG_CONTENT!r}}}"],
        "choices": choices if choices is not None else _CHOICES,
    }


def _choice_payload() -> dict:
    return {
        "prompt_kind": "choice",
        "summary": "Pick an implementation strategy",
        "details": [
            "1. Minimal patch - smaller diff, keeps current structure",
            "2. Refactor - cleaner model, larger surface area",
        ],
        "choices": [
            {
                "key": "1",
                "label": "Minimal patch",
                "payload": {"choice_id": "minimal", "choice_label": "Minimal patch"},
            },
            {
                "key": "2",
                "label": "Refactor",
                "payload": {"choice_id": "refactor", "choice_label": "Refactor"},
            },
        ],
        "cancel_payload": {"choice_cancelled": True},
    }


# ---- pure helpers ---------------------------------------------------


def test_split_args_separates_body_from_short_fields() -> None:
    short, body = _split_args({"path": "a.txt", "content": "x\ny", "append": False})
    assert short == {"path": "a.txt", "append": False}
    assert body == {"content": "x\ny"}


def test_split_args_treats_long_scalar_as_body() -> None:
    long_value = "x" * 100
    short, body = _split_args({"note": long_value, "count": 3})
    assert short == {"count": 3}
    assert body == {"note": long_value}


def test_safe_literal_handles_malformed() -> None:
    assert _safe_literal("{not valid") is None
    assert _safe_literal("[1, 2, 3]") is None  # list, not dict
    assert _safe_literal("{'x': 1}") == {"x": 1}


# ---- widget state (no pilot needed) --------------------------------


def test_approval_entry_initial_state_highlights_first_choice() -> None:
    entry = ApprovalEntry(_payload())
    assert entry._highlight == 0
    assert entry._expanded is False
    assert len(entry._choices) == 4


def test_approval_entry_move_up_down_cycles_highlight() -> None:
    entry = ApprovalEntry(_payload())
    entry.action_move_down()
    assert entry._highlight == 1
    entry.action_move_down()
    entry.action_move_down()
    entry.action_move_down()
    assert entry._highlight == 0  # wrapped
    entry.action_move_up()
    assert entry._highlight == 3  # wrapped backwards


def test_approval_entry_confirm_posts_highlighted_decision() -> None:
    entry = ApprovalEntry(_payload())
    entry.action_move_down()  # now at allow_turn
    with patch.object(entry, "post_message") as mock_post:
        entry.action_confirm()
    assert mock_post.called
    msg = mock_post.call_args[0][0]
    assert isinstance(msg, ApprovalEntry.ApprovalDecided)
    assert msg.decision == {"decision": "allow_turn"}


def test_approval_entry_deny_posts_deny_regardless_of_highlight() -> None:
    entry = ApprovalEntry(_payload())
    entry.action_move_down()
    entry.action_move_down()
    with patch.object(entry, "post_message") as mock_post:
        entry.action_deny()
    msg = mock_post.call_args[0][0]
    assert msg.decision == {"decision": "deny"}


def test_choice_entry_confirm_posts_selected_payload() -> None:
    entry = ApprovalEntry(_choice_payload())
    entry.action_move_down()
    with patch.object(entry, "post_message") as mock_post:
        entry.action_confirm()
    msg = mock_post.call_args[0][0]
    assert msg.decision == {"choice_id": "refactor", "choice_label": "Refactor"}


def test_choice_entry_deny_posts_cancel_payload() -> None:
    entry = ApprovalEntry(_choice_payload())
    with patch.object(entry, "post_message") as mock_post:
        entry.action_deny()
    msg = mock_post.call_args[0][0]
    assert msg.decision == {"choice_cancelled": True}


def test_approval_entry_toggle_expand_flips_state() -> None:
    entry = ApprovalEntry(_payload())
    assert entry._expanded is False
    entry.action_toggle_expand()
    assert entry._expanded is True
    entry.action_toggle_expand()
    assert entry._expanded is False


def test_approval_entry_single_choice_hard_deny_still_works() -> None:
    """Hard-deny path only has one choice ('n=Acknowledged')."""
    payload = _payload(choices=[{"key": "n", "label": "Ack", "decision": "deny_once"}])
    entry = ApprovalEntry(payload)
    entry.action_move_down()  # no-op cycle on single choice
    assert entry._highlight == 0
    with patch.object(entry, "post_message") as mock_post:
        entry.action_confirm()
    assert mock_post.call_args[0][0].decision == {"decision": "deny_once"}


def test_approval_entry_empty_choices_confirm_noop() -> None:
    payload = _payload(choices=[])
    entry = ApprovalEntry(payload)
    with patch.object(entry, "post_message") as mock_post:
        entry.action_confirm()
        entry.action_move_up()
        entry.action_move_down()
    assert mock_post.called is False


# ---- rendering (body folding) --------------------------------------


def test_approval_entry_folds_long_body_to_preview() -> None:
    entry = ApprovalEntry(_payload())
    rendered = _render_str(entry)
    assert "more lines" in rendered
    assert "ctrl+o to expand" in rendered
    # Non-body scalar still shown
    assert "path:" in rendered and "a.txt" in rendered


def test_approval_entry_short_body_not_folded() -> None:
    short = "only\ntwo"
    payload = _payload(details=[f"Arguments: {{'path': 'a.txt', 'content': {short!r}}}"])
    entry = ApprovalEntry(payload)
    rendered = _render_str(entry)
    assert "more lines" not in rendered
    assert "only" in rendered and "two" in rendered


def test_approval_entry_expanded_shows_all_body_lines() -> None:
    entry = ApprovalEntry(_payload())
    entry.action_toggle_expand()
    rendered = _render_str(entry)
    assert "more lines" not in rendered
    # All 7 lines present
    for i in range(7):
        assert f"line{i}" in rendered


def test_approval_entry_dangerous_reason_rendered() -> None:
    payload = _payload(
        details=[
            "Potentially destructive: rm -rf",
            "Arguments: {'command': 'rm', 'args': ['-rf', '/tmp/x']}",
        ]
    )
    entry = ApprovalEntry(payload)
    rendered = _render_str(entry)
    assert "rm -rf" in rendered


def test_choice_entry_renders_choice_header_and_cancel_hint() -> None:
    entry = ApprovalEntry(_choice_payload())
    rendered = _render_str(entry)
    assert "User choice required" in rendered
    assert "Esc cancel" in rendered


# ---- app-level wiring ----------------------------------------------


def test_show_approval_inline_mounts_entry_and_focuses_it(tmp_path) -> None:
    async def run_test() -> None:
        services = SimpleNamespace(
            paths=SimpleNamespace(global_dir=tmp_path),
            settings=SimpleNamespace(auto_mode="default", model="m"),
        )
        app = MiniCodeApp(services)
        async with app.run_test() as pilot:
            app._show_approval_inline(_payload())
            await pilot.pause()
            entries = list(app.query(ApprovalEntry))
            assert len(entries) == 1
            assert app._approval_entry is entries[0]
            assert app.focused is entries[0]

    asyncio.run(run_test())


def test_approval_decided_removes_entry_restores_input_resumes_turn(tmp_path) -> None:
    async def run_test() -> None:
        services = SimpleNamespace(
            paths=SimpleNamespace(global_dir=tmp_path),
            settings=SimpleNamespace(auto_mode="default", model="m"),
        )
        app = MiniCodeApp(services)
        launch_calls: list[dict] = []

        def fake_launch(**kwargs):
            launch_calls.append(kwargs)

        async with app.run_test() as pilot:
            app._launch_turn = fake_launch  # type: ignore[assignment]
            app._show_approval_inline(_payload())
            await pilot.pause()

            entry = app._approval_entry
            assert entry is not None
            entry.action_move_down()  # highlight allow_turn
            entry.action_confirm()
            await pilot.pause()

            assert app._approval_entry is None
            assert list(app.query(ApprovalEntry)) == []
            assert app.focused is app.query_one(Input)
            assert launch_calls == [{"resume": {"decision": "allow_turn"}}]

    asyncio.run(run_test())


def test_choice_decided_removes_entry_restores_input_resumes_turn(tmp_path) -> None:
    async def run_test() -> None:
        services = SimpleNamespace(
            paths=SimpleNamespace(global_dir=tmp_path),
            settings=SimpleNamespace(auto_mode="default", model="m"),
        )
        app = MiniCodeApp(services)
        launch_calls: list[dict] = []

        def fake_launch(**kwargs):
            launch_calls.append(kwargs)

        async with app.run_test() as pilot:
            app._launch_turn = fake_launch  # type: ignore[assignment]
            app._show_approval_inline(_choice_payload())
            await pilot.pause()

            entry = app._approval_entry
            assert entry is not None
            entry.action_move_down()
            entry.action_confirm()
            await pilot.pause()

            assert app._approval_entry is None
            assert list(app.query(ApprovalEntry)) == []
            assert app.focused is app.query_one(Input)
            assert launch_calls == [{"resume": {"choice_id": "refactor", "choice_label": "Refactor"}}]

    asyncio.run(run_test())


def test_approval_entry_up_down_priority_via_pilot(tmp_path) -> None:
    """Integration: with entry focused, arrow keys do NOT leak to App history."""
    async def run_test() -> None:
        services = SimpleNamespace(
            paths=SimpleNamespace(global_dir=tmp_path),
            settings=SimpleNamespace(auto_mode="default", model="m"),
        )
        app = MiniCodeApp(services)
        app._history = ["hi"]
        app._history_index = len(app._history)

        async with app.run_test() as pilot:
            app._show_approval_inline(_payload())
            await pilot.pause()

            entry = app._approval_entry
            assert entry is not None
            await pilot.press("down")
            await pilot.pause()
            assert entry._highlight == 1
            # History index untouched (arrow was consumed by entry)
            assert app._history_index == len(app._history)

    asyncio.run(run_test())


# ---- helpers --------------------------------------------------------


def _render_str(entry: ApprovalEntry) -> str:
    """Stringify the Group renderable for substring checks."""
    group = entry._build_renderable()
    parts: list[str] = []
    for piece in group.renderables:
        parts.append(str(piece))
    return "\n".join(parts)
