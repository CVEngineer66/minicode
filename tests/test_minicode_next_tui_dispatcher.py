from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("langchain_core")

from minicode.ui.tui.commands import SLASH_COMMANDS, find_matching_slash_commands
from minicode.ui.tui.dispatcher import SlashDispatcher


def _fake_services():
    class _Sessions:
        def list_sessions(self):
            return []

        def get_latest_session(self):
            return None

        def get_session(self, tid):
            return None

        def branch(self, source):
            return "new123"

        def compact(self, tid, ctx):
            return 0

        def load_messages(self, tid):
            return []

    class _Mem:
        def stats(self):
            return {"total": 0}

        def search(self, q):
            return []

    class _Cost:
        def short_summary(self):
            return "Cost: $0.0000"

        def stats(self):
            return {"total_cost_usd": 0}

    class _Ctx:
        def stats(self, messages):
            return SimpleNamespace(
                context_window=128000,
                messages_count=0,
                total_tokens=0,
                usage_percentage=0.0,
                is_near_limit=False,
                should_compact=False,
            )

        class manager:  # noqa: N801
            model = "default"

    class _Profile:
        def load_merged(self):
            return SimpleNamespace()

        def to_prompt_section(self, _p):
            return ""

    class _Auto:
        def get_mode(self):
            return SimpleNamespace(value="default")

        def set_mode(self, target):
            return f"Mode set to {target}"

    class _Tracker:
        def list_tasks(self, ws):
            return []

    class _Skills:
        def list_skills(self):
            return []

    class _Mcp:
        def list_servers(self):
            return []

    return SimpleNamespace(
        sessions=_Sessions(),
        memory=_Mem(),
        cost=_Cost(),
        context=_Ctx(),
        profile=_Profile(),
        auto=_Auto(),
        task_tracker=_Tracker(),
        skills=_Skills(),
        mcp=_Mcp(),
        hooks=SimpleNamespace(stats=lambda: {"total_hooks": 0}),
        permissions=SimpleNamespace(summary=lambda: {"turn_allow_count": 0, "patterns": {}}),
        collaboration=SimpleNamespace(list_agents=lambda: []),
        settings=SimpleNamespace(workspace="/tmp", model="default"),
    )


def test_find_matching_slash_prefix():
    matches = find_matching_slash_commands("/mo")
    assert any(m.usage == "/mode default" for m in matches)
    assert any(m.usage == "/model" for m in matches)


def test_find_matching_slash_empty_returns_all():
    assert len(find_matching_slash_commands("/")) == len(SLASH_COMMANDS)


def test_find_matching_slash_non_slash_empty():
    assert find_matching_slash_commands("hello") == []


def test_dispatch_non_slash_not_handled():
    disp = SlashDispatcher(_fake_services())
    r = disp.dispatch("hello")
    assert r.handled is False


def test_dispatch_help():
    disp = SlashDispatcher(_fake_services())
    r = disp.dispatch("/help")
    assert r.handled and "Available commands" in r.output


def test_dispatch_unknown_command():
    disp = SlashDispatcher(_fake_services())
    r = disp.dispatch("/not_a_command")
    assert r.handled
    assert "Unknown" in r.output


def test_dispatch_quit_sets_quit_flag():
    disp = SlashDispatcher(_fake_services())
    r = disp.dispatch("/quit")
    assert r.handled and r.quit


def test_dispatch_mode_shows_current():
    disp = SlashDispatcher(_fake_services())
    r = disp.dispatch("/mode")
    assert "Current mode" in r.output


def test_dispatch_mode_sets_target():
    disp = SlashDispatcher(_fake_services())
    r = disp.dispatch("/mode auto")
    assert "auto" in r.output.lower()


def test_dispatch_cost_includes_summary():
    disp = SlashDispatcher(_fake_services())
    r = disp.dispatch("/cost")
    assert "Cost" in r.output


def test_dispatch_memory_stats():
    disp = SlashDispatcher(_fake_services())
    r = disp.dispatch("/memory")
    assert "Memory stats" in r.output


def test_dispatch_context_no_session():
    disp = SlashDispatcher(_fake_services())
    r = disp.dispatch("/context")
    assert "No active session" in r.output


def test_dispatch_tasks_empty():
    disp = SlashDispatcher(_fake_services())
    r = disp.dispatch("/tasks")
    assert "No tracked tasks" in r.output
