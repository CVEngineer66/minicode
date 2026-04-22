from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

pytest.importorskip("langchain_core")
pytest.importorskip("textual")

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from textual import events
from textual.containers import VerticalScroll
from textual.widgets import Footer, Input, Static

from minicode.core.types import PermissionPolicy, SessionMeta, ToolCapability, ToolContext, ToolResult, ToolSpec
from minicode.features.sessions import InputHistoryRepository
from minicode.features.tools.graph_adapter import ToolGraphAdapter
from minicode.ui.tui.commands import SLASH_COMMANDS, find_matching_slash_commands
from minicode.ui.tui.app import CommandOutputScreen, CommandOutputScroll, MiniCodeApp, SlashSuggest, TranscriptScroll, _entries_from_messages, _find_running_tool_entry, _format_slash_suggest_prompt, _history_from_messages, _resolve_thread_id


def test_entries_from_messages_merges_tool_results_by_call_id() -> None:
    messages = [
        HumanMessage(content="hello"),
        AIMessage(
            content="",
            tool_calls=[
                {"id": "tool-1", "name": "read_file", "args": {"path": "a.txt"}},
                {"id": "tool-2", "name": "read_file", "args": {"path": "b.txt"}},
            ],
        ),
        ToolMessage(content="FILE A", tool_call_id="tool-1", name="read_file"),
        ToolMessage(content="FILE B", tool_call_id="tool-2", name="read_file"),
        AIMessage(content="<final> done"),
    ]

    entries = _entries_from_messages(messages)

    assert [kind for kind, _ in entries] == ["user", "tool", "tool", "assistant"]
    first_tool = entries[1][1]
    second_tool = entries[2][1]
    assert first_tool["tool_call_id"] == "tool-1"
    assert first_tool["tool_status"] == "success"
    assert first_tool["tool_output"] == "FILE A"
    assert second_tool["tool_call_id"] == "tool-2"
    assert second_tool["tool_status"] == "success"
    assert second_tool["tool_output"] == "FILE B"


def test_find_running_tool_entry_prefers_tool_call_id() -> None:
    first = SimpleNamespace(
        kind="tool",
        tool_call_id="tool-1",
        tool_name="read_file",
        tool_status="running",
    )
    second = SimpleNamespace(
        kind="tool",
        tool_call_id="tool-2",
        tool_name="read_file",
        tool_status="running",
    )

    match = _find_running_tool_entry(
        [first, second],
        tool_call_id="tool-1",
        tool_name="read_file",
    )

    assert match is first


def test_resolve_thread_id_supports_unique_prefix() -> None:
    sessions = [
        SessionMeta(
            thread_id="abc123456789",
            workspace="workspace",
            created_at=0.0,
            updated_at=2.0,
            model="model",
            title="first",
        ),
        SessionMeta(
            thread_id="def987654321",
            workspace="workspace",
            created_at=0.0,
            updated_at=1.0,
            model="model",
            title="second",
        ),
    ]
    services = SimpleNamespace(
        settings=SimpleNamespace(workspace="workspace"),
        sessions=SimpleNamespace(list_sessions=lambda workspace: sessions),
    )

    resolved, error = _resolve_thread_id(services, "abc123")

    assert resolved == "abc123456789"
    assert error is None


def test_resolve_thread_id_rejects_ambiguous_prefix() -> None:
    sessions = [
        SessionMeta(
            thread_id="abc123456789",
            workspace="workspace",
            created_at=0.0,
            updated_at=2.0,
            model="model",
            title="first",
        ),
        SessionMeta(
            thread_id="abc999999999",
            workspace="workspace",
            created_at=0.0,
            updated_at=1.0,
            model="model",
            title="second",
        ),
    ]
    services = SimpleNamespace(
        settings=SimpleNamespace(workspace="workspace"),
        sessions=SimpleNamespace(list_sessions=lambda workspace: sessions),
    )

    resolved, error = _resolve_thread_id(services, "abc")

    assert resolved is None
    assert error == "Ambiguous session prefix: abc (abc123456789, abc999999999)"


def test_history_from_messages_collects_user_inputs_in_order() -> None:
    messages = [
        HumanMessage(content="first"),
        AIMessage(content="reply"),
        HumanMessage(content="second"),
        HumanMessage(content="second"),
    ]

    history = _history_from_messages(messages)

    assert history == ["first", "second"]


def test_minicode_app_loads_persisted_history_on_init(tmp_path) -> None:
    repo = InputHistoryRepository(tmp_path / "history.json")
    repo.save(["first", "second"])
    services = SimpleNamespace(
        paths=SimpleNamespace(global_dir=tmp_path),
        settings=SimpleNamespace(auto_mode="default", model="model"),
    )

    app = MiniCodeApp(services)

    assert app._history == ["first", "second"]
    assert app._history_index == 2


def test_restore_history_from_messages_replaces_navigation_history(tmp_path) -> None:
    repo = InputHistoryRepository(tmp_path / "history.json")
    repo.save(["old"])
    services = SimpleNamespace(
        paths=SimpleNamespace(global_dir=tmp_path),
        settings=SimpleNamespace(auto_mode="default", model="model"),
    )
    app = MiniCodeApp(services)

    app._restore_history_from_messages(
        [
            HumanMessage(content="resume one"),
            AIMessage(content="reply"),
            HumanMessage(content="resume two"),
        ]
    )

    assert app._history == ["resume one", "resume two"]
    assert app._history_index == 2
    assert repo.load() == ["old", "resume one", "resume two"]


def test_format_slash_suggest_prompt_lists_usage_and_description() -> None:
    rendered = _format_slash_suggest_prompt(SLASH_COMMANDS[0])

    assert "/help" in rendered
    assert "show available commands" in rendered


def test_slash_suggest_shows_options_after_typing_slash(tmp_path) -> None:
    async def run_test() -> None:
        services = SimpleNamespace(
            paths=SimpleNamespace(global_dir=tmp_path),
            settings=SimpleNamespace(auto_mode="default", model="model"),
        )
        app = MiniCodeApp(services)
        async with app.run_test() as pilot:
            input_widget = app.query_one(Input)
            input_widget.focus()
            await pilot.press("/")
            await pilot.pause()
            suggest = app.query_one(SlashSuggest)
            assert suggest.is_visible is True
            assert suggest.option_count > 0
            assert str(suggest.get_option_at_index(0).prompt).startswith("/help")

    asyncio.run(run_test())


def test_slash_suggest_does_not_overlap_input_or_status_bar(tmp_path) -> None:
    async def run_test() -> None:
        services = SimpleNamespace(
            paths=SimpleNamespace(global_dir=tmp_path),
            settings=SimpleNamespace(auto_mode="default", model="model", workspace="default"),
            sessions=SimpleNamespace(list_sessions=lambda workspace: [], load_messages=lambda thread_id: []),
        )
        app = MiniCodeApp(services)
        async with app.run_test(size=(80, 20)) as pilot:
            input_widget = app.query_one(Input)
            status_bar = app.query_one("#status-bar", Static)
            footer = app.query_one(Footer)
            input_widget.focus()
            await pilot.press("/")
            await pilot.pause()
            suggest = app.query_one(SlashSuggest)
            assert suggest.is_visible is True
            assert suggest.region.overlaps(input_widget.region) is False
            assert suggest.region.overlaps(status_bar.region) is False
            assert status_bar.region.overlaps(footer.region) is False

    asyncio.run(run_test())


def test_slash_suggest_filters_to_matching_prefix(tmp_path) -> None:
    async def run_test() -> None:
        services = SimpleNamespace(
            paths=SimpleNamespace(global_dir=tmp_path),
            settings=SimpleNamespace(auto_mode="default", model="model"),
        )
        app = MiniCodeApp(services)
        async with app.run_test() as pilot:
            input_widget = app.query_one(Input)
            input_widget.focus()
            await pilot.press("/")
            await pilot.press("s")
            await pilot.pause()
            suggest = app.query_one(SlashSuggest)
            prompts = [str(suggest.get_option_at_index(i).prompt) for i in range(suggest.option_count)]
            assert suggest.is_visible is True
            assert any(prompt.startswith("/sessions") for prompt in prompts)
            assert all(prompt.startswith("/s") for prompt in prompts)

    asyncio.run(run_test())


def test_slash_suggest_filters_compact_for_com_prefix(tmp_path) -> None:
    async def run_test() -> None:
        services = SimpleNamespace(
            paths=SimpleNamespace(global_dir=tmp_path),
            settings=SimpleNamespace(auto_mode="default", model="model", workspace="default"),
            sessions=SimpleNamespace(list_sessions=lambda workspace: [], load_messages=lambda thread_id: []),
        )
        app = MiniCodeApp(services)
        async with app.run_test() as pilot:
            input_widget = app.query_one(Input)
            input_widget.focus()
            await pilot.press("/")
            await pilot.press("c")
            await pilot.press("o")
            await pilot.press("m")
            await pilot.pause()
            suggest = app.query_one(SlashSuggest)
            prompts = [str(suggest.get_option_at_index(i).prompt) for i in range(suggest.option_count)]
            assert suggest.is_visible is True
            assert prompts == [prompts[0]]
            assert prompts[0].startswith("/compact")

    asyncio.run(run_test())


def test_command_output_screen_consumes_scroll_keys_instead_of_history(tmp_path) -> None:
    async def run_test() -> None:
        services = SimpleNamespace(
            paths=SimpleNamespace(global_dir=tmp_path),
            settings=SimpleNamespace(auto_mode="default", model="model", workspace="default"),
            sessions=SimpleNamespace(list_sessions=lambda workspace: [], load_messages=lambda thread_id: []),
        )
        app = MiniCodeApp(services)
        app._history = ["older", "newer"]
        app._history_index = len(app._history)
        async with app.run_test(size=(80, 20)) as pilot:
            input_widget = app.query_one(Input)
            input_widget.focus()
            input_widget.value = "draft"
            input_widget.cursor_position = len(input_widget.value)
            app.push_screen(CommandOutputScreen("/help", "\n".join(f"line {i}" for i in range(80))))
            await pilot.pause()

            scroll = app.screen.query_one(CommandOutputScroll)
            await pilot.press("pagedown")
            await pilot.pause()
            assert scroll.scroll_y > 0
            assert input_widget.value == "draft"
            assert app._history_index == len(app._history)

            previous_scroll = scroll.scroll_y
            await pilot.press("up")
            await pilot.pause()
            # Modal now actually consumes up (rather than letting App's
            # priority history_prev swallow it silently), so scroll moves
            # back up. Still must not touch main input / history.
            assert scroll.scroll_y < previous_scroll
            assert input_widget.value == "draft"
            assert app._history_index == len(app._history)

    asyncio.run(run_test())


def test_command_output_screen_mouse_wheel_stays_inside_modal(tmp_path) -> None:
    async def run_test() -> None:
        services = SimpleNamespace(
            paths=SimpleNamespace(global_dir=tmp_path),
            settings=SimpleNamespace(auto_mode="default", model="model", workspace="default"),
            sessions=SimpleNamespace(list_sessions=lambda workspace: [], load_messages=lambda thread_id: []),
        )
        app = MiniCodeApp(services)
        async with app.run_test(size=(80, 20)) as pilot:
            transcript = app.query_one("#transcript-scroll", VerticalScroll)
            app.push_screen(CommandOutputScreen("/help", "\n".join(f"line {i}" for i in range(80))))
            await pilot.pause()

            scroll = app.screen.query_one(CommandOutputScroll)
            body = scroll.query(Static).last()
            body.post_message(
                events.MouseScrollDown(
                    body,
                    x=1,
                    y=1,
                    delta_x=0,
                    delta_y=1,
                    button=0,
                    shift=False,
                    meta=False,
                    ctrl=False,
                    screen_x=10,
                    screen_y=10,
                )
            )
            await pilot.pause()
            assert scroll.scroll_y > 0
            assert transcript.scroll_y == 0

    asyncio.run(run_test())


def test_slash_command_matching_supports_token_prefixes() -> None:
    compact = [command.usage for command in find_matching_slash_commands("/com")]
    resume = [command.usage for command in find_matching_slash_commands("/resume abc123")]
    memory = [command.usage for command in find_matching_slash_commands("/memory s")]

    assert compact == ["/compact"]
    assert resume == ["/resume <thread_id>"]
    assert memory == ["/memory search <query>"]


def test_history_navigation_continues_past_slash_commands(tmp_path) -> None:
    async def run_test() -> None:
        services = SimpleNamespace(
            paths=SimpleNamespace(global_dir=tmp_path),
            settings=SimpleNamespace(auto_mode="default", model="model"),
        )
        app = MiniCodeApp(services)
        app._history = ["/sessions", "/skills", "hello"]
        app._history_index = len(app._history)
        async with app.run_test() as pilot:
            input_widget = app.query_one(Input)
            input_widget.focus()

            await pilot.press("up")
            await pilot.pause()
            assert input_widget.value == "hello"

            await pilot.press("up")
            await pilot.pause()
            assert input_widget.value == "/skills"

            await pilot.press("up")
            await pilot.pause()
            assert input_widget.value == "/sessions"

    asyncio.run(run_test())


def test_history_navigation_restores_current_draft_after_browsing(tmp_path) -> None:
    async def run_test() -> None:
        services = SimpleNamespace(
            paths=SimpleNamespace(global_dir=tmp_path),
            settings=SimpleNamespace(auto_mode="default", model="model", workspace="default"),
            sessions=SimpleNamespace(list_sessions=lambda workspace: [], load_messages=lambda thread_id: []),
        )
        app = MiniCodeApp(services)
        app._history = ["/sessions", "hello"]
        app._history_index = len(app._history)
        async with app.run_test() as pilot:
            input_widget = app.query_one(Input)
            input_widget.focus()
            await pilot.press("d", "r", "a", "f", "t")
            await pilot.pause()
            await pilot.press("up")
            await pilot.pause()
            assert input_widget.value == "hello"
            await pilot.press("down")
            await pilot.pause()
            assert input_widget.value == "draft"

    asyncio.run(run_test())


def test_editing_recalled_slash_history_reopens_matching_suggestions(tmp_path) -> None:
    async def run_test() -> None:
        services = SimpleNamespace(
            paths=SimpleNamespace(global_dir=tmp_path),
            settings=SimpleNamespace(auto_mode="default", model="model", workspace="default"),
            sessions=SimpleNamespace(list_sessions=lambda workspace: [], load_messages=lambda thread_id: []),
        )
        app = MiniCodeApp(services)
        app._history = ["/co"]
        app._history_index = len(app._history)
        async with app.run_test() as pilot:
            input_widget = app.query_one(Input)
            input_widget.focus()
            await pilot.press("up")
            await pilot.pause()
            suggest = app.query_one(SlashSuggest)
            assert suggest.is_visible is False
            await pilot.press("m")
            await pilot.pause()
            prompts = [str(suggest.get_option_at_index(i).prompt) for i in range(suggest.option_count)]
            assert input_widget.value == "/com"
            assert suggest.is_visible is True
            assert prompts == [prompts[0]]
            assert prompts[0].startswith("/compact")

    asyncio.run(run_test())


def test_transcript_mouse_wheel_does_not_change_input_history(tmp_path) -> None:
    async def run_test() -> None:
        services = SimpleNamespace(
            paths=SimpleNamespace(global_dir=tmp_path),
            settings=SimpleNamespace(auto_mode="default", model="model", workspace="default"),
            sessions=SimpleNamespace(list_sessions=lambda workspace: [], load_messages=lambda thread_id: []),
        )
        app = MiniCodeApp(services)
        app._history = ["older", "newer"]
        app._history_index = len(app._history)
        async with app.run_test(size=(80, 12)) as pilot:
            input_widget = app.query_one(Input)
            input_widget.focus()
            input_widget.value = "draft"
            input_widget.cursor_position = len(input_widget.value)
            for index in range(20):
                app._add_entry("assistant", body=f"line {index}")
            await pilot.pause()

            transcript = app.query_one("#transcript-scroll", TranscriptScroll)
            entry = app.query(Static).first()
            before_history_index = app._history_index
            entry.post_message(
                events.MouseScrollDown(
                    entry,
                    x=1,
                    y=1,
                    delta_x=0,
                    delta_y=1,
                    button=0,
                    shift=False,
                    meta=False,
                    ctrl=False,
                    screen_x=10,
                    screen_y=10,
                )
            )
            await pilot.pause()
            assert transcript.scroll_y > 0
            assert input_widget.value == "draft"
            assert app._history_index == before_history_index

    asyncio.run(run_test())


class _Registry:
    def __init__(self, spec: ToolSpec) -> None:
        self.spec = spec

    def get(self, name: str) -> ToolSpec:
        assert name == self.spec.name
        return self.spec


class _PolicyEngine:
    def build_request(self, **_: object) -> None:
        return None

    def decision_key(self, name: str, arguments: dict[str, object]) -> str:
        return f"{name}:{arguments}"


class _Permissions:
    def __init__(self) -> None:
        self.policy_engine = _PolicyEngine()
