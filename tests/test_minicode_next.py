from __future__ import annotations

import json
import os
from pathlib import Path

from langchain_core.messages import AIMessage

from minicode.app.bootstrap import bootstrap_services
from minicode.runtime.runner import run_turn


class FakeChatModel:
    def __init__(self, responses):
        self.responses = list(responses)
        self.bound_tools = None

    def bind_tools(self, tools):
        self.bound_tools = tools
        return self

    def invoke(self, messages):
        if not self.responses:
            raise AssertionError("No scripted response left")
        return self.responses.pop(0)

    def stream(self, messages):
        # Emit the next scripted response as a single chunk — sufficient
        # for the graph's accumulation logic in tests.
        yield self.invoke(messages)


def _bootstrap(tmp_path: Path, monkeypatch) -> object:
    monkeypatch.setenv("MINICODE_NEXT_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("MINICODE_NEXT_PROVIDER", "openai")
    monkeypatch.setenv("MINICODE_NEXT_MODEL", "gpt-4o-mini")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    return bootstrap_services(str(workspace))


def test_run_turn_returns_final_text(tmp_path: Path, monkeypatch) -> None:
    services = _bootstrap(tmp_path, monkeypatch)
    model = FakeChatModel([AIMessage(content="<final> done")])
    result = run_turn(services=services, prompt="hello", chat_model=model)
    assert result.final_text == "done"
    assert result.error is None
    assert result.interrupt is None


def test_run_turn_write_file_interrupt_and_resume(tmp_path: Path, monkeypatch) -> None:
    services = _bootstrap(tmp_path, monkeypatch)
    model = FakeChatModel(
        [
            AIMessage(
                content="",
                tool_calls=[{"name": "write_file", "args": {"path": "note.txt", "content": "hello"}, "id": "tool-1", "type": "tool_call"}],
            ),
            AIMessage(content="<final> finished"),
        ]
    )
    first = run_turn(services=services, prompt="write a file", chat_model=model)
    assert first.interrupt is not None
    resumed = run_turn(
        services=services,
        thread_id=first.thread_id,
        resume={"decision": "allow_once"},
        chat_model=model,
    )
    assert resumed.final_text == "finished"
    output_file = Path(services.settings.workspace) / "note.txt"
    assert output_file.read_text(encoding="utf-8") == "hello"


def test_ask_user_tool_ends_turn(tmp_path: Path, monkeypatch) -> None:
    services = _bootstrap(tmp_path, monkeypatch)
    model = FakeChatModel(
        [
            AIMessage(
                content="",
                tool_calls=[{"name": "ask_user", "args": {"question": "Which file should I edit?"}, "id": "tool-1", "type": "tool_call"}],
            )
        ]
    )
    result = run_turn(services=services, prompt="need clarification", chat_model=model)
    assert result.await_user is True
    assert result.final_text == "Which file should I edit?"


def test_migration_imports_legacy_session(tmp_path: Path, monkeypatch) -> None:
    legacy_home = tmp_path / "legacy"
    sessions_dir = legacy_home / "sessions"
    sessions_dir.mkdir(parents=True)
    (sessions_dir / "demo.json").write_text(
        json.dumps(
            {
                "session_id": "legacy-demo",
                "workspace": str(tmp_path / "workspace"),
                "messages": [
                    {"role": "user", "content": "hello"},
                    {"role": "assistant", "content": "world"},
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("MINICODE_LEGACY_HOME", str(legacy_home))
    services = _bootstrap(tmp_path, monkeypatch)
    sessions = services.sessions.list_sessions()
    assert any(session.thread_id == "legacy-demo" for session in sessions)


def test_run_turn_emits_model_call_start(tmp_path: Path, monkeypatch) -> None:
    services = _bootstrap(tmp_path, monkeypatch)
    model = FakeChatModel([AIMessage(content="<final> ok")])
    result = run_turn(services=services, prompt="hi", chat_model=model)
    kinds = [event.kind for event in result.events]
    assert "model_call_start" in kinds


def test_run_turn_emits_thinking_from_claude_blocks(tmp_path: Path, monkeypatch) -> None:
    services = _bootstrap(tmp_path, monkeypatch)
    # Claude extended-thinking chunk: content is a list with a "thinking" block
    # alongside a "text" block.
    chunk = AIMessage(
        content=[
            {"type": "thinking", "thinking": "step 1 reasoning"},
            {"type": "text", "text": "<final> answer"},
        ]
    )
    result = run_turn(services=services, prompt="hi", chat_model=FakeChatModel([chunk]))
    thinking_events = [e for e in result.events if e.kind == "assistant_thinking"]
    assert thinking_events
    assert thinking_events[0].payload.get("text") == "step 1 reasoning"


def test_run_turn_emits_thinking_from_reasoning_content(tmp_path: Path, monkeypatch) -> None:
    services = _bootstrap(tmp_path, monkeypatch)
    # DeepSeek / OpenAI o-series convention: reasoning delta in additional_kwargs.
    chunk = AIMessage(
        content="<final> answer",
        additional_kwargs={"reasoning_content": "deliberating..."},
    )
    result = run_turn(services=services, prompt="hi", chat_model=FakeChatModel([chunk]))
    thinking_events = [e for e in result.events if e.kind == "assistant_thinking"]
    assert thinking_events
    assert thinking_events[0].payload.get("text") == "deliberating..."


def test_tool_result_payload_has_line_count(tmp_path: Path, monkeypatch) -> None:
    services = _bootstrap(tmp_path, monkeypatch)
    workspace = Path(services.settings.workspace)
    (workspace / "sample.txt").write_text("line1\nline2\nline3\n", encoding="utf-8")
    model = FakeChatModel(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "read_file",
                        "args": {"path": "sample.txt"},
                        "id": "tool-1",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="<final> done"),
        ]
    )
    result = run_turn(services=services, prompt="read it", chat_model=model, mode="bypass")
    results = [e for e in result.events if e.kind == "tool_result"]
    assert results, "expected a tool_result event"
    payload = results[0].payload
    assert payload.get("line_count") == 3
    assert payload.get("content_length") >= len("line1\nline2\nline3")


# ---------------------------------------------------------------------------
# Thinking resolution + provider adapters
# ---------------------------------------------------------------------------


def _make_settings(**overrides):
    from minicode.platform.config import Settings

    base = dict(
        provider="openai",
        model="gpt-4o-mini",
        base_url=None,
        api_key_env="OPENAI_API_KEY",
        api_key="k",
        auto_mode="default",
        system_prompt="",
        workspace="/tmp",
        extra_headers={},
        extra_body={},
        thinking="auto",
        thinking_budget_tokens=2048,
    )
    base.update(overrides)
    return Settings(**base)


def test_resolve_thinking_enabled_respects_policy() -> None:
    from minicode.runtime.model_factory import resolve_thinking_enabled

    assert resolve_thinking_enabled(_make_settings(thinking="on"), "bypass") is True
    assert resolve_thinking_enabled(_make_settings(thinking="off"), "plan") is False


def test_resolve_thinking_enabled_auto_follows_mode() -> None:
    from minicode.runtime.model_factory import resolve_thinking_enabled

    auto = _make_settings(thinking="auto")
    assert resolve_thinking_enabled(auto, "plan") is True
    assert resolve_thinking_enabled(auto, "bypass") is False
    # default / auto / unknown modes -> let hybrid-thinking models self-regulate.
    assert resolve_thinking_enabled(auto, "default") is True
    assert resolve_thinking_enabled(auto, None) is True


def test_apply_thinking_anthropic_sets_budget_for_claude4() -> None:
    from minicode.runtime.model_factory import apply_thinking_to_kwargs

    settings = _make_settings(provider="anthropic", model="claude-sonnet-4-6")
    kwargs = {"model": "claude-sonnet-4-6"}
    apply_thinking_to_kwargs(settings, kwargs, enabled=True, budget_tokens=2048)
    assert kwargs["thinking"] == {"type": "enabled", "budget_tokens": 2048}


def test_apply_thinking_anthropic_skips_claude3() -> None:
    from minicode.runtime.model_factory import apply_thinking_to_kwargs

    settings = _make_settings(provider="anthropic", model="claude-3-5-sonnet")
    kwargs = {"model": "claude-3-5-sonnet"}
    apply_thinking_to_kwargs(settings, kwargs, enabled=True, budget_tokens=2048)
    assert "thinking" not in kwargs


def test_apply_thinking_qwen_uses_extra_body() -> None:
    from minicode.runtime.model_factory import apply_thinking_to_kwargs

    settings = _make_settings(model="qwen3-235b-a22b")
    kwargs = {"extra_body": {"existing": 1}}
    apply_thinking_to_kwargs(settings, kwargs, enabled=True, budget_tokens=0)
    assert kwargs["extra_body"]["enable_thinking"] is True
    assert kwargs["extra_body"]["existing"] == 1
    apply_thinking_to_kwargs(settings, kwargs, enabled=False, budget_tokens=0)
    assert kwargs["extra_body"]["enable_thinking"] is False


def test_apply_thinking_zhipu_uses_extra_body_object() -> None:
    from minicode.runtime.model_factory import apply_thinking_to_kwargs

    settings = _make_settings(model="glm-4.6")
    kwargs: dict = {}
    apply_thinking_to_kwargs(settings, kwargs, enabled=True, budget_tokens=0)
    assert kwargs["extra_body"]["thinking"] == {"type": "enabled"}


def test_apply_thinking_plain_openai_is_noop() -> None:
    from minicode.runtime.model_factory import apply_thinking_to_kwargs

    settings = _make_settings(model="gpt-4o-mini")
    kwargs: dict = {}
    apply_thinking_to_kwargs(settings, kwargs, enabled=True, budget_tokens=2048)
    assert kwargs == {}


