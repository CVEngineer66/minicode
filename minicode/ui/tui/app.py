"""Textual-based TUI for MiniCode Next.

This module hosts ``MiniCodeApp`` — orchestration, input handling, turn
runner, slash-command routing — plus the CLI entry point. Widgets,
screens, and parsers live in sibling modules. The symbols tests import
from here are re-exported below.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Callable

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, VerticalScroll
from textual.reactive import reactive
from textual.widgets import Footer, Header

from minicode.runtime.runner import run_turn

from .commands import SLASH_COMMANDS
from .constants import AUTO_MODES, ENTRY_KINDS, EVENT_KINDS, IDS, entry_class
from .dispatcher import SlashDispatcher
from .entries import EntryView, _find_running_tool_entry, _parse_read_args
from .parser import parse_input
from .approval_entry import ApprovalEntry
from .screens import CommandOutputScreen, CommandOutputScroll, PickerScreen
from .scroll import TranscriptScroll, WheelScrollView
from .session_io import (
    _entries_from_messages,
    _history_from_messages,
    _make_history_repo,
    _resolve_thread_id,
    _tool_kwargs_from_call,
)
from .statusline import ActivityIndicator, StatusLine
from .suggest import SlashSuggest, SlashSuggestOption, _format_slash_suggest_prompt
from .widgets.composer import ComposerInput

__all__ = [
    "ActivityIndicator",
    "ApprovalEntry",
    "CommandOutputScreen",
    "CommandOutputScroll",
    "EntryView",
    "MiniCodeApp",
    "PickerScreen",
    "SlashSuggest",
    "SlashSuggestOption",
    "StatusLine",
    "TranscriptScroll",
    "WheelScrollView",
    "_entries_from_messages",
    "_find_running_tool_entry",
    "_format_slash_suggest_prompt",
    "_history_from_messages",
    "_make_history_repo",
    "_parse_read_args",
    "_resolve_thread_id",
    "_tool_kwargs_from_call",
    "run_tui_app",
]


class MiniCodeApp(App):
    # Layout invariant: the bottom-panel Container wraps StatusLine,
    # SlashSuggest, and Input as a single dock-bottom region. StatusLine
    # is always ``height: 1`` and never toggles ``display`` so Input
    # stays at a stable pixel row — the old ``ActivityIndicator``
    # show/hide caused per-event relayout that visually "stacked" Input.
    CSS = f"""
    Screen {{
        background: $surface;
    }}
    #{IDS.transcript} {{
        height: 1fr;
        padding: 0 1;
        border-top: solid $accent 50%;
        border-bottom: solid $accent 50%;
    }}
    #{IDS.bottom_panel} {{
        height: auto;
    }}
    #{IDS.slash_suggest} {{
        height: auto;
        background: $panel;
        color: $text;
        border: round $accent;
        padding: 0 1;
        margin: 0 1;
        display: none;
    }}
    ComposerInput#{IDS.prompt_input} {{
        margin: 0;
        border: round $accent;
        padding: 0 1;
        background: $panel-darken-1;
        scrollbar-size-vertical: 0;
        scrollbar-size-horizontal: 0;
    }}
    EntryView {{
        height: auto;
        margin-bottom: 1;
        padding: 0 1;
    }}
    .{entry_class(ENTRY_KINDS.user)} {{
        background: $boost;
        border-left: thick $primary;
        color: $text;
        padding: 0 1;
    }}
    .{entry_class(ENTRY_KINDS.assistant)} {{ color: $text; }}
    .{entry_class(ENTRY_KINDS.tool)} {{ color: $warning; }}
    .{entry_class(ENTRY_KINDS.progress)} {{ color: $warning 60%; }}
    .{entry_class(ENTRY_KINDS.system)} {{ color: $text-muted; }}
    .{entry_class(ENTRY_KINDS.thinking)} {{ color: $text-muted 70%; }}
    .entry-approval {{
        border-left: thick $error;
        background: $boost;
        padding: 0 1;
    }}
    .cmd-output-box {{
        padding: 1 2;
        background: $surface;
    }}
    .picker-box {{
        width: 60%;
        max-width: 80;
        height: auto;
        border: round $accent;
        background: $panel;
        padding: 1 2;
    }}
    #{IDS.picker_options} {{
        height: auto;
    }}
    """

    BINDINGS = [
        Binding("ctrl+c", "quit",              "Quit",           priority=True),
        Binding("escape", "cancel_turn",       "Cancel turn"),
        Binding("ctrl+o", "toggle_expand_all", "Toggle details"),
        Binding("ctrl+l", "clear_transcript",  "Clear"),
        Binding("up",     "history_prev",      "History prev", show=False),
        Binding("down",   "history_next",      "History next", show=False),
        Binding("tab",    "complete_slash",    "Complete",     show=False, priority=True),
    ]

    expand_all: reactive[bool] = reactive(False)

    def __init__(self, services: Any, initial_thread_id: str | None = None) -> None:
        super().__init__()
        self.services = services
        self.thread_id: str | None = initial_thread_id
        self.mode: str = services.settings.auto_mode
        self.dispatcher = SlashDispatcher(services)
        self.streaming_entry: EntryView | None = None
        self.streaming_thinking: EntryView | None = None
        self._history_repo = _make_history_repo(services)
        self._history: list[str] = (
            self._history_repo.load() if self._history_repo is not None else []
        )
        self._history_index: int = len(self._history)
        self._history_draft: str = ""
        self._ignore_next_input_change = 0
        self._busy: bool = False
        # Bumped by ``action_cancel_turn``; stale sink callbacks from a
        # detached turn check this and no-op.
        self._turn_epoch: int = 0
        self._turn_started_at: float | None = None
        self._active_tools: dict[str, str] = {}
        self._approval_entry: ApprovalEntry | None = None
        self._handlers: dict[str, Callable[[dict], None]] = {
            EVENT_KINDS.model_call_start:   self._on_model_call_start,
            EVENT_KINDS.assistant_thinking: self._on_assistant_thinking,
            EVENT_KINDS.assistant_token:    self._on_assistant_token,
            EVENT_KINDS.assistant_message:  self._on_assistant_message,
            EVENT_KINDS.progress:           self._on_progress,
            EVENT_KINDS.tool_start:         self._on_tool_start,
            EVENT_KINDS.tool_result:        self._on_tool_result,
            EVENT_KINDS.context_compacted:  self._on_context_compacted,
            EVENT_KINDS.session_finalized:  self._on_session_finalized,
        }

    # ---------- compose & mount ----------

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        yield TranscriptScroll(id=IDS.transcript)
        with Container(id=IDS.bottom_panel):
            yield SlashSuggest(id=IDS.slash_suggest)
            yield ComposerInput(
                placeholder="Type a message or /help for commands",
                id=IDS.prompt_input,
            )
        yield Footer()

    def on_mount(self) -> None:
        self.title = "MiniCode Next"
        self.sub_title = self._format_subtitle()
        self.query_one(ComposerInput).focus()
        if self.thread_id:
            self._replay_thread(self.thread_id)

    def _replay_thread(self, thread_id: str) -> None:
        resolved, error = _resolve_thread_id(self.services, thread_id)
        if error is not None:
            self._add_entry("system", body=error)
            self._follow_tail()
            return
        self.thread_id = resolved
        try:
            messages = self.services.sessions.load_messages(resolved)
        except Exception as exc:  # noqa: BLE001
            self._add_entry("system", body=f"Failed to load session {resolved[:12]}: {exc}")
            self._follow_tail()
            return
        if not messages:
            self._add_entry("system", body=f"Session {resolved[:12]} has no messages yet.")
            self._follow_tail()
            return
        for kind, kwargs in _entries_from_messages(messages):
            self._add_entry(kind, **kwargs)
        self._restore_history_from_messages(messages)
        self._add_entry(
            "system", body=f"Resumed {resolved[:12]} ({len(messages)} messages)"
        )
        self.query_one(ComposerInput).focus()
        self._follow_tail()

    # ---------- transcript helpers ----------

    def _add_entry(self, kind: str, **kwargs: Any) -> EntryView:
        entry = EntryView(kind=kind, **kwargs)
        self.query_one(f"#{IDS.transcript}", VerticalScroll).mount(entry)
        return entry

    def _follow_tail(self) -> None:
        """Scroll transcript to bottom. Two passes — current layout, then
        after-refresh to catch Rich Markdown height settling."""
        try:
            container = self.query_one(f"#{IDS.transcript}", VerticalScroll)
        except Exception:
            return
        container.scroll_end(animate=False)
        self.call_after_refresh(container.scroll_end, animate=False)

    def _is_near_bottom(self) -> bool:
        try:
            c = self.query_one(f"#{IDS.transcript}", VerticalScroll)
        except Exception:
            return True
        return c.scroll_y >= c.max_scroll_y - 3

    # ---------- actions ----------

    def action_clear_transcript(self) -> None:
        self.streaming_entry = None
        self._detach_streaming_thinking(drop_if_empty=False)
        container = self.query_one(f"#{IDS.transcript}", VerticalScroll)
        for widget in list(container.query(EntryView)):
            widget.remove()

    def action_toggle_expand_all(self) -> None:
        self.expand_all = not self.expand_all
        for entry in self.query(EntryView):
            if entry.kind in ("tool", "thinking"):
                entry.set_expanded(self.expand_all)

    def action_cancel_turn(self) -> None:
        # Escape precedence: close suggest > cancel turn > no-op. The
        # dropdown has no close key of its own, so we reuse Escape.
        if self._suggest_visible():
            self.query_one(SlashSuggest).hide_suggestions()
            return
        # Best-effort: we can't kill the API call, but bumping the epoch
        # makes any trailing sink events no-op, and ``_release_turn_state``
        # will still run when the thread returns.
        if not self._busy:
            return
        self._turn_epoch += 1
        self.streaming_entry = None
        self._detach_streaming_thinking(drop_if_empty=False)
        self._active_tools.clear()
        elapsed = self._consume_turn_elapsed()
        suffix = f" after {self._format_elapsed(elapsed)}" if elapsed is not None else ""
        self._add_entry("system", body="[cancelled — waiting for the model to unwind]")
        self._follow_tail()

    # ---------- history nav ----------

    def action_history_prev(self) -> None:
        if self._suggest_visible():
            self.query_one(SlashSuggest).move(-1)
            return
        if len(self.screen_stack) > 1:
            return
        if not self._history or self._history_index <= 0:
            return
        if self._history_index == len(self._history):
            self._history_draft = self.query_one(ComposerInput).value
        self._history_index -= 1
        self._set_prompt_value(self._history[self._history_index], show_suggestions=False)

    def action_history_next(self) -> None:
        if self._suggest_visible():
            self.query_one(SlashSuggest).move(1)
            return
        if len(self.screen_stack) > 1:
            return
        if self._history_index >= len(self._history):
            return
        self._history_index += 1
        next_value = (
            self._history[self._history_index]
            if self._history_index < len(self._history)
            else self._history_draft
        )
        self._set_prompt_value(next_value, show_suggestions=False)

    def _suggest_visible(self) -> bool:
        try:
            return self.query_one(SlashSuggest).is_visible
        except Exception:
            return False

    # ---------- input ----------

    def on_text_area_changed(self, event: ComposerInput.Changed) -> None:
        if event.text_area.id != IDS.prompt_input:
            return
        if self._ignore_next_input_change:
            self._ignore_next_input_change -= 1
            return
        try:
            suggest = self.query_one(SlashSuggest)
        except Exception:
            return
        self._history_index = len(self._history)
        self._history_draft = event.text_area.text
        suggest.update_matches(event.text_area.text)

    def action_complete_slash(self) -> None:
        if not self._suggest_visible():
            return
        usage = self.query_one(SlashSuggest).current_usage()
        if not usage:
            return
        head = usage.split(" <")[0]
        self._set_prompt_value(head + " " if " <" in usage else head, show_suggestions=False)

    def on_composer_input_submitted(self, event: ComposerInput.Submitted) -> None:
        if self._suggest_visible():
            known_heads = {c.usage.split(" ")[0] for c in SLASH_COMMANDS}
            head = event.composer.text.strip().split(" ", 1)[0]
            if head not in known_heads:
                self.action_complete_slash()
                return
        # Reject concurrent submissions; a second ``run_turn`` on the
        # same thread_id would race on the SqliteSaver checkpoint.
        if self._busy:
            return
        raw_text = event.composer.submission_text()
        text = raw_text.strip()
        event.composer.clear_value()
        self.query_one(SlashSuggest).hide_suggestions()
        if not text:
            return
        self._record_history_entry(text)

        cmd, args = parse_input(text)
        if cmd in ("quit", "exit"):
            self.exit()
            return
        if cmd == "clear":
            self.action_clear_transcript()
            return
        if cmd != "prompt":
            self._handle_slash(text, cmd, args)
            return

        self._add_entry("user", body=text)
        self._follow_tail()
        self._busy = True
        self._launch_turn(prompt=text)

    # ---------- slash commands ----------

    def _handle_slash(self, raw: str, cmd: str, args: list[str]) -> None:
        if cmd == "resume":
            self._do_resume(args[0]) if args else self._open_resume_picker()
            return
        if cmd == "mode":
            self._apply_mode(args[0]) if args else self._open_mode_picker()
            return
        if cmd == "model":
            self._apply_model(args[0]) if args else self._open_model_picker()
            return
        result = self.dispatcher.dispatch(raw)
        if result.quit:
            self.exit()
            return
        self.push_screen(CommandOutputScreen(f"/{cmd}", result.output or f"/{cmd} ok"))

    def _open_picker(
        self,
        title: str,
        options: list[tuple[str, str]],
        apply: Callable[[str], None],
    ) -> None:
        def _cb(value: str | None) -> None:
            if value:
                apply(value)

        self.push_screen(PickerScreen(title, options), _cb)

    def _open_mode_picker(self) -> None:
        options = [
            (m, m + ("  (current)" if m == self.mode else "")) for m in AUTO_MODES
        ]
        self._open_picker("Select mode", options, self._apply_mode)

    def _apply_mode(self, value: str) -> None:
        normalized = value.strip()
        auto = getattr(self.services, "auto", None)
        if auto is not None:
            try:
                auto.set_mode(normalized)
            except ValueError as exc:
                self.push_screen(CommandOutputScreen("/mode", f"Invalid mode: {exc}"))
                return
            self.mode = auto.get_mode().value
        else:
            if normalized not in AUTO_MODES:
                self.push_screen(CommandOutputScreen("/mode", f"Invalid mode: {normalized}"))
                return
            self.mode = normalized
        self.services.settings.auto_mode = self.mode
        self.sub_title = self._format_subtitle()

    def _format_subtitle(self) -> str:
        settings = self.services.settings
        if settings.provider and settings.model:
            label = f"{settings.provider}:{settings.model}"
        else:
            label = "(no model)"
        return f"{label}  -  mode: {self.mode}"

    def _open_model_picker(self) -> None:
        settings = self.services.settings
        catalog = getattr(settings, "catalog", None)
        identifiers = catalog.all_identifiers() if catalog is not None else []
        if not identifiers:
            self.push_screen(
                CommandOutputScreen(
                    "/model",
                    "No models configured. Edit ~/.minicode/config.json "
                    "(see ~/.minicode/config.example.json for a template), "
                    "then restart MiniCode.",
                )
            )
            return
        current = f"{settings.provider}:{settings.model}" if settings.provider and settings.model else ""
        options = [
            (ident, ident + ("  (current)" if ident == current else ""))
            for ident in identifiers
        ]
        self._open_picker("Select model", options, self._apply_model)

    def _apply_model(self, value: str) -> None:
        settings = self.services.settings
        catalog = getattr(settings, "catalog", None)
        resolved = catalog.resolve(value) if catalog is not None else None
        if resolved is None:
            self.push_screen(CommandOutputScreen("/model", f"Unknown model: {value}"))
            return
        entry, model = resolved
        settings.apply_provider(entry, model)
        try:
            from minicode.platform.config import save_current_model
            save_current_model(self.services.paths, f"{entry.name}:{model}")
        except OSError:
            # Disk write failed — the session still has the new model,
            # user just loses persistence across restart.
            pass
        ctx = getattr(self.services, "context", None)
        if ctx is not None:
            try:
                ctx.set_model(model)
            except Exception:
                pass
        self.sub_title = self._format_subtitle()

    def _open_resume_picker(self) -> None:
        sessions = self.services.sessions.list_sessions(
            workspace=self.services.settings.workspace
        )
        if not sessions:
            self.push_screen(CommandOutputScreen("/resume", "No sessions in this workspace."))
            return
        options = [
            (s.thread_id, f"{s.title or '(untitled)'}  -  {s.thread_id[:12]}")
            for s in sessions[:50]
        ]
        self._open_picker("Resume session", options, self._do_resume)

    def _do_resume(self, target: str) -> None:
        resolved, error = _resolve_thread_id(self.services, target)
        if error is not None:
            self.push_screen(CommandOutputScreen("/resume", error))
            return
        messages = self.services.sessions.load_messages(resolved)
        self.thread_id = resolved
        self.action_clear_transcript()
        for kind, kwargs in _entries_from_messages(messages):
            self._add_entry(kind, **kwargs)
        self._restore_history_from_messages(messages)
        self._add_entry("system", body=f"Resumed {resolved[:12]}")
        self.query_one(ComposerInput).focus()
        self._follow_tail()

    # ---------- prompt helpers ----------

    def _set_prompt_value(self, value: str, *, show_suggestions: bool) -> None:
        # Preserve multiline history entries; only strip qwen hybrid-thinking
        # suffixes that were historically appended as control hints.
        for suffix in ("\n\n/think", "\n\n/no_think"):
            if value.endswith(suffix):
                value = value[: -len(suffix)]
                break
        self._ignore_next_input_change += 1
        input_widget = self.query_one(ComposerInput)
        input_widget.set_value(value)
        suggest = self.query_one(SlashSuggest)
        if show_suggestions:
            suggest.update_matches(value)
        else:
            suggest.hide_suggestions()

    def _record_history_entry(self, text: str) -> None:
        stripped = text.strip()
        if not stripped:
            return
        if not self._history or self._history[-1] != stripped:
            self._history.append(stripped)
        if self._history_repo is not None:
            self._history_repo.append(stripped)
        self._history_index = len(self._history)

    def _restore_history_from_messages(self, messages: list[Any]) -> None:
        restored = _history_from_messages(messages)
        if not restored:
            self._history_index = len(self._history)
            return
        self._history = restored
        if self._history_repo is not None:
            for entry in restored:
                self._history_repo.append(entry)
        self._history_index = len(self._history)

    # ---------- turn runner ----------

    def _launch_turn(self, prompt: str | None = None, resume: dict | None = None) -> None:
        turn_epoch = self._turn_epoch
        self._turn_started_at = time.perf_counter()

        def sink(event: Any) -> None:
            def apply_if_current() -> None:
                if self._turn_epoch != turn_epoch:
                    return
                self._apply_graph_event(event)

            self.call_from_thread(apply_if_current)

        def work_fn() -> None:
            try:
                result = run_turn(
                    services=self.services,
                    prompt=prompt,
                    thread_id=self.thread_id,
                    mode=self.mode,
                    resume=resume,
                    event_sink=sink,
                )
                self.call_from_thread(self._on_turn_finished, result, turn_epoch)
            except BaseException as exc:  # noqa: BLE001
                self.call_from_thread(
                    self._on_turn_error,
                    f"{type(exc).__name__}: {exc}",
                    turn_epoch,
                )

        threading.Thread(target=work_fn, daemon=True).start()

    def _apply_graph_event(self, event: Any) -> None:
        was_following = self._is_near_bottom()
        kind = getattr(event, "kind", "")
        payload = getattr(event, "payload", {}) or {}
        handler = self._handlers.get(kind)
        if handler is not None:
            handler(payload)
        if was_following:
            self._follow_tail()

    # ---------- per-event handlers ----------

    def _on_model_call_start(self, _payload: dict) -> None:
        if self.streaming_thinking is None:
            self.streaming_thinking = self._add_entry("thinking", expanded=True)

    def _on_assistant_thinking(self, payload: dict) -> None:
        chunk = payload.get("text", "")
        if not chunk:
            return
        if self.streaming_thinking is None:
            self.streaming_thinking = self._add_entry(
                "thinking", body=chunk, expanded=True
            )
        else:
            self.streaming_thinking.append_chunk(chunk)

    def _on_assistant_token(self, payload: dict) -> None:
        chunk = payload.get("text", "")
        if not chunk:
            return
        self._detach_streaming_thinking(drop_if_empty=True)
        if self.streaming_entry is None:
            self.streaming_entry = self._add_entry("assistant", body=chunk, streaming=True)
        else:
            self.streaming_entry.append_chunk(chunk)

    def _on_assistant_message(self, payload: dict) -> None:
        text = payload.get("text", "")
        self._detach_streaming_thinking(drop_if_empty=True)
        if self.streaming_entry is not None:
            self.streaming_entry.set_body(text)
            self.streaming_entry = None
        else:
            self._add_entry("assistant", body=text)

    def _on_progress(self, payload: dict) -> None:
        self.streaming_entry = None
        self._detach_streaming_thinking(drop_if_empty=True)
        self._add_entry("progress", body=payload.get("text", ""))

    def _on_tool_start(self, payload: dict) -> None:
        self.streaming_entry = None
        self._detach_streaming_thinking(drop_if_empty=True)
        tool_name = str(payload.get("tool_name", "tool"))
        tool_call_id = str(payload.get("tool_call_id", ""))
        if tool_call_id:
            self._active_tools[tool_call_id] = tool_name
        self._add_entry(
            "tool",
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            tool_args=str(payload.get("arguments", {})),
            tool_status="running",
            expanded=self.expand_all,
        )

    def _on_tool_result(self, payload: dict) -> None:
        tool_call_id = str(payload.get("tool_call_id", ""))
        name = str(payload.get("tool_name", ""))
        ok = bool(payload.get("ok"))
        content = str(payload.get("content", ""))
        line_count = int(payload.get("line_count", 0) or 0)
        content_length = int(payload.get("content_length", 0) or 0)
        widget = _find_running_tool_entry(
            self.query(EntryView), tool_call_id=tool_call_id, tool_name=name
        )
        if widget is not None:
            widget.set_tool_result(
                "success" if ok else "error",
                content,
                line_count=line_count,
                content_length=content_length,
            )
        if tool_call_id:
            self._active_tools.pop(tool_call_id, None)

    def _on_context_compacted(self, payload: dict) -> None:
        self._add_entry(
            "system",
            body=f"context compacted: removed {payload.get('removed', 0)} messages",
        )

    def _on_session_finalized(self, _payload: dict) -> None:
        self.streaming_entry = None
        self._detach_streaming_thinking(drop_if_empty=True)
        self._active_tools.clear()

    # ---------- turn completion ----------

    def _on_turn_finished(self, result: Any, turn_epoch: int = 0) -> None:
        if turn_epoch != self._turn_epoch:
            self._release_turn_state()
            return
        self.thread_id = getattr(result, "thread_id", self.thread_id)
        interrupt = getattr(result, "interrupt", None)
        if interrupt:
            prompt_kind = str(interrupt.get("prompt_kind") or "approval")
            label = "Awaiting user choice" if prompt_kind == "choice" else "Awaiting approval"
            self._add_turn_timing_entry(label, preposition="after")
            self._release_turn_state()
            self._busy = False
            self._show_approval_inline(interrupt)
            return
        error = getattr(result, "error", None)
        if error:
            elapsed = self._consume_turn_elapsed()
            prefix = f"Failed after {self._format_elapsed(elapsed)}" if elapsed is not None else "Failed"
            self._add_entry("system", body=f"{prefix}: {error}")
            self._follow_tail()
            self._release_turn_state()
            return
        self._add_turn_timing_entry("Completed", preposition="in")
        self._release_turn_state()

    def _on_turn_error(self, msg: str, turn_epoch: int = 0) -> None:
        if turn_epoch != self._turn_epoch:
            self._release_turn_state()
            return
        elapsed = self._consume_turn_elapsed()
        prefix = f"Failed after {self._format_elapsed(elapsed)}" if elapsed is not None else "Failed"
        self._add_entry("system", body=f"{prefix}: {msg}")
        self._release_turn_state()
        self._follow_tail()

    def _release_turn_state(self) -> None:
        self._busy = False
        self.streaming_entry = None
        self._detach_streaming_thinking(drop_if_empty=True)
        self._active_tools.clear()

    def _on_approval_decision(self, decision: dict | None) -> None:
        if decision is None:
            decision = {"decision": "deny"}
        self._busy = True
        self._launch_turn(resume=decision)

    def _show_approval_inline(self, payload: dict) -> None:
        entry = ApprovalEntry(payload)
        self.query_one(f"#{IDS.transcript}", VerticalScroll).mount(entry)
        self._approval_entry = entry
        self._follow_tail()
        entry.focus()

    def on_approval_entry_approval_decided(
        self, event: ApprovalEntry.ApprovalDecided
    ) -> None:
        event.stop()
        entry = self._approval_entry
        if entry is not None:
            entry.remove()
            self._approval_entry = None
        self.query_one(ComposerInput).focus()
        self._on_approval_decision(event.decision)

    def _detach_streaming_thinking(self, *, drop_if_empty: bool) -> None:
        entry = self.streaming_thinking
        if entry is None:
            return
        if drop_if_empty and not entry.body.strip():
            entry.remove()
        self.streaming_thinking = None

    def _consume_turn_elapsed(self) -> float | None:
        started_at = self._turn_started_at
        self._turn_started_at = None
        if started_at is None:
            return None
        return max(0.0, time.perf_counter() - started_at)

    def _add_turn_timing_entry(self, label: str, *, preposition: str) -> None:
        elapsed = self._consume_turn_elapsed()
        if elapsed is None:
            return
        self._add_entry("system", body=f"{label} {preposition} {self._format_elapsed(elapsed)}")
        self._follow_tail()

    @staticmethod
    def _format_elapsed(elapsed: float) -> str:
        if elapsed < 1:
            return f"{elapsed * 1000:.0f} ms"
        if elapsed < 10:
            return f"{elapsed:.1f}s"
        return f"{elapsed:.0f}s"


def run_tui_app(services: Any, initial_thread_id: str | None = None) -> int:
    app = MiniCodeApp(services, initial_thread_id=initial_thread_id)
    app.run()
    return 0
