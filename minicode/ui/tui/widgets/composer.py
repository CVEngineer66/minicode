from __future__ import annotations

from dataclasses import dataclass

from textual.binding import Binding
from textual.message import Message
from textual.widgets import TextArea


class ComposerInput(TextArea):
    """Multiline composer with soft-wrap and auto-growing height."""

    BINDINGS = [
        *TextArea.BINDINGS,
        Binding("enter", "submit", "Submit", show=False, priority=True),
    ]

    MIN_CONTENT_LINES = 1
    MAX_CONTENT_LINES = 8
    _CHROME_LINES = 2

    @dataclass
    class Submitted(Message):
        composer: "ComposerInput"

        @property
        def control(self) -> "ComposerInput":
            return self.composer

    def __init__(self, **kwargs) -> None:
        super().__init__(
            "",
            soft_wrap=True,
            compact=True,
            show_line_numbers=False,
            highlight_cursor_line=False,
            **kwargs,
        )

    @property
    def value(self) -> str:
        return self.text

    def action_submit(self) -> None:
        self.post_message(self.Submitted(self))

    def set_value(self, value: str) -> None:
        self.load_text(value)
        self.move_cursor(self.document.end)
        self._sync_height()

    def clear_value(self) -> None:
        self.load_text("")
        self._sync_height()

    def submission_text(self) -> str:
        return self.text

    def edit(self, edit) -> object:
        result = super().edit(edit)
        self._sync_height()
        return result

    def load_text(self, text: str) -> None:
        super().load_text(text)
        self._sync_height()

    def _on_resize(self) -> None:
        super()._on_resize()
        self._sync_height()

    # ---------- arrow-key delegation ----------
    #
    # TextArea.BINDINGS binds ``up`` / ``down`` to cursor movement, so a
    # focused composer eats the arrow keys and the app-level bindings
    # (history prev / next, slash-suggest nav) never fire. Override the
    # action methods to delegate back to the app when:
    #   * slash-suggest dropdown is open — any up/down should navigate it;
    #   * cursor sits on the first (up) / last (down) row — user wants to
    #     scroll through prior prompts, not move inside a single-line field.
    # In every other case we fall through to ``super()`` so multi-line
    # editing still works.
    #
    # ``select=True`` means the user is shift-arrow-ing to extend a
    # selection — always keep that as cursor movement, never history nav.

    def _app_action(self, name: str) -> bool:
        app = getattr(self, "app", None)
        method = getattr(app, name, None)
        suggest_check = getattr(app, "_suggest_visible", None)
        if method is None:
            return False
        if callable(suggest_check) and suggest_check():
            method()
            return True
        return False

    def action_cursor_up(self, select: bool = False) -> None:  # type: ignore[override]
        if not select:
            if self._app_action("action_history_prev"):
                return
            row, _ = self.cursor_location
            if row == 0:
                app = getattr(self, "app", None)
                method = getattr(app, "action_history_prev", None)
                if method is not None:
                    method()
                    return
        super().action_cursor_up(select)

    def action_cursor_down(self, select: bool = False) -> None:  # type: ignore[override]
        if not select:
            if self._app_action("action_history_next"):
                return
            row, _ = self.cursor_location
            last_row = max(0, self.document.line_count - 1)
            if row >= last_row:
                app = getattr(self, "app", None)
                method = getattr(app, "action_history_next", None)
                if method is not None:
                    method()
                    return
        super().action_cursor_down(select)

    def _sync_height(self) -> None:
        wrap_width = max(1, self.wrap_width)
        self.wrapped_document.wrap(wrap_width, self.indent_width)
        content_lines = max(self.MIN_CONTENT_LINES, self.wrapped_document.height)
        content_lines = min(content_lines, self.MAX_CONTENT_LINES)
        self.styles.height = content_lines + self._CHROME_LINES
