"""Slash-command suggestion dropdown."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rich.text import Text
from textual import events
from textual.widgets import Static

from .commands import find_matching_slash_commands


@dataclass(frozen=True, slots=True)
class SlashSuggestOption:
    usage: str
    prompt: str


class SlashSuggest(Static):
    """Floating dropdown listing slash commands that match the current input."""

    MAX_VISIBLE = 8
    # Border (round) adds 2 rows of chrome around the options list.
    _BORDER_ROWS = 2

    def __init__(self, **kwargs: Any) -> None:
        # Non-empty sentinel avoids a None-visual crash when Textual tries
        # to render Static("") during layout warm-up (see statusline.py).
        super().__init__(" ", markup=False, **kwargs)
        self._options: list[SlashSuggestOption] = []
        self.highlighted: int | None = None
        self._window_start: int = 0

    # ---- public API used by app.py and tests ----

    def update_matches(self, input_text: str) -> None:
        commands = find_matching_slash_commands(input_text) if input_text.startswith("/") else []
        self._options = [
            SlashSuggestOption(usage=c.usage, prompt=_format_slash_suggest_prompt(c))
            for c in commands
        ]
        if not self._options:
            self.hide_suggestions()
            return
        self.highlighted = 0
        self._window_start = 0
        self.display = True
        self._resize_to_content()
        self._refresh_content()

    def _resize_to_content(self) -> None:
        """Grow the dropdown to fit options exactly, capped at MAX_VISIBLE."""
        visible_rows = min(len(self._options), self.MAX_VISIBLE)
        self.styles.max_height = visible_rows + self._BORDER_ROWS

    def hide_suggestions(self) -> None:
        self._options = []
        self.highlighted = None
        self._window_start = 0
        self.display = False
        self.update(" ")

    def move(self, delta: int) -> None:
        if not self._options:
            return
        current = self.highlighted if self.highlighted is not None else 0
        self.highlighted = (current + delta) % len(self._options)
        self._ensure_visible()
        self._refresh_content()

    def on_mouse_scroll_up(self, event: events.MouseScrollUp) -> None:
        if not self.is_visible:
            return
        self.move(-1)
        event.stop()

    def on_mouse_scroll_down(self, event: events.MouseScrollDown) -> None:
        if not self.is_visible:
            return
        self.move(1)
        event.stop()

    @property
    def is_visible(self) -> bool:
        return bool(self.display and self._options)

    @property
    def option_count(self) -> int:
        return len(self._options)

    @property
    def highlighted_option(self) -> SlashSuggestOption | None:
        if self.highlighted is None or not (0 <= self.highlighted < len(self._options)):
            return None
        return self._options[self.highlighted]

    @property
    def window_start(self) -> int:
        return self._window_start

    def current_usage(self) -> str | None:
        option = self.highlighted_option
        return option.usage if option else None

    def get_option_at_index(self, index: int) -> SlashSuggestOption:
        return self._options[index]

    def visible_prompts(self) -> list[str]:
        return [o.prompt for o in self._visible()]

    # ---- internal ----

    def _visible(self) -> list[SlashSuggestOption]:
        return self._options[self._window_start : self._window_start + self.MAX_VISIBLE]

    def _ensure_visible(self) -> None:
        if self.highlighted is None:
            return
        if self.highlighted < self._window_start:
            self._window_start = self.highlighted
            return
        end = self._window_start + self.MAX_VISIBLE
        if self.highlighted >= end:
            self._window_start = self.highlighted - self.MAX_VISIBLE + 1

    def _refresh_content(self) -> None:
        lines = Text()
        visible = self._visible()
        for offset, option in enumerate(visible):
            index = self._window_start + offset
            prefix = "› " if index == self.highlighted else "  "
            line = Text(prefix + option.prompt)
            line.stylize("black on bright_cyan" if index == self.highlighted else "white")
            lines.append_text(line)
            if offset < len(visible) - 1:
                lines.append("\n")
        self.update(lines)


def _format_slash_suggest_prompt(command: Any) -> str:
    return f"{command.usage:<26}  {command.description}"
