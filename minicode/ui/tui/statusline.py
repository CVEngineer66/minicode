"""Status line widget: constant-height row for the activity spinner.

Kept at ``height: 1`` via CSS and never toggles ``display`` — that was the
root cause of the old ``ActivityIndicator`` bouncing the Input row on
every event. When idle we render an empty ``Text``; the row stays, the
layout below stays put.
"""

from __future__ import annotations

from typing import Any

from rich.markup import escape
from rich.text import Text
from textual.widgets import Static


class StatusLine(Static):
    SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
    TICK_SECONDS = 0.12

    def __init__(self, **kwargs: Any) -> None:
        # Non-empty initial content: Textual's visual pipeline turns
        # Static("") into a None visual and crashes layout. A single
        # space keeps the 1-row height stable without rendering any
        # glyph the user can see.
        super().__init__(" ", markup=False, **kwargs)
        self._activity: str = ""
        self._frame: int = 0
        self._timer: Any = None

    def set_activity(self, activity: str) -> None:
        activity = activity or ""
        if activity == self._activity:
            return
        self._activity = activity
        if activity:
            if self._timer is None:
                self._timer = self.set_interval(self.TICK_SECONDS, self._tick)
        else:
            if self._timer is not None:
                self._timer.stop()
                self._timer = None
        self.update(self._build_renderable())

    def _tick(self) -> None:
        self._frame = (self._frame + 1) % len(self.SPINNER)
        self.update(self._build_renderable())

    def _build_renderable(self) -> Text | str:
        if not self._activity:
            return " "
        glyph = self.SPINNER[self._frame]
        return Text.from_markup(
            f"[bright_yellow bold]{glyph}[/] [italic]{escape(self._activity)}[/]"
        )


# Back-compat alias: legacy callers imported ``ActivityIndicator`` from
# ``.indicator``; the whole module went away and this name is the
# replacement so external imports keep working.
ActivityIndicator = StatusLine
