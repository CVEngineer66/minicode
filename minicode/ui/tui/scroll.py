"""Scroll containers.

Plain wrappers around Textual's ``VerticalScroll`` that take over mouse
wheel handling so transcript / command-output regions don't forward
wheel events to parent containers.
"""

from __future__ import annotations

from textual import events
from textual.containers import VerticalScroll


class WheelScrollView(VerticalScroll):
    """Scroll view that owns mouse-wheel input for its content region."""

    SCROLL_LINES = 3

    def on_mouse_scroll_up(self, event: events.MouseScrollUp) -> None:
        self.scroll_relative(y=-self.SCROLL_LINES, animate=False)
        event.stop()

    def on_mouse_scroll_down(self, event: events.MouseScrollDown) -> None:
        self.scroll_relative(y=self.SCROLL_LINES, animate=False)
        event.stop()


class TranscriptScroll(WheelScrollView):
    """Transcript container with explicit mouse-wheel ownership."""
