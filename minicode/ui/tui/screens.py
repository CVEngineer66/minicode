"""Modal screens: command-output viewer, arrow-key picker."""

from __future__ import annotations

from typing import Any

from rich.markup import escape
from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container
from textual.screen import ModalScreen
from textual.widgets import OptionList, Static
from textual.widgets.option_list import Option

from .scroll import WheelScrollView


def _scroll_by(container: Any, *, rows: int = 0, page: int = 0, home: bool = False, end: bool = False) -> None:
    if container is None:
        return
    if home:
        container.scroll_home(animate=False)
        return
    if end:
        container.scroll_end(animate=False)
        return
    if page:
        if page < 0:
            container.scroll_page_up(animate=False)
        else:
            container.scroll_page_down(animate=False)
        return
    if rows:
        container.scroll_relative(y=rows, animate=False)


# Shared keyboard bindings for any modal that wraps a scrollable content
# region. Subclasses mix in ``ScrollableModalMixin`` and prepend these to
# their own BINDINGS list.
SCROLL_BINDINGS: list[Binding] = [
    Binding("up",       "modal_scroll_up",   show=False, priority=True),
    Binding("down",     "modal_scroll_down", show=False, priority=True),
    Binding("pageup",   "modal_page_up",     show=False, priority=True),
    Binding("pagedown", "modal_page_down",   show=False, priority=True),
    Binding("home",     "modal_scroll_home", show=False, priority=True),
    Binding("end",      "modal_scroll_end",  show=False, priority=True),
]


class ScrollableModalMixin:
    """Provides ``modal_scroll_*`` actions for a modal's scrollable region.

    Subclasses implement ``_scroll_target()`` to return the container that
    receives scroll operations (typically a ``WheelScrollView``). The
    mixin owns the key → action wiring; the caller still composes
    BINDINGS to retain screen-specific keys alongside ``SCROLL_BINDINGS``.
    """

    def _scroll_target(self) -> Any:
        raise NotImplementedError

    def action_modal_scroll_up(self) -> None:   _scroll_by(self._scroll_target(), rows=-3)
    def action_modal_scroll_down(self) -> None: _scroll_by(self._scroll_target(), rows=3)
    def action_modal_page_up(self) -> None:     _scroll_by(self._scroll_target(), page=-1)
    def action_modal_page_down(self) -> None:   _scroll_by(self._scroll_target(), page=1)
    def action_modal_scroll_home(self) -> None: _scroll_by(self._scroll_target(), home=True)
    def action_modal_scroll_end(self) -> None:  _scroll_by(self._scroll_target(), end=True)


class CommandOutputScroll(WheelScrollView):
    """Scrollable slash-command output region.

    Key bindings live on ``CommandOutputScreen`` (via ``ScrollableModalMixin``)
    so the Screen is the single source of truth for scroll shortcuts.
    """


class CommandOutputScreen(ScrollableModalMixin, ModalScreen[None]):
    """Full-screen overlay showing slash-command output. ESC returns."""

    BINDINGS = SCROLL_BINDINGS + [
        Binding("escape", "dismiss", "Back", priority=True),
    ]

    def __init__(self, title: str, body: str) -> None:
        super().__init__()
        self._title_text = title
        self._body_text = body

    def compose(self) -> ComposeResult:
        with CommandOutputScroll(classes="cmd-output-box"):
            yield Static(
                Text.from_markup(
                    f"[bold bright_cyan]{escape(self._title_text)}[/]  [dim](Esc to return)[/]"
                )
            )
            yield Static(Text(self._body_text or ""))

    def on_mount(self) -> None:
        self.query_one(CommandOutputScroll).focus()

    def _scroll_target(self) -> Any:
        try:
            return self.query_one(CommandOutputScroll)
        except Exception:
            return None

    def action_dismiss(self) -> None:  # type: ignore[override]
        self.dismiss(None)


class PickerScreen(ModalScreen[str | None]):
    """Arrow-key selector modal. Returns the chosen option id, or None on cancel.

    Intentionally does NOT mix in ``ScrollableModalMixin``: ``OptionList``
    owns up/down for item navigation and handles the wheel itself.
    """

    BINDINGS = [Binding("escape", "cancel", "Cancel", priority=True)]

    # Grow the option list to fit items exactly, capped so an accidental
    # huge list can't swamp the modal box.
    MAX_VISIBLE = 15

    def __init__(self, title: str, options: list[tuple[str, str]]) -> None:
        super().__init__()
        self._title_text = title
        self._options = options

    def compose(self) -> ComposeResult:
        with Container(classes="picker-box"):
            yield Static(
                Text.from_markup(
                    f"[bold]{escape(self._title_text)}[/]  [dim](Enter confirm, Esc cancel)[/]"
                )
            )
            yield OptionList(
                *[Option(label, id=value) for value, label in self._options],
                id="picker-options",
            )

    def on_mount(self) -> None:
        option_list = self.query_one(OptionList)
        option_list.styles.max_height = min(len(self._options), self.MAX_VISIBLE)
        option_list.focus()

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        self.dismiss(event.option.id)

    def action_cancel(self) -> None:
        self.dismiss(None)
