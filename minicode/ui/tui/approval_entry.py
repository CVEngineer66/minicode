"""Inline interrupt card widget.

Replaces the old full-screen ``ApprovalScreen``. Sits in the transcript
as a focusable widget that owns its own arrow-key / Enter / Ctrl+O
bindings. Emits ``ApprovalDecided`` message when the user picks a
choice; the app then removes the card, restores Input focus, and
resumes the turn with the selected payload.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Any

from rich.console import Group
from rich.markup import escape
from rich.text import Text
from textual.binding import Binding
from textual.message import Message
from textual.widgets import Static

# Arg keys whose values are treated as "execution body" and folded to
# _BODY_PREVIEW_LINES. Scalars (path, count, append, ...) are rendered
# one-per-line above the body section.
_BODY_FIELDS: tuple[str, ...] = (
    "content",
    "updated_content",
    "replace",
    "search",
    "command",
    "args",
    "script",
    "diff",
    "patch",
)
_BODY_PREVIEW_LINES = 5


@dataclass(frozen=True, slots=True)
class _Choice:
    key: str
    label: str
    payload: dict[str, Any]


class ApprovalEntry(Static):
    """Focusable transcript card that collects an interrupt decision."""

    can_focus = True

    BINDINGS = [
        Binding("up", "move_up", show=False, priority=True),
        Binding("down", "move_down", show=False, priority=True),
        Binding("enter", "confirm", show=False, priority=True),
        Binding("escape", "deny", show=False, priority=True),
        Binding("ctrl+o", "toggle_expand", show=False, priority=True),
    ]

    class ApprovalDecided(Message):
        def __init__(self, decision: dict[str, Any]) -> None:
            super().__init__()
            self.decision = decision

    def __init__(self, payload: dict[str, Any], **kwargs: Any) -> None:
        super().__init__("", markup=False, **kwargs)
        self._prompt_kind = str(payload.get("prompt_kind") or "approval")
        self._summary = str(payload.get("summary", "Approval required"))
        self._details = [str(d) for d in payload.get("details", [])]
        self._choices = [
            _Choice(
                key=str(c.get("key", "")),
                label=str(c.get("label", "")),
                payload=dict(c.get("payload") or {"decision": str(c.get("decision", "allow_once"))}),
            )
            for c in payload.get("choices", [])
        ]
        self._cancel_payload = dict(
            payload.get("cancel_payload")
            or ({"decision": "deny"} if self._prompt_kind == "approval" else {"choice_cancelled": True})
        )
        self._highlight = 0
        self._expanded = False
        self.add_class("entry-approval")

    def on_mount(self) -> None:
        self.update(self._build_renderable())

    # ---- actions ----------------------------------------------------

    def action_move_up(self) -> None:
        if not self._choices:
            return
        self._highlight = (self._highlight - 1) % len(self._choices)
        self.update(self._build_renderable())

    def action_move_down(self) -> None:
        if not self._choices:
            return
        self._highlight = (self._highlight + 1) % len(self._choices)
        self.update(self._build_renderable())

    def action_confirm(self) -> None:
        if not self._choices:
            return
        choice = self._choices[self._highlight]
        self.post_message(self.ApprovalDecided(dict(choice.payload)))

    def action_deny(self) -> None:
        self.post_message(self.ApprovalDecided(dict(self._cancel_payload)))

    def action_toggle_expand(self) -> None:
        self._expanded = not self._expanded
        self.update(self._build_renderable())

    # ---- rendering --------------------------------------------------

    def _build_renderable(self) -> Any:
        if self._prompt_kind == "choice":
            header_prefix = "[bold cyan]? User choice required[/]"
            hint_text = "[dim](up/down select - Enter confirm - Esc cancel - Ctrl+O expand)[/]"
        else:
            header_prefix = "[bold red]! Approval required[/]"
            hint_text = "[dim](up/down select - Enter confirm - Esc deny - Ctrl+O expand)[/]"
        header = Text.from_markup(f"{header_prefix}\n[bold]{escape(self._summary)}[/]")
        body = self._render_details()
        choices = self._render_choices()
        hint = Text.from_markup(hint_text)
        return Group(header, body, choices, hint)

    def _render_details(self) -> Text:
        lines = Text()
        args_dict: dict[str, Any] | None = None
        for line in self._details:
            if line.startswith("Arguments: "):
                args_dict = _safe_literal(line[len("Arguments: "):])
                continue
            lines.append_text(Text.from_markup(f"  [yellow]{escape(line)}[/]\n"))

        if args_dict is None:
            return lines

        short_fields, body_fields = _split_args(args_dict)
        for k, v in short_fields.items():
            lines.append_text(Text.from_markup(f"  [dim]{escape(k)}:[/] {escape(str(v))}\n"))
        for k, v in body_fields.items():
            lines.append_text(self._render_body_field(k, v))
        return lines

    def _render_body_field(self, key: str, value: Any) -> Text:
        text = value if isinstance(value, str) else repr(value)
        body_lines = text.splitlines() or [""]
        count = len(body_lines)
        if self._expanded or count <= _BODY_PREVIEW_LINES:
            preview = text
            tail = ""
        else:
            preview = "\n".join(body_lines[:_BODY_PREVIEW_LINES])
            remaining = count - _BODY_PREVIEW_LINES
            tail = f"\n  [dim]... ({remaining} more lines - ctrl+o to expand)[/]"
        return Text.from_markup(f"  [cyan]{escape(key)}:[/]\n" f"  [dim]{escape(preview)}[/]{tail}\n")

    def _render_choices(self) -> Text:
        if not self._choices:
            return Text()
        lines = Text()
        for i, choice in enumerate(self._choices):
            prefix = "> " if i == self._highlight else "  "
            line = Text(prefix + choice.label)
            line.stylize("black on bright_cyan" if i == self._highlight else "white")
            lines.append_text(line)
            lines.append("\n")
        return lines


def _safe_literal(raw: str) -> dict[str, Any] | None:
    try:
        parsed = ast.literal_eval(raw)
    except (SyntaxError, ValueError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _split_args(
    args: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Separate short scalar args from long 'body' args."""
    short: dict[str, Any] = {}
    body: dict[str, Any] = {}
    for k, v in args.items():
        if k in _BODY_FIELDS:
            body[k] = v
            continue
        if isinstance(v, str) and ("\n" in v or len(v) > 80):
            body[k] = v
            continue
        short[k] = v
    return short, body
