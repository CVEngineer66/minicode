"""Transcript entry widget.

One ``EntryView`` per conversation row (user / assistant / tool /
progress / thinking / system). The app layer sets state via
``append_chunk`` / ``set_body`` / ``set_tool_result`` / ``set_expanded``
and the widget owns the Rich rendering.
"""

from __future__ import annotations

import ast
from typing import Any

from rich.console import Group
from rich.markdown import Markdown
from rich.markup import escape
from rich.text import Text
from textual.widgets import Static

from .constants import ENTRY_KINDS, entry_class

_STATUS_STYLE: dict[str, tuple[str, str]] = {
    "running": ("yellow", "~"),
    "success": ("green", "v"),
    "error": ("red", "x"),
}

_EXPAND_HINT = "[dim](ctrl+o to expand)[/]"


class EntryView(Static):
    """A single transcript entry."""

    def __init__(
        self,
        kind: str,
        *,
        body: str = "",
        tool_call_id: str = "",
        tool_name: str = "",
        tool_args: str = "",
        tool_status: str = "",
        tool_output: str = "",
        tool_line_count: int = 0,
        tool_content_length: int = 0,
        expanded: bool = False,
        streaming: bool = False,
        **kwargs: Any,
    ) -> None:
        self.kind = kind
        self.body = body
        self.tool_call_id = tool_call_id
        self.tool_name = tool_name
        self.tool_args = tool_args
        self.tool_status = tool_status
        self.tool_output = tool_output
        self.tool_line_count = tool_line_count
        self.tool_content_length = tool_content_length
        self.expanded = expanded
        # Plain Text during streaming (cheap per-token update), Markdown
        # only after finalize — avoids O(n^2) re-parse of growing body.
        self.streaming = streaming
        super().__init__(self._build_renderable(), **kwargs)
        self.add_class(entry_class(kind))

    def append_chunk(self, chunk: str) -> None:
        self.body += chunk
        self.streaming = True
        self.update(self._build_renderable())

    def set_body(self, body: str) -> None:
        self.body = body
        self.streaming = False
        self.update(self._build_renderable())

    def set_tool_result(
        self,
        status: str,
        output: str,
        *,
        line_count: int = 0,
        content_length: int = 0,
    ) -> None:
        self.tool_status = status
        self.tool_output = output
        self.tool_line_count = line_count
        self.tool_content_length = content_length
        self.update(self._build_renderable())

    def set_expanded(self, expanded: bool) -> None:
        if self.expanded == expanded:
            return
        self.expanded = expanded
        self.update(self._build_renderable())

    def toggle_expanded(self) -> None:
        self.set_expanded(not self.expanded)

    def _build_renderable(self) -> Any:
        k = self.kind
        if k == ENTRY_KINDS.user:
            return Text(self.body or "")
        if k == ENTRY_KINDS.assistant:
            body = self.body or " "
            return Text(body) if self.streaming else Markdown(body, code_theme="monokai")
        if k == ENTRY_KINDS.progress:
            return Text.from_markup(
                f"[yellow bold]* progress[/]\n[dim]{escape(self.body)}[/]"
            )
        if k == ENTRY_KINDS.system:
            return Text.from_markup(f"[dim]{escape(self.body)}[/]")
        if k == ENTRY_KINDS.thinking:
            return self._thinking_markup()
        if k == ENTRY_KINDS.tool:
            return self._tool_markup()
        return Text(self.body or " ")

    # --- thinking -----------------------------------------------------

    def _thinking_markup(self) -> Text:
        header = "[dim italic]~ thinking[/]"
        body = self.body or ""
        if self.expanded:
            if body.strip():
                return Group(
                    Text.from_markup(f"{header}  [dim](ctrl+o to collapse)[/]"),
                    Text(body, style="dim italic"),
                )  # type: ignore[return-value]
            return Text.from_markup(f"{header}  [dim](waiting for reasoning...)[/]")
        first = _first_line(body, 80)
        if first:
            return Text.from_markup(f"{header}  [dim]{escape(first)}[/]  {_EXPAND_HINT}")
        return Text.from_markup(f"{header}  {_EXPAND_HINT}")

    # --- tool ---------------------------------------------------------

    def _tool_markup(self) -> Text:
        if self.tool_name == "read_file":
            return self._read_file_markup()
        color, glyph = _STATUS_STYLE.get(self.tool_status, ("white", "*"))
        args = _truncate(self.tool_args or "", 80)
        header = (
            f"[{color} bold]{glyph} {escape(self.tool_name)}[/]"
            f"[dim]({escape(args)})[/] [italic {color}]{escape(self.tool_status or '')}[/]"
        )
        return self._render_with_body(header, self.tool_output or "", preview_chars=120)

    def _read_file_markup(self) -> Text:
        color, glyph = _STATUS_STYLE.get(self.tool_status, ("white", "*"))
        path, start_line, end_line = _parse_read_args(self.tool_args)
        targeted = start_line is not None and end_line is not None
        if targeted:
            span = end_line - start_line + 1
            core = f"Read lines {start_line}-{end_line} ({span} lines)"
        elif self.tool_line_count > 0:
            core = f"Read {self.tool_line_count} lines"
        else:
            core = "Read"
        path_part = f" [dim]{escape(path)}[/]" if path else ""
        status_part = (
            f" [italic {color}]{escape(self.tool_status or '')}[/]"
            if self.tool_status
            else ""
        )
        header = f"[{color} bold]{glyph} {core}[/]{path_part}{status_part}"
        body = self.tool_output or ""
        if not body.strip():
            return Text.from_markup(header)
        if self.expanded:
            if targeted:
                return Text.from_markup(f"{header}\n[dim]{escape(body)}[/]")
            lines = body.splitlines()
            preview = "\n".join(lines[:10])
            remaining = max(self.tool_line_count - 10, len(lines) - 10)
            tail = f"\n[dim]... ({remaining} more lines)[/]" if remaining > 0 else ""
            return Text.from_markup(f"{header}\n[dim]{escape(preview)}[/]{tail}")
        first = _first_line(body, 120)
        if first:
            return Text.from_markup(
                f"{header}\n  [dim]|_ {escape(first)}[/]  {_EXPAND_HINT}"
            )
        return Text.from_markup(f"{header}  {_EXPAND_HINT}")

    def _render_with_body(self, header: str, body: str, *, preview_chars: int) -> Text:
        if self.expanded and body.strip():
            return Text.from_markup(f"{header}\n[dim]{escape(body)}[/]")
        first = _first_line(body, preview_chars)
        if first:
            return Text.from_markup(
                f"{header}\n  [dim]|_ {escape(first)}[/]  {_EXPAND_HINT}"
            )
        return Text.from_markup(header)


def _truncate(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _first_line(body: str, limit: int) -> str:
    if not body:
        return ""
    first = body.splitlines()[0].strip()
    return _truncate(first, limit)


def _find_running_tool_entry(
    entries: Any,
    *,
    tool_call_id: str = "",
    tool_name: str = "",
) -> Any | None:
    for entry in reversed(list(entries)):
        if entry.kind != "tool" or entry.tool_status != "running":
            continue
        if tool_call_id and entry.tool_call_id == tool_call_id:
            return entry
        if not tool_call_id and tool_name and entry.tool_name == tool_name:
            return entry
    return None


def _parse_read_args(raw: str) -> tuple[str, int | None, int | None]:
    """Best-effort parse of the ``read_file`` argument dict stringified via ``str(dict)``.

    Returns ``("", None, None)`` on any failure so the caller can still
    render a minimal header without crashing on malformed input.
    """
    if not raw:
        return "", None, None
    try:
        parsed = ast.literal_eval(raw)
    except (SyntaxError, ValueError):
        return "", None, None
    if not isinstance(parsed, dict):
        return "", None, None
    return (
        str(parsed.get("path", "") or ""),
        _coerce_line(parsed.get("start_line")),
        _coerce_line(parsed.get("end_line")),
    )


def _coerce_line(value: Any) -> int | None:
    # Guard: ``isinstance(True, int) is True``, so a hallucinated
    # ``start_line: true`` would silently become line 1 without this.
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)) and value:
        return int(value)
    return None
