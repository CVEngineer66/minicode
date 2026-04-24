"""Central constants for the TUI layer.

Widget IDs, event kind strings, and the auto-mode list live here so no
individual widget or handler hardcodes them. Any change to the contract
(an event rename, an id rename) is a one-file edit.
"""

from __future__ import annotations

from types import SimpleNamespace

# ``status-bar`` matches the selector used by
# tests/test_minicode_next_tui.py::test_slash_suggest_does_not_overlap_input_or_status_bar.
IDS = SimpleNamespace(
    transcript="transcript-scroll",
    bottom_panel="bottom-panel",
    status_line="status-bar",
    slash_suggest="slash-suggest",
    prompt_input="prompt-input",
    picker_options="picker-options",
)

AUTO_MODES: tuple[str, ...] = ("default", "bypass")

EVENT_KINDS = SimpleNamespace(
    model_call_start="model_call_start",
    assistant_thinking="assistant_thinking",
    assistant_token="assistant_token",
    assistant_message="assistant_message",
    progress="progress",
    tool_start="tool_start",
    tool_result="tool_result",
    context_compacted="context_compacted",
    session_finalized="session_finalized",
)

ENTRY_KINDS = SimpleNamespace(
    user="user",
    assistant="assistant",
    tool="tool",
    progress="progress",
    system="system",
    thinking="thinking",
)


def entry_class(kind: str) -> str:
    """CSS class applied by ``EntryView`` for a given transcript kind.

    Keeps ``ENTRY_KINDS`` and the ``.entry-*`` selectors in ``app.py`` in
    lockstep — renaming the prefix is a one-file edit.
    """
    return f"entry-{kind}"
