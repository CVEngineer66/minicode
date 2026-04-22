from __future__ import annotations

from .commands import find_matching_slash_commands

def parse_input(text: str) -> tuple[str, list[str]]:
    stripped = text.strip()
    if not stripped.startswith("/"):
        return "prompt", [text]
    parts = stripped[1:].split()
    return (parts[0] if parts else "", parts[1:])


def get_visible_commands(text: str):
    return find_matching_slash_commands(text)
