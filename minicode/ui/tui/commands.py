from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SlashCommand:
    usage: str
    description: str


SLASH_COMMANDS: list[SlashCommand] = [
    SlashCommand("/help", "show available commands"),
    SlashCommand("/sessions", "list saved sessions"),
    SlashCommand("/resume latest", "resume the latest session"),
    SlashCommand("/resume <thread_id>", "resume a specific session"),
    SlashCommand("/branch", "branch the current session into a new thread"),
    SlashCommand("/compact", "force context compaction on the current session"),
    SlashCommand("/mode default", "set mode to default"),
    SlashCommand("/mode auto", "set mode to auto"),
    SlashCommand("/mode bypass", "set mode to bypass"),
    SlashCommand("/mode plan", "set mode to plan"),
    SlashCommand("/model", "show or change the active model"),
    SlashCommand("/memory", "show memory stats or search entries"),
    SlashCommand("/memory search <query>", "search long-term memory"),
    SlashCommand("/profile", "show merged user profile"),
    SlashCommand("/cost", "show token/cost summary for the session"),
    SlashCommand("/context", "show context window usage"),
    SlashCommand("/tasks", "show tracked tasks"),
    SlashCommand("/agents", "show collaboration agents"),
    SlashCommand("/skills", "list installed skills"),
    SlashCommand("/mcp", "list MCP servers"),
    SlashCommand("/hooks", "show hook statistics"),
    SlashCommand("/permissions", "show permission summary"),
    SlashCommand("/clear", "clear the visible transcript"),
    SlashCommand("/quit", "exit the TUI"),
]


def find_matching_slash_commands(input_text: str) -> list[SlashCommand]:
    stripped = input_text.strip()
    if not stripped.startswith("/"):
        return []
    if stripped == "/":
        return SLASH_COMMANDS
    wanted = stripped.lower().split()
    matches: list[SlashCommand] = []
    for command in SLASH_COMMANDS:
        usage_tokens = command.usage.lower().split()
        if len(wanted) > len(usage_tokens):
            continue
        if all(_slash_usage_token_matches(input_token, usage_token) for input_token, usage_token in zip(wanted, usage_tokens)):
            matches.append(command)
    return matches


def _slash_usage_token_matches(input_token: str, usage_token: str) -> bool:
    if usage_token.startswith("<") and usage_token.endswith(">"):
        return bool(input_token)
    return usage_token.startswith(input_token)
