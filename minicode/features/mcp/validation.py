from __future__ import annotations

import os
from pathlib import Path

# Characters that can be misused to chain shell commands in MCP server argv.
DANGEROUS_SHELL_CHARS = set("|&;`$(){}<>\n\r")

# MCP JSON-RPC payload cap (defence against malicious servers).
MAX_MCP_PAYLOAD_BYTES = 50 * 1024 * 1024  # 50 MB

# Commands we are willing to spawn as MCP servers.
ALLOWED_COMMANDS: frozenset[str] = frozenset(
    {
        "node",
        "npm",
        "npx",
        "python",
        "python3",
        "pip",
        "pip3",
        "uv",
        "deno",
        "bun",
        "cargo",
        "go",
        "java",
        "javac",
        "ruby",
        "gem",
        "dotnet",
        "curl",
        "wget",
    }
)

DANGEROUS_SHELLS: frozenset[str] = frozenset(
    {"cmd.exe", "command.com", "powershell.exe", "pwsh.exe", "bash", "sh"}
)


class McpValidationError(RuntimeError):
    pass


def validate_command(command: str) -> None:
    """Reject path-traversal, blocked shells, and uncontrolled absolute paths."""
    normalized = Path(command).resolve().as_posix()
    if ".." in normalized or "~" in normalized:
        raise McpValidationError("MCP command contains path traversal characters")

    base = Path(command).name.lower()
    if base.endswith(".exe"):
        base = base[:-4]

    if any(normalized.lower().endswith(s) for s in DANGEROUS_SHELLS):
        raise McpValidationError(f"MCP command '{command}' is a blocked shell")

    if Path(command).is_absolute():
        home = Path.home().as_posix()
        allowed_dirs = [
            "/usr/bin",
            "/usr/local/bin",
            "/usr/local/sbin",
            "/usr/sbin",
            "/opt",
            "/opt/homebrew/bin",
            "/opt/homebrew/sbin",
            "/snap/bin",
            "/home/linuxbrew/.linuxbrew/bin",
            f"{home}/.local/bin",
            f"{home}/.cargo/bin",
            f"{home}/.nvm",
        ]
        if os.name == "nt":
            allowed_dirs.extend(
                ["C:/Program Files", "C:/Program Files (x86)", "C:/Windows/System32"]
            )
        ok = any(normalized.lower().startswith(d.lower()) for d in allowed_dirs)
        if not ok and base not in ALLOWED_COMMANDS:
            raise McpValidationError(
                f"MCP command '{command}' is not whitelisted"
            )
    else:
        if base not in ALLOWED_COMMANDS:
            raise McpValidationError(
                f"MCP command '{command}' is not in the allowed list"
            )


def validate_args(args: list[str]) -> None:
    for arg in args:
        if not isinstance(arg, str):
            raise McpValidationError(f"MCP arg is not a string: {arg!r}")
        if any(ch in DANGEROUS_SHELL_CHARS for ch in arg):
            raise McpValidationError(
                f"MCP arg contains dangerous shell characters: {arg!r}"
            )


def sanitize_tool_segment(value: str) -> str:
    cleaned = "".join(
        c.lower() if c.isalnum() or c in {"_", "-"} else "_" for c in value
    )
    return cleaned.strip("_") or "tool"
