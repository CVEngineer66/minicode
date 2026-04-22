from __future__ import annotations

import fnmatch
import os
from typing import Iterable


def _win() -> bool:
    return os.name == "nt"


def normalize_path(path: str) -> str:
    """Resolve symlinks + case-normalize on Windows for stable matching."""
    from pathlib import Path

    try:
        resolved = str(Path(path).resolve())
    except OSError:
        resolved = path
    return resolved.lower() if _win() else resolved


def is_within(root: str, target: str) -> bool:
    root_n = normalize_path(root)
    target_n = normalize_path(target)
    return target_n == root_n or target_n.startswith(root_n + os.sep)


def match_any_path(target: str, patterns: Iterable[str]) -> bool:
    """True if target is inside any prefix in patterns OR matches a glob pattern."""
    target_n = normalize_path(target)
    for pattern in patterns:
        if not pattern:
            continue
        if "*" in pattern or "?" in pattern or "[" in pattern:
            candidate = target_n.lower() if _win() else target_n
            needle = pattern.lower() if _win() else pattern
            if fnmatch.fnmatch(candidate, needle):
                return True
        else:
            if is_within(pattern, target):
                return True
    return False


def match_command(command: str, args: list[str], patterns: Iterable[str]) -> bool:
    """True if "command args..." matches any glob pattern."""
    signature = " ".join([command, *args]).strip()
    for pattern in patterns:
        if not pattern:
            continue
        if fnmatch.fnmatch(signature, pattern):
            return True
        if fnmatch.fnmatch(command, pattern):
            return True
    return False


def classify_dangerous_command(command: str, args: list[str]) -> str | None:
    """Return a human-readable reason if this invocation is dangerous, else None."""
    norm = [a.strip() for a in args if a.strip()]
    sig = " ".join([command, *norm]).strip()

    if command == "git":
        if "reset" in norm and "--hard" in norm:
            return f"git reset --hard can discard local changes ({sig})"
        if "clean" in norm:
            return f"git clean can delete untracked files ({sig})"
        if "checkout" in norm and "--" in norm:
            return f"git checkout -- can overwrite working tree files ({sig})"
        if "push" in norm and any(a in {"--force", "-f"} for a in norm):
            return f"git push --force rewrites remote history ({sig})"

    if command == "npm" and "publish" in norm:
        return f"npm publish affects a registry outside this machine ({sig})"

    if command == "rm":
        combined = "".join(a for a in norm if a.startswith("-")).lower()
        if "r" in combined and "f" in combined:
            if any(a in {"/", "/*"} for a in norm) or "--no-preserve-root" in norm:
                return f"rm -rf can cause catastrophic data loss ({sig})"
            return f"rm -rf can cause catastrophic data loss ({sig})"

    if command in {"dd", "mkfs", "mkfs.ext4", "mkfs.vfat", "fdisk", "format"}:
        return f"{command} can modify or destroy disk partitions ({sig})"

    if command == "chmod" and any(a.endswith("777") for a in norm):
        return f"chmod 777 opens permissions to all users ({sig})"

    if command in {"node", "python", "python3", "bun", "bash", "sh", "zsh", "powershell", "pwsh"}:
        return f"{command} can execute arbitrary local code ({sig})"

    if command == "diskutil":
        return f"diskutil can erase or partition disks ({sig})"
    if command == "launchctl" and any(a in {"unload", "bootout", "disable"} for a in norm):
        return f"launchctl can disable system services ({sig})"
    if command == "dscl":
        return f"dscl can modify directory services and user accounts ({sig})"

    return None
