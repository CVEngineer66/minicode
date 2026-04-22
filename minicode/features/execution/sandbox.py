from __future__ import annotations

import fnmatch
from pathlib import Path

SENSITIVE_PATTERNS: tuple[str, ...] = (
    ".env",
    ".env.*",
    "*.pem",
    "*.key",
    "id_rsa",
    "id_rsa.*",
    "id_ed25519",
    "id_ed25519.*",
    "*.pfx",
    "*.p12",
    "credentials",
    "credentials.*",
    ".npmrc",
    ".pypirc",
    ".aws/credentials",
    ".ssh/*",
    "*.secret",
)


class PathEscapeError(RuntimeError):
    """Raised when a target path resolves outside allowed roots."""


def resolve_path_within(target: str | Path, roots: list[Path]) -> Path:
    """Resolve `target` and ensure it is inside at least one of `roots`.

    Raises PathEscapeError otherwise. `roots` are resolved to absolute paths.
    Relative targets are resolved against the first root.
    """
    if not roots:
        raise ValueError("At least one root must be provided")
    resolved_roots = [Path(r).resolve() for r in roots]
    path = Path(target)
    if not path.is_absolute():
        path = resolved_roots[0] / path
    resolved = path.resolve()
    for root in resolved_roots:
        try:
            resolved.relative_to(root)
            return resolved
        except ValueError:
            continue
    raise PathEscapeError(f"Path {resolved} escapes allowed roots: {resolved_roots}")


def is_sensitive_path(path: str | Path) -> bool:
    """Check if the path matches a known sensitive filename pattern."""
    name = Path(path).name.lower()
    full = str(path).replace("\\", "/").lower()
    for pattern in SENSITIVE_PATTERNS:
        p = pattern.lower()
        if "/" in p:
            if fnmatch.fnmatch(full, f"*{p}") or fnmatch.fnmatch(full, f"*/{p}"):
                return True
        elif fnmatch.fnmatch(name, p):
            return True
    return False
