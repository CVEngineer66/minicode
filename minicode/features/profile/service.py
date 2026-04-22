from __future__ import annotations

from typing import Any

from .repository import ProfileRepository
from .types import CodingStyle, UserPreferences, UserProfile

_TRUE = frozenset({"true", "yes", "1"})

_PREF_STR_KEYS = {
    "preferences.language": "language",
    "preferences.verbosity": "verbosity",
    "preferences.response_style": "response_style",
    "preferences.preferred_framework": "preferred_framework",
    "preferences.preferred_test_framework": "preferred_test_framework",
}
_STYLE_STR_KEYS = {
    "coding_style.indent_style": "indent_style",
    "coding_style.quote_style": "quote_style",
    "coding_style.naming_convention": "naming_convention",
}
_STYLE_INT_KEYS = {
    "coding_style.indent_size": "indent_size",
    "coding_style.max_line_length": "max_line_length",
}
_STYLE_BOOL_KEYS = {
    "coding_style.semicolons": "semicolons",
    "coding_style.trailing_comma": "trailing_comma",
}


class ProfileService:
    """User profile loader with merge semantics (project overrides global).

    Boundaries:
    - Writes are scoped explicitly to global or project
    - to_prompt_section caps free-text at 300 chars to avoid prompt bloat
    """

    def __init__(self, repository: ProfileRepository) -> None:
        self.repository = repository

    def load_merged(self) -> UserProfile:
        g = self.repository.load_global()
        p = self.repository.load_project()
        if g is None and p is None:
            return UserProfile()
        if g is None:
            return p  # type: ignore[return-value]
        if p is None:
            return g
        return self._merge(g, p)

    def load_global(self) -> UserProfile | None:
        return self.repository.load_global()

    def load_project(self) -> UserProfile | None:
        return self.repository.load_project()

    def to_prompt_section(self, profile: UserProfile) -> str:
        parts: list[str] = ["## User Profile", ""]
        p = profile.preferences
        prefs: list[str] = []
        if p.language:
            prefs.append(f"Language: {p.language}")
        if p.verbosity:
            prefs.append(f"Verbosity: {p.verbosity}")
        if p.response_style:
            prefs.append(f"Response style: {p.response_style}")
        if p.preferred_framework:
            prefs.append(f"Preferred framework: {p.preferred_framework}")
        if p.preferred_test_framework:
            prefs.append(f"Preferred test framework: {p.preferred_test_framework}")
        if p.auto_format:
            prefs.append("Auto-format on edit: yes")
        if prefs:
            parts.append("Preferences: " + ", ".join(prefs))

        cs = profile.coding_style
        style: list[str] = []
        if cs.indent_style:
            suffix = f" ({cs.indent_size})" if cs.indent_size else ""
            style.append(f"indent: {cs.indent_style}{suffix}")
        if cs.quote_style:
            style.append(f"quotes: {cs.quote_style}")
        if cs.naming_convention:
            style.append(f"naming: {cs.naming_convention}")
        if cs.max_line_length:
            style.append(f"max line: {cs.max_line_length}")
        if style:
            parts.append("Coding style: " + ", ".join(style))

        if profile.common_patterns:
            parts.append("Common patterns: " + "; ".join(profile.common_patterns[:5]))
        if profile.project_context:
            parts.append(f"Project context: {profile.project_context[:200]}")
        if profile.custom_instructions:
            parts.append(f"Custom instructions: {profile.custom_instructions[:300]}")

        if len(parts) <= 2:
            return ""
        return "\n".join(parts)

    def inject_into_prompt(self, system_prompt: str, profile: UserProfile | None = None) -> str:
        profile = profile or self.load_merged()
        block = self.to_prompt_section(profile)
        if not block:
            return system_prompt
        return f"{system_prompt}\n\n{block}"

    def set(self, key: str, value: str, scope: str = "global") -> bool:
        """Set a profile key. Returns False if key is unknown."""
        if scope == "project":
            profile = self.repository.load_project() or UserProfile()
        else:
            profile = self.repository.load_global() or UserProfile()
        if not self._apply_setting(profile, key, value):
            return False
        if scope == "project":
            self.repository.save_project(profile)
        else:
            self.repository.save_global(profile)
        return True

    def search(self, query: str, profile: UserProfile | None = None) -> list[str]:
        profile = profile or self.load_merged()
        q = query.lower()
        matches: list[str] = []
        for attr in (
            "language",
            "verbosity",
            "response_style",
            "preferred_framework",
            "preferred_test_framework",
        ):
            val = getattr(profile.preferences, attr, "")
            if val and q in val.lower():
                matches.append(f"preference.{attr} = {val}")
        for attr in ("indent_style", "quote_style", "naming_convention"):
            val = getattr(profile.coding_style, attr, "")
            if val and q in val.lower():
                matches.append(f"coding_style.{attr} = {val}")
        for pattern in profile.common_patterns:
            if q in pattern.lower():
                matches.append(f"pattern: {pattern}")
        for text, label in (
            (profile.project_context, "project_context"),
            (profile.custom_instructions, "custom_instructions"),
        ):
            if text and q in text.lower():
                matches.append(f"{label}: (matched)")
        return matches

    def delete(self, scope: str) -> bool:
        if scope == "project":
            return self.repository.delete_project()
        return self.repository.delete_global()

    # --- internals ---
    @staticmethod
    def _apply_setting(profile: UserProfile, key: str, value: str) -> bool:
        if key in _PREF_STR_KEYS:
            setattr(profile.preferences, _PREF_STR_KEYS[key], value)
            return True
        if key == "preferences.auto_format":
            profile.preferences.auto_format = value.lower() in _TRUE
            return True
        if key in _STYLE_STR_KEYS:
            setattr(profile.coding_style, _STYLE_STR_KEYS[key], value)
            return True
        if key in _STYLE_INT_KEYS:
            try:
                setattr(profile.coding_style, _STYLE_INT_KEYS[key], int(value))
                return True
            except ValueError:
                return False
        if key in _STYLE_BOOL_KEYS:
            setattr(profile.coding_style, _STYLE_BOOL_KEYS[key], value.lower() in _TRUE)
            return True
        if key == "project_context":
            profile.project_context = value
            return True
        if key == "custom_instructions":
            profile.custom_instructions = value
            return True
        return False

    @staticmethod
    def _merge(g: UserProfile, p: UserProfile) -> UserProfile:
        merged = UserProfile()
        for attr in (
            "language",
            "verbosity",
            "response_style",
            "preferred_framework",
            "preferred_test_framework",
        ):
            setattr(
                merged.preferences,
                attr,
                getattr(p.preferences, attr, "") or getattr(g.preferences, attr, ""),
            )
        merged.preferences.auto_format = p.preferences.auto_format or g.preferences.auto_format

        for attr in ("indent_style", "quote_style", "naming_convention"):
            setattr(
                merged.coding_style,
                attr,
                getattr(p.coding_style, attr, "") or getattr(g.coding_style, attr, ""),
            )
        for attr in ("indent_size", "max_line_length"):
            setattr(
                merged.coding_style,
                attr,
                getattr(p.coding_style, attr, 0) or getattr(g.coding_style, attr, 0),
            )
        merged.coding_style.semicolons = p.coding_style.semicolons or g.coding_style.semicolons
        merged.coding_style.trailing_comma = (
            p.coding_style.trailing_comma or g.coding_style.trailing_comma
        )

        seen: set[str] = set()
        for pattern in g.common_patterns + p.common_patterns:
            if pattern not in seen:
                merged.common_patterns.append(pattern)
                seen.add(pattern)
        merged.project_context = p.project_context or g.project_context
        merged.custom_instructions = p.custom_instructions or g.custom_instructions
        merged.source_path = f"{g.source_path} + {p.source_path}"
        return merged
