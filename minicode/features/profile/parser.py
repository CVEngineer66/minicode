from __future__ import annotations

import re

from .types import CodingStyle, UserPreferences, UserProfile

_SECTION_RE = re.compile(r"^##\s+(.+)$", re.MULTILINE)
_KV_RE = re.compile(r"^-\s+\*\*(.+?)\*\*:\s*(.+)$")
_LIST_ITEM_RE = re.compile(r"^-\s+(.+)$", re.MULTILINE)
_TRUE = frozenset({"true", "yes", "1"})


def _kv(body: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in body.strip().splitlines():
        m = _KV_RE.match(line.strip())
        if m:
            out[m.group(1).strip().lower().replace(" ", "_")] = m.group(2).strip()
    return out


def _list_items(body: str) -> list[str]:
    items: list[str] = []
    for line in body.strip().splitlines():
        m = _LIST_ITEM_RE.match(line.strip())
        if m:
            items.append(m.group(1).strip())
    return items


def parse_user_md(content: str) -> UserProfile:
    profile = UserProfile(raw_content=content)
    parts = _SECTION_RE.split(content)
    sections: dict[str, str] = {}
    for i in range(1, len(parts) - 1, 2):
        sections[parts[i].strip().lower().replace(" ", "_")] = parts[i + 1]

    if "preferences" in sections:
        kv = _kv(sections["preferences"])
        p = profile.preferences
        p.language = kv.get("language", "")
        p.verbosity = kv.get("verbosity", "")
        p.response_style = kv.get("response_style", "")
        p.preferred_framework = kv.get("preferred_framework", "")
        p.preferred_test_framework = kv.get("preferred_test_framework", "")
        p.auto_format = kv.get("auto_format", "").lower() in _TRUE

    if "coding_style" in sections:
        kv = _kv(sections["coding_style"])
        cs = profile.coding_style
        cs.indent_style = kv.get("indent_style", "")
        try:
            cs.indent_size = int(kv.get("indent_size", "0"))
        except ValueError:
            cs.indent_size = 0
        cs.quote_style = kv.get("quote_style", "")
        cs.semicolons = kv.get("semicolons", "").lower() in _TRUE
        cs.trailing_comma = kv.get("trailing_comma", "").lower() in _TRUE
        try:
            cs.max_line_length = int(kv.get("max_line_length", "0"))
        except ValueError:
            cs.max_line_length = 0
        cs.naming_convention = kv.get("naming_convention", "")

    if "common_patterns" in sections:
        profile.common_patterns = _list_items(sections["common_patterns"])
    if "project_context" in sections:
        profile.project_context = sections["project_context"].strip()
    if "custom_instructions" in sections:
        profile.custom_instructions = sections["custom_instructions"].strip()
    return profile


def serialize_user_md(profile: UserProfile) -> str:
    lines: list[str] = ["# User Profile", ""]
    p = profile.preferences
    pref_any = any(
        [
            p.language,
            p.verbosity,
            p.response_style,
            p.preferred_framework,
            p.preferred_test_framework,
            p.auto_format,
        ]
    )
    if pref_any:
        lines.append("## Preferences")
        if p.language:
            lines.append(f"- **Language**: {p.language}")
        if p.verbosity:
            lines.append(f"- **Verbosity**: {p.verbosity}")
        if p.response_style:
            lines.append(f"- **Response Style**: {p.response_style}")
        if p.preferred_framework:
            lines.append(f"- **Preferred Framework**: {p.preferred_framework}")
        if p.preferred_test_framework:
            lines.append(f"- **Preferred Test Framework**: {p.preferred_test_framework}")
        if p.auto_format:
            lines.append("- **Auto Format**: true")
        lines.append("")

    cs = profile.coding_style
    style_any = any(
        [
            cs.indent_style,
            cs.indent_size,
            cs.quote_style,
            cs.naming_convention,
            cs.semicolons,
            cs.trailing_comma,
            cs.max_line_length,
        ]
    )
    if style_any:
        lines.append("## Coding Style")
        if cs.indent_style:
            lines.append(f"- **Indent Style**: {cs.indent_style}")
        if cs.indent_size:
            lines.append(f"- **Indent Size**: {cs.indent_size}")
        if cs.quote_style:
            lines.append(f"- **Quote Style**: {cs.quote_style}")
        if cs.semicolons:
            lines.append("- **Semicolons**: true")
        if cs.trailing_comma:
            lines.append("- **Trailing Comma**: true")
        if cs.max_line_length:
            lines.append(f"- **Max Line Length**: {cs.max_line_length}")
        if cs.naming_convention:
            lines.append(f"- **Naming Convention**: {cs.naming_convention}")
        lines.append("")

    if profile.common_patterns:
        lines.append("## Common Patterns")
        for pattern in profile.common_patterns:
            lines.append(f"- {pattern}")
        lines.append("")
    if profile.project_context:
        lines.append("## Project Context")
        lines.append(profile.project_context)
        lines.append("")
    if profile.custom_instructions:
        lines.append("## Custom Instructions")
        lines.append(profile.custom_instructions)
        lines.append("")
    return "\n".join(lines)
