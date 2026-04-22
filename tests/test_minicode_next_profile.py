from __future__ import annotations

from pathlib import Path

import pytest

from minicode.features.profile import (
    ProfileRepository,
    ProfileService,
    UserProfile,
    parse_user_md,
    serialize_user_md,
)

SAMPLE = """# User Profile

## Preferences
- **Language**: zh-CN
- **Verbosity**: concise
- **Auto Format**: true

## Coding Style
- **Indent Style**: spaces
- **Indent Size**: 4
- **Quote Style**: double
- **Max Line Length**: 100

## Common Patterns
- Prefer dataclasses over dicts
- Use pytest for tests

## Project Context
A terminal coding assistant.

## Custom Instructions
Always write tests alongside code.
"""


def test_parse_sections():
    p = parse_user_md(SAMPLE)
    assert p.preferences.language == "zh-CN"
    assert p.preferences.verbosity == "concise"
    assert p.preferences.auto_format is True
    assert p.coding_style.indent_style == "spaces"
    assert p.coding_style.indent_size == 4
    assert p.coding_style.max_line_length == 100
    assert len(p.common_patterns) == 2
    assert "terminal" in p.project_context
    assert "tests" in p.custom_instructions


def test_roundtrip_serialize(tmp_path: Path):
    p = parse_user_md(SAMPLE)
    text = serialize_user_md(p)
    p2 = parse_user_md(text)
    assert p2.preferences.language == p.preferences.language
    assert p2.coding_style.indent_size == p.coding_style.indent_size
    assert p2.common_patterns == p.common_patterns


def test_service_merge_project_overrides_global(tmp_path: Path):
    g = tmp_path / "global.md"
    proj = tmp_path / "proj.md"
    g.write_text("# UP\n\n## Preferences\n- **Language**: en-US\n- **Verbosity**: detailed\n", encoding="utf-8")
    proj.write_text("# UP\n\n## Preferences\n- **Verbosity**: concise\n", encoding="utf-8")
    svc = ProfileService(ProfileRepository(g, proj))
    merged = svc.load_merged()
    assert merged.preferences.language == "en-US"
    assert merged.preferences.verbosity == "concise"


def test_service_set_and_search(tmp_path: Path):
    g = tmp_path / "g.md"
    p = tmp_path / "p.md"
    svc = ProfileService(ProfileRepository(g, p))
    assert svc.set("preferences.language", "zh-CN", scope="global") is True
    assert svc.set("coding_style.indent_size", "4", scope="project") is True
    assert svc.set("does.not.exist", "x") is False
    merged = svc.load_merged()
    assert merged.preferences.language == "zh-CN"
    assert merged.coding_style.indent_size == 4
    hits = svc.search("zh", merged)
    assert any("language" in h for h in hits)


def test_prompt_section_empty_when_no_content():
    svc = ProfileService(ProfileRepository(Path("x"), Path("y")))
    assert svc.to_prompt_section(UserProfile()) == ""


def test_prompt_injection():
    svc = ProfileService(ProfileRepository(Path("x"), Path("y")))
    prof = parse_user_md(SAMPLE)
    out = svc.inject_into_prompt("SYS", prof)
    assert "SYS" in out
    assert "User Profile" in out
    assert "zh-CN" in out
