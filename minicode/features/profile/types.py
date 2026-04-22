from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class UserPreferences:
    language: str = ""
    verbosity: str = ""
    response_style: str = ""
    preferred_framework: str = ""
    preferred_test_framework: str = ""
    auto_format: bool = False


@dataclass
class CodingStyle:
    indent_style: str = ""
    indent_size: int = 0
    quote_style: str = ""
    semicolons: bool = False
    trailing_comma: bool = False
    max_line_length: int = 0
    naming_convention: str = ""


@dataclass
class UserProfile:
    preferences: UserPreferences = field(default_factory=UserPreferences)
    coding_style: CodingStyle = field(default_factory=CodingStyle)
    common_patterns: list[str] = field(default_factory=list)
    project_context: str = ""
    custom_instructions: str = ""
    source_path: str = ""
    raw_content: str = ""
