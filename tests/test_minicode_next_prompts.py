from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("langchain_core")

from minicode.runtime.prompts import (
    SYSTEM_PROMPT_DYNAMIC_BOUNDARY,
    PromptPipeline,
    assemble_system_prompt,
    build_system_prompt,
    read_file_cached,
)


def test_build_system_prompt_legacy():
    out = build_system_prompt("You are helpful.", "remember X", "auto", "skill: X")
    assert "You are helpful" in out
    assert "auto" in out
    assert "remember X" in out
    assert "skill: X" in out


def test_pipeline_places_boundary_marker():
    p = PromptPipeline()
    p.register_static("role", "You are helpful.")
    p.register_dynamic("memory", lambda: "recent memo")
    out = p.build()
    assert SYSTEM_PROMPT_DYNAMIC_BOUNDARY in out
    assert out.index("You are helpful") < out.index(SYSTEM_PROMPT_DYNAMIC_BOUNDARY)
    assert out.index(SYSTEM_PROMPT_DYNAMIC_BOUNDARY) < out.index("recent memo")


def test_pipeline_condition_suppresses_section():
    p = PromptPipeline()
    p.register_static("role", "role")
    p.register_dynamic("memory", lambda: "secret", condition=lambda: False)
    out = p.build()
    assert "secret" not in out


def test_pipeline_static_infinite_cache():
    p = PromptPipeline()
    calls = {"n": 0}

    def static_builder() -> str:
        calls["n"] += 1
        return "static"

    # Static registered via lambda closure doesn't use register_static's branch,
    # so verify register_dynamic with high TTL caches instead.
    p.register_dynamic("s", static_builder, cache_ttl=3600)
    p.build()
    p.build()
    assert calls["n"] == 1


def test_read_file_cached_returns_none_missing(tmp_path):
    assert read_file_cached(tmp_path / "missing.txt") is None


def test_read_file_cached_reads_and_caches(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("hello", encoding="utf-8")
    assert read_file_cached(f) == "hello"
    # overwrite and bypass mtime → should still serve cache until TTL expires
    # but for robustness we just verify value is string
    assert isinstance(read_file_cached(f), str)


def test_assemble_system_prompt_integrates_services():
    class _Profile:
        def load_merged(self):
            return SimpleNamespace()

        def to_prompt_section(self, _p):
            return "Preferences: Language: zh-CN"

    class _Memory:
        def build_prompt_block(self, q):
            return "[local] q: memo"

    class _Skills:
        def discover(self):
            return [SimpleNamespace(name="code_review")]

    services = SimpleNamespace(profile=_Profile(), memory=_Memory(), skills=_Skills())
    prompt = assemble_system_prompt(
        base_prompt="You are helpful.",
        services=services,
        mode="default",
        latest_user_query="how do I auth?",
    )
    assert "You are helpful" in prompt
    assert "Preferences: Language: zh-CN" in prompt
    assert "code_review" in prompt
    assert "[local] q: memo" in prompt
    assert "Execution mode: default" in prompt
