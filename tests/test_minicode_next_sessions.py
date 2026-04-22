from __future__ import annotations

import pytest

pytest.importorskip("langchain_core")

from minicode.features.sessions import InputHistoryRepository


def test_history_roundtrip(tmp_path):
    repo = InputHistoryRepository(tmp_path / "history.json", max_entries=3)
    repo.append("one")
    repo.append("two")
    repo.append("three")
    repo.append("four")
    assert repo.load() == ["two", "three", "four"]


def test_history_dedup_consecutive(tmp_path):
    repo = InputHistoryRepository(tmp_path / "history.json")
    repo.append("x")
    repo.append("x")
    assert repo.load() == ["x"]


def test_history_ignores_empty(tmp_path):
    repo = InputHistoryRepository(tmp_path / "history.json")
    repo.append("")
    repo.append("   ")
    assert repo.load() == []


def test_history_returns_empty_on_bad_json(tmp_path):
    path = tmp_path / "history.json"
    path.write_text("not json", encoding="utf-8")
    assert InputHistoryRepository(path).load() == []
