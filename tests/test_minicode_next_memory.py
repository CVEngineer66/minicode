from __future__ import annotations

import pytest

pytest.importorskip("langchain_core")

from minicode.features.memory import (
    ContinuityManager,
    MemoryService,
    WorkingMemoryTracker,
    tokenize,
)
from minicode.features.memory.tfidf import compute_corpus_idf, tfidf_score


class _FakeRepo:
    def __init__(self) -> None:
        self.store: list[dict] = []

    def put(self, scope, workspace, entry_key, content, tags=None) -> None:
        self.store.append(
            {
                "scope": scope,
                "workspace": workspace,
                "entry_key": entry_key,
                "content": content,
                "tags": tags or [],
                "updated_at": 0,
                "created_at": 0,
                "usage_count": 0,
            }
        )

    def list(self, scope=None, workspace=None):
        return [r for r in self.store if (not scope or r["scope"] == scope)]

    def search(self, query, workspace, limit=5):
        q = query.lower()
        hits = [r for r in self.store if q in r["content"].lower() or q in r["entry_key"].lower()]
        return hits[:limit]


def test_tokenize_mixed_languages():
    assert "hello" in tokenize("Hello world")
    assert "世" in tokenize("世界")


def test_tfidf_basic():
    docs = [tokenize("python code review"), tokenize("database schema migration")]
    idf = compute_corpus_idf(docs)
    score = tfidf_score(tokenize("python review"), docs[0], idf)
    assert score > 0


def test_memory_service_put_search_inject():
    svc = MemoryService(_FakeRepo(), workspace="/tmp")
    svc.put("local", "auth-design", "We chose JWT for stateless auth", tags=["auth"])
    svc.put("local", "db-migration", "Migrated users table to new schema")
    results = svc.search("auth JWT")
    assert results
    assert "auth-design" in results[0]["entry_key"]

    prompt = svc.inject_into_prompt("SYS", "tell me about auth")
    assert "Project Memory" in prompt
    assert "JWT" in prompt


def test_memory_service_remember_turn_empty_noop():
    svc = MemoryService(_FakeRepo(), workspace="/tmp")
    svc.remember_turn("", "")
    assert not svc.list()


def test_working_memory_limits():
    wm = WorkingMemoryTracker(max_entries=2, max_tokens=10_000)
    wm.add("a")
    wm.add("b")
    wm.add("c")
    assert len(wm.protected_content()) == 2


def test_continuity_manager_trim():
    cm = ContinuityManager(max_markers=2)
    cm.add("task_start", "a")
    cm.add("task_start", "b")
    cm.add("task_start", "c")
    assert len(cm.recent()) == 2
