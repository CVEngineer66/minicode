from __future__ import annotations

import math
import time
from typing import Any

from minicode.features.context.token_estimator import estimate_tokens

from .tfidf import compute_corpus_idf, tfidf_score, tokenize
from .working import ContinuityManager, WorkingMemoryTracker


def _score_entry(entry: dict[str, Any], query_tokens: list[str], idf: dict[str, float]) -> float:
    if not query_tokens:
        return 0.0
    text = f"{entry.get('content', '')} {entry.get('entry_key', '')} {' '.join(entry.get('tags', []))}"
    doc_tokens = tokenize(text)
    tfidf = tfidf_score(query_tokens, doc_tokens, idf)
    query_lower = " ".join(query_tokens).lower()
    content_lower = entry.get("content", "").lower()
    substring = 2.0 if query_lower in content_lower else (
        1.0 if any(q in content_lower for q in query_tokens) else 0.0
    )
    tag_bonus = 0.0
    for tag in entry.get("tags", []):
        if query_lower in tag.lower():
            tag_bonus += 1.5
            break
    usage_bonus = math.log1p(entry.get("usage_count", 0)) * 0.3
    age_hours = max(0.0, (time.time() - entry.get("updated_at", time.time())) / 3600)
    recency = 1.0 / (1.0 + age_hours / 24.0) * 0.5
    return tfidf + substring + tag_bonus + usage_bonus + recency


class MemoryService:
    """Long-term layered memory backed by SQLite repository, with TF-IDF ranking.

    Boundaries:
    - Prompt injection is capped by `max_tokens` (default 8000)
    - Turn recall is capped by `recall_limit` (default 5)
    - Working memory is an in-process companion (not persisted)
    """

    RECALL_LIMIT = 5
    PROMPT_MAX_TOKENS = 8000
    SCOPE_PRIORITY = ("local", "project", "user")

    def __init__(self, repository: Any, workspace: str) -> None:
        self.repository = repository
        self.workspace = workspace
        self.working = WorkingMemoryTracker()
        self.continuity = ContinuityManager()

    # ---- writes ----
    def put(
        self,
        scope: str,
        entry_key: str,
        content: str,
        tags: list[str] | None = None,
        workspace: str | None = None,
    ) -> None:
        self.repository.put(scope, workspace or self.workspace, entry_key, content, tags)

    def remember_turn(self, user_query: str, final_text: str) -> None:
        if not user_query.strip() or not final_text.strip():
            return
        entry_key = user_query[:80].replace("\n", " ")
        content = f"User: {user_query}\nAssistant: {final_text}"
        self.put("local", entry_key, content, tags=["turn"])

    # ---- search / recall ----
    def search(self, query: str, limit: int | None = None) -> list[dict[str, Any]]:
        if not query.strip():
            return []
        limit = limit or self.RECALL_LIMIT
        raw = self.repository.search(query, self.workspace, limit=max(limit * 3, limit))
        if not raw:
            return []
        query_tokens = tokenize(query)
        corpus = [tokenize(f"{e.get('content', '')} {e.get('entry_key', '')}") for e in raw]
        idf = compute_corpus_idf(corpus)
        scored = sorted(
            ((_score_entry(e, query_tokens, idf), e) for e in raw),
            key=lambda x: x[0],
            reverse=True,
        )
        return [e for score, e in scored[:limit] if score > 0]

    # ---- prompt injection ----
    def build_prompt_block(self, user_query: str, max_tokens: int | None = None) -> str:
        if not user_query.strip():
            return ""
        budget = max_tokens or self.PROMPT_MAX_TOKENS
        entries = self.search(user_query, limit=self.RECALL_LIMIT * 2)
        if not entries:
            return ""
        # Prioritize by scope order, cap by token budget
        entries.sort(key=lambda e: self._scope_rank(e.get("scope", "user")))
        lines: list[str] = []
        total = 0
        for entry in entries:
            line = f"[{entry['scope']}] {entry.get('entry_key', '')}: {entry.get('content', '')}"
            cost = estimate_tokens(line)
            if total + cost > budget:
                break
            lines.append(line)
            total += cost
        # Include protected working memory (always; it's small by design)
        protected = self.working.protected_content()
        if protected:
            lines.append("## Working memory:")
            for item in protected:
                cost = estimate_tokens(item)
                if total + cost > budget:
                    break
                lines.append(f"- {item}")
                total += cost
        return "\n".join(lines)

    def inject_into_prompt(self, system_prompt: str, user_query: str) -> str:
        block = self.build_prompt_block(user_query)
        if not block:
            return system_prompt
        return (
            f"{system_prompt}\n\n## Project Memory & Context\n\n"
            f"Information accumulated from previous sessions and current task:\n\n{block}"
        )

    # ---- introspection ----
    def list(self, scope: str | None = None) -> list[dict[str, Any]]:
        return self.repository.list(scope=scope, workspace=self.workspace)

    def stats(self) -> dict[str, Any]:
        entries = self.list()
        by_scope: dict[str, int] = {}
        for e in entries:
            by_scope[e.get("scope", "user")] = by_scope.get(e.get("scope", "user"), 0) + 1
        return {
            "total": len(entries),
            "by_scope": by_scope,
            "working": self.working.stats(),
            "continuity_markers": len(self.continuity.recent(100)),
        }

    @classmethod
    def _scope_rank(cls, scope: str) -> int:
        try:
            return cls.SCOPE_PRIORITY.index(scope)
        except ValueError:
            return len(cls.SCOPE_PRIORITY)
