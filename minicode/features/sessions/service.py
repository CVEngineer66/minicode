from __future__ import annotations

import time
import uuid
from datetime import datetime
from typing import Any

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.base import Checkpoint

from minicode.core.types import SessionMeta

from .repository import SessionRepository, checkpoint_from_messages


def format_session_time(updated_at: float, *, now: float | None = None) -> str:
    """Render a session's ``updated_at`` as ``YYYY-MM-DD HH:MM (<relative>)``.

    Relative fragment uses English coarse granularity (``just now`` /
    ``N minutes ago`` / ``N hours ago`` / ``N days ago``); older than 30 days
    falls back to the absolute-only form.
    """
    if updated_at <= 0:
        return "-"
    reference = time.time() if now is None else now
    delta = max(0.0, reference - updated_at)
    absolute = datetime.fromtimestamp(updated_at).strftime("%Y-%m-%d %H:%M")
    if delta < 60:
        relative = "just now"
    elif delta < 3600:
        minutes = int(delta // 60)
        relative = f"{minutes} minute{'s' if minutes != 1 else ''} ago"
    elif delta < 86400:
        hours = int(delta // 3600)
        relative = f"{hours} hour{'s' if hours != 1 else ''} ago"
    elif delta < 86400 * 30:
        days = int(delta // 86400)
        relative = f"{days} day{'s' if days != 1 else ''} ago"
    else:
        return absolute
    return f"{absolute} ({relative})"


def format_session_preview(title: str, *, max_chars: int = 8) -> str:
    """Render a compact single-line preview for session list displays."""
    text = " ".join((title or "").split())
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "..."


class SessionService:
    """Session metadata + message checkpoint access with branch/compact/archive.

    Boundaries (lifecycle tier):
    - `AUTOCOMPACT_MESSAGE_THRESHOLD` — when exceeded, compact() can be called
      by the runner to summarize and prune.
    - `branch()` creates a fork so long conversations can split off safely.
    """

    AUTOCOMPACT_MESSAGE_THRESHOLD = 100

    def __init__(self, repository: SessionRepository, workspace: str, model: str) -> None:
        self.repository = repository
        self.workspace = workspace
        self.model = model

    # --- basic CRUD ---
    def ensure_thread(self, thread_id: str | None, title: str = "") -> str:
        stripped_title = (title or "").strip()
        if thread_id:
            existing = self.repository.get(thread_id)
            if existing:
                if existing.workspace != self.workspace:
                    # Cross-workspace write would silently pollute the other
                    # workspace's session. Refuse here so run_turn surfaces
                    # a clear error instead of mutating someone else's row.
                    raise ValueError(
                        f"Session {thread_id[:8]} belongs to workspace "
                        f"{existing.workspace!r}, not {self.workspace!r}."
                    )
                effective_title = stripped_title or existing.title
                self.repository.upsert(
                    SessionMeta(
                        thread_id=existing.thread_id,
                        workspace=existing.workspace,
                        created_at=existing.created_at,
                        updated_at=time.time(),
                        model=self.model,
                        title=effective_title,
                    )
                )
                return thread_id
        resolved = thread_id or uuid.uuid4().hex[:12]
        if not stripped_title:
            # No user content yet — skip the session_meta row so it doesn't
            # show up in `sessions list`. A subsequent ensure_thread() call
            # carrying a real title will create the row.
            return resolved
        now = time.time()
        self.repository.upsert(
            SessionMeta(
                thread_id=resolved,
                workspace=self.workspace,
                created_at=now,
                updated_at=now,
                model=self.model,
                title=stripped_title,
            )
        )
        return resolved

    def upsert_session(self, meta: SessionMeta) -> None:
        self.repository.upsert(meta)

    def list_sessions(self, workspace: str | None = None) -> list[SessionMeta]:
        return self.repository.list(workspace)

    def get_session(self, thread_id: str) -> SessionMeta | None:
        return self.repository.get(thread_id)

    def get_latest_session(self, workspace: str | None = None) -> SessionMeta | None:
        return self.repository.latest(workspace)

    def resolve_thread_id(
        self, target: str, *, workspace: str | None = None
    ) -> tuple[str | None, str | None]:
        """Resolve a user-entered id/prefix against a workspace's sessions.

        Returns ``(thread_id, None)`` on success or ``(None, error)`` when the
        input is empty with no sessions, ambiguous, or unknown. ``"latest"`` /
        empty string resolves to the most recent session. Callers should pass
        ``workspace`` to scope the search; omitting it searches globally
        (used only by legacy paths).
        """
        sessions = self.repository.list(workspace)
        if not sessions:
            return None, "No sessions in this workspace."
        normalized = (target or "").strip()
        if not normalized or normalized == "latest":
            return sessions[0].thread_id, None
        for session in sessions:
            if session.thread_id == normalized:
                return session.thread_id, None
        matches = [s.thread_id for s in sessions if s.thread_id.startswith(normalized)]
        if len(matches) == 1:
            return matches[0], None
        if len(matches) > 1:
            preview = ", ".join(tid[:12] for tid in matches[:5])
            suffix = " ..." if len(matches) > 5 else ""
            return None, f"Ambiguous session prefix: {normalized} ({preview}{suffix})"
        return None, f"Unknown session: {normalized}"

    def load_messages(self, thread_id: str) -> list[BaseMessage]:
        return self.repository.load_messages(thread_id)

    # --- message tagging ---
    def message_count(self, thread_id: str) -> int:
        return len(self.load_messages(thread_id))

    def needs_compaction(self, thread_id: str) -> bool:
        return self.message_count(thread_id) >= self.AUTOCOMPACT_MESSAGE_THRESHOLD

    def summary(self, thread_id: str, max_chars: int = 400) -> str:
        """Terse human-readable summary for session listings."""
        messages = self.load_messages(thread_id)
        parts: list[str] = []
        first_user = next(
            (m for m in messages if isinstance(m, HumanMessage)), None
        )
        last_assistant = next(
            (m for m in reversed(messages) if not isinstance(m, (HumanMessage, SystemMessage))),
            None,
        )
        if first_user is not None:
            parts.append(f"Q: {self._text(first_user)[:120]}")
        if last_assistant is not None:
            parts.append(f"A: {self._text(last_assistant)[:120]}")
        text = " | ".join(parts)
        return text[:max_chars]

    # --- branching ---
    def branch(self, source_thread_id: str, title: str = "") -> str:
        """Create a new thread_id whose initial messages mirror the source.

        Uses the checkpointer's put API so both threads share history but evolve
        independently going forward. Returns the new thread id.
        """
        source_messages = self.load_messages(source_thread_id)
        new_thread_id = uuid.uuid4().hex[:12]
        now = time.time()
        source = self.repository.get(source_thread_id)
        self.repository.upsert(
            SessionMeta(
                thread_id=new_thread_id,
                workspace=source.workspace if source else self.workspace,
                created_at=now,
                updated_at=now,
                model=self.model,
                title=title or f"branch of {source_thread_id[:8]}",
            )
        )
        self._put_messages(new_thread_id, source_messages)
        return new_thread_id

    # --- compaction ---
    def compact(
        self,
        thread_id: str,
        context_service: Any,
    ) -> int:
        """Compact stored messages in-place. Returns the number of messages removed.

        `context_service` is the ContextService (or compatible) providing
        `should_compact(messages)` and `compact(messages) -> (msgs, result)`.
        """
        messages = self.load_messages(thread_id)
        if not messages or not context_service.should_compact(messages):
            return 0
        new_messages, result = context_service.compact(messages)
        self._put_messages(thread_id, new_messages)
        self._touch(thread_id)
        return result.removed_count

    # --- archival ---
    def archive(self, thread_id: str) -> bool:
        """Mark a session archived by renaming its title; no physical delete."""
        existing = self.repository.get(thread_id)
        if existing is None:
            return False
        if existing.title.startswith("[archived]"):
            return True
        self.repository.upsert(
            SessionMeta(
                thread_id=existing.thread_id,
                workspace=existing.workspace,
                created_at=existing.created_at,
                updated_at=time.time(),
                model=existing.model,
                title=f"[archived] {existing.title}".strip(),
            )
        )
        return True

    def unarchive(self, thread_id: str) -> bool:
        existing = self.repository.get(thread_id)
        if existing is None or not existing.title.startswith("[archived]"):
            return False
        restored = existing.title[len("[archived]") :].strip()
        self.repository.upsert(
            SessionMeta(
                thread_id=existing.thread_id,
                workspace=existing.workspace,
                created_at=existing.created_at,
                updated_at=time.time(),
                model=existing.model,
                title=restored,
            )
        )
        return True

    # --- internals ---
    def _put_messages(self, thread_id: str, messages: list[BaseMessage]) -> None:
        checkpoint = checkpoint_from_messages(messages)
        config = {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}
        with self.repository.db.checkpointer() as saver:
            saver.put(config, checkpoint, metadata={"source": "session-service"}, new_versions={"messages": 1})

    def _touch(self, thread_id: str) -> None:
        meta = self.repository.get(thread_id)
        if meta is None:
            return
        self.repository.upsert(
            SessionMeta(
                thread_id=meta.thread_id,
                workspace=meta.workspace,
                created_at=meta.created_at,
                updated_at=time.time(),
                model=meta.model,
                title=meta.title,
            )
        )

    @staticmethod
    def _text(message: BaseMessage) -> str:
        content = getattr(message, "content", "")
        if isinstance(content, str):
            return content
        return str(content)
