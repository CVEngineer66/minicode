from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import BaseMessage, SystemMessage

from .summarizer import compress_tool_pair, summarize_removed
from .token_estimator import estimate_message_tokens, estimate_messages_tokens, message_to_dict

DEFAULT_CONTEXT_WINDOWS: dict[str, int] = {
    # Claude 4.6 / 4.7 — 1M context natively, no beta header required.
    "claude-opus-4-7": 1_000_000,
    "claude-opus-4-6": 1_000_000,
    "claude-sonnet-4-6": 1_000_000,
    # Claude 4.5 — 200K context (1M on Sonnet 4.5 is gated behind a beta
    # header we don't set). Haiku 4.5 ships with a full dated alias too.
    "claude-opus-4-5": 200_000,
    "claude-sonnet-4-5": 200_000,
    "claude-haiku-4-5": 200_000,
    "claude-haiku-4-5-20251001": 200_000,
    # Earlier Claude 4 family.
    "claude-opus-4-1": 200_000,
    "claude-opus-4-0": 200_000,
    "claude-sonnet-4-0": 200_000,
    "claude-sonnet-4-20250514": 200_000,
    "claude-opus-4-20250514": 200_000,
    # Haiku 3 (legacy, retiring soon).
    "claude-haiku-3-20240307": 200_000,
    # Other providers.
    "gpt-4o": 128_000,
    "gpt-4o-mini": 128_000,
    "o1": 200_000,
    "o1-mini": 128_000,
    "o3-mini": 200_000,
    "google/gemini-2.5-pro": 1_000_000,
    "google/gemini-2.5-flash": 1_000_000,
    "deepseek/deepseek-r1": 128_000,
    # Fallback for unknown IDs — 200K is a safe baseline for every modern
    # tool-capable model we're likely to see; older-smaller models (128K
    # gpt-4o etc.) are already keyed above.
    "default": 200_000,
}

AUTOCOMPACT_THRESHOLD = 0.95
MIN_MESSAGES_TO_KEEP = 10
COMPACTION_TARGETS = [0.70, 0.50, 0.30]

_EDIT_TOOLS = frozenset({"edit_file", "write_file", "modify_file", "patch_file", "multi_edit"})
_READ_TOOLS = frozenset({"read_file", "list_files", "grep_files", "file_tree"})


@dataclass
class ContextStats:
    total_tokens: int = 0
    context_window: int = 0
    usage_percentage: float = 0.0
    messages_count: int = 0
    is_near_limit: bool = False
    should_compact: bool = False


@dataclass
class CompactionResult:
    messages: list[dict[str, Any]]
    removed_count: int
    before_tokens: int
    after_tokens: int
    summary: str


def window_for(model: str) -> int:
    return DEFAULT_CONTEXT_WINDOWS.get(model, DEFAULT_CONTEXT_WINDOWS["default"])


@dataclass
class ContextManager:
    """Token-budget tracker and compactor over dict-shaped messages.

    Runner-facing helpers operate on LangChain BaseMessage and convert to the
    dict schema used internally (see token_estimator.message_to_dict).
    """

    model: str = "default"
    context_window: int = 0
    compaction_level: int = 0
    compaction_history: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.context_window:
            self.context_window = window_for(self.model)

    def set_model(self, model: str) -> None:
        self.model = model
        self.context_window = window_for(model)

    def stats(self, messages: list[BaseMessage] | list[dict[str, Any]]) -> ContextStats:
        if not messages:
            return ContextStats(context_window=self.context_window)
        total = estimate_messages_tokens(messages)
        usage = (total / self.context_window * 100) if self.context_window > 0 else 0
        return ContextStats(
            total_tokens=total,
            context_window=self.context_window,
            usage_percentage=usage,
            messages_count=len(messages),
            is_near_limit=usage >= 80,
            should_compact=usage >= AUTOCOMPACT_THRESHOLD * 100,
        )

    def should_compact(self, messages: list[BaseMessage] | list[dict[str, Any]]) -> bool:
        threshold = max(0.60, AUTOCOMPACT_THRESHOLD - self.compaction_level * 0.10)
        stats = self.stats(messages)
        return stats.usage_percentage >= threshold * 100

    def compact(
        self,
        messages: list[BaseMessage] | list[dict[str, Any]],
    ) -> CompactionResult:
        dicts = [m if isinstance(m, dict) else message_to_dict(m) for m in messages]
        before_tokens = estimate_messages_tokens(dicts)
        if not self.should_compact(dicts):
            return CompactionResult(dicts, 0, before_tokens, before_tokens, "")

        target_pct = COMPACTION_TARGETS[min(self.compaction_level, 2)]
        target = int(self.context_window * target_pct)

        system_msgs = [m for m in dicts if m.get("role") == "system"]
        others = [m for m in dicts if m.get("role") != "system"]

        # Phase 1: drop progress
        filtered = [m for m in others if m.get("role") != "assistant_progress"]
        if estimate_messages_tokens(filtered) <= target:
            return self._finalize(system_msgs, others, filtered, before_tokens)

        # Phase 2: truncate large tool results
        filtered = [self._maybe_truncate(m) for m in filtered]
        if estimate_messages_tokens(filtered) <= target:
            return self._finalize(system_msgs, others, filtered, before_tokens)

        # Phase 3: compress tool_call + result pairs
        compressed: list[dict[str, Any]] = []
        i = 0
        while i < len(filtered):
            m = filtered[i]
            if (
                m.get("role") == "assistant_tool_call"
                and i + 1 < len(filtered)
                and filtered[i + 1].get("role") == "tool_result"
            ):
                compressed.append(
                    {"role": "assistant", "content": compress_tool_pair(m, filtered[i + 1])}
                )
                i += 2
            else:
                compressed.append(m)
                i += 1
        if estimate_messages_tokens(compressed) <= target:
            return self._finalize(system_msgs, others, compressed, before_tokens)

        # Phase 4: priority-based removal, protect last 6
        priority = {"user": 0, "assistant": 1, "assistant_tool_call": 2, "tool_result": 3}
        PROTECT = 6
        while (
            estimate_messages_tokens(compressed) > target
            and len(compressed) > MIN_MESSAGES_TO_KEEP
        ):
            end = max(MIN_MESSAGES_TO_KEEP, len(compressed) - PROTECT)
            best_idx, best_prio = None, -1
            for idx in range(end):
                prio = priority.get(compressed[idx].get("role", ""), 1)
                if prio > best_prio:
                    best_prio = prio
                    best_idx = idx
            if best_idx is None:
                break
            del compressed[best_idx]
        return self._finalize(system_msgs, others, compressed, before_tokens)

    @staticmethod
    def _maybe_truncate(m: dict[str, Any]) -> dict[str, Any]:
        if m.get("role") != "tool_result":
            return m
        content = m.get("content", "") or ""
        DEFAULT = 2000
        if len(content) <= DEFAULT:
            return m
        tool_name = m.get("toolName", "")
        if m.get("isError"):
            threshold = 4000
        elif tool_name in _EDIT_TOOLS:
            threshold = 3000
        elif tool_name in _READ_TOOLS:
            threshold = 1500
        else:
            threshold = DEFAULT
        if len(content) <= threshold:
            return m
        lines = content.split("\n")
        head, tail = [], []
        head_chars = 0
        for line in lines:
            if head_chars + len(line) + 1 > threshold * 0.7:
                break
            head.append(line)
            head_chars += len(line) + 1
        tail_chars = 0
        for line in reversed(lines):
            if tail_chars + len(line) + 1 > threshold * 0.3:
                break
            tail.insert(0, line)
            tail_chars += len(line) + 1
        omitted = len(lines) - len(head) - len(tail)
        truncated = "\n".join(head)
        if omitted > 0:
            truncated += f"\n... [{omitted} lines truncated for compaction] ...\n"
        truncated += "\n".join(tail)
        return {**m, "content": truncated}

    def _finalize(
        self,
        system_msgs: list[dict[str, Any]],
        original_others: list[dict[str, Any]],
        filtered: list[dict[str, Any]],
        before_tokens: int,
    ) -> CompactionResult:
        retained = {id(m) for m in filtered}
        removed = [m for m in original_others if id(m) not in retained]
        summary = summarize_removed(removed)
        after = estimate_messages_tokens(system_msgs + filtered)
        marker = {
            "role": "system",
            "content": (
                f"[Context compacted at {time.strftime('%H:%M:%S')}. "
                f"{len(removed)} messages removed. "
                f"Tokens: {before_tokens:,} → {after:,}]"
                + (f"\nSummary of removed conversation:\n{summary}" if summary else "")
            ),
        }
        result_msgs = system_msgs + [marker] + filtered
        self.compaction_history.append(
            {
                "timestamp": time.time(),
                "before_tokens": before_tokens,
                "after_tokens": estimate_messages_tokens(result_msgs),
                "messages_removed": len(removed),
                "compaction_level": self.compaction_level,
            }
        )
        self.compaction_level = min(self.compaction_level + 1, 3)
        return CompactionResult(
            messages=result_msgs,
            removed_count=len(removed),
            before_tokens=before_tokens,
            after_tokens=estimate_messages_tokens(result_msgs),
            summary=summary,
        )

    def compact_base_messages(self, messages: list[BaseMessage]) -> tuple[list[BaseMessage], CompactionResult]:
        """Compact a BaseMessage list; non-compacted messages are preserved by identity.

        Returns the new BaseMessage list and the compaction result. The returned
        messages combine: SystemMessages preserved, plus a synthetic SystemMessage
        carrying the compaction marker, plus any BaseMessages whose dicts survived.
        """
        result = self.compact(messages)
        dict_id_map = {id(message_to_dict(m)): m for m in messages}
        # identity won't match after dict creation; fallback by role+content
        out: list[BaseMessage] = []
        for d in result.messages:
            if d.get("role") == "system" and d.get("content", "").startswith("[Context compacted"):
                out.append(SystemMessage(content=d["content"]))
                continue
            match = self._find_match(messages, d)
            if match is not None:
                out.append(match)
        return out, result

    @staticmethod
    def _find_match(messages: list[BaseMessage], d: dict[str, Any]) -> BaseMessage | None:
        target_content = d.get("content", "")
        for m in messages:
            md = message_to_dict(m)
            if md.get("role") == d.get("role") and md.get("content", "") == target_content:
                return m
        return None
