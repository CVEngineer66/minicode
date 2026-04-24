from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable

from minicode.features.sessions import format_session_time

from .commands import SLASH_COMMANDS


@dataclass(slots=True)
class DispatchResult:
    handled: bool
    output: str = ""
    quit: bool = False


def _fmt_kv(pairs: dict[str, Any], max_key_width: int = 20) -> str:
    lines: list[str] = []
    for key, value in pairs.items():
        lines.append(f"  {key:<{max_key_width}} {value}")
    return "\n".join(lines)


class SlashDispatcher:
    """Resolves user-entered slash commands against live AppServices.

    Each handler returns a DispatchResult. Unknown commands return handled=False
    so the TUI can either show the /help list or forward the text as a prompt.
    """

    def __init__(self, services: Any) -> None:
        self.services = services
        self._handlers: dict[str, Callable[[str], DispatchResult]] = {
            "help": self._help,
            "quit": self._quit,
            "exit": self._quit,
            "clear": self._clear,
            "sessions": self._sessions,
            "resume": self._resume,
            "branch": self._branch,
            "compact": self._compact,
            "mode": self._mode,
            "model": self._model,
            "memory": self._memory,
            "profile": self._profile,
            "cost": self._cost,
            "context": self._context,
            "tasks": self._tasks,
            "agents": self._agents,
            "skills": self._skills,
            "mcp": self._mcp,
            "hooks": self._hooks,
            "permissions": self._permissions,
        }

    def dispatch(self, line: str) -> DispatchResult:
        stripped = line.strip()
        if not stripped.startswith("/"):
            return DispatchResult(handled=False)
        parts = stripped[1:].split(maxsplit=1)
        if not parts:
            return DispatchResult(handled=True, output=self._render_help())
        name = parts[0].lower()
        rest = parts[1] if len(parts) > 1 else ""
        handler = self._handlers.get(name)
        if handler is None:
            return DispatchResult(
                handled=True,
                output=f"Unknown command: /{name}. Type /help for the list.",
            )
        try:
            return handler(rest)
        except BaseException as exc:  # noqa: BLE001
            return DispatchResult(handled=True, output=f"Command failed: {exc}")

    # ---------- handlers ----------

    def _help(self, _rest: str) -> DispatchResult:
        return DispatchResult(handled=True, output=self._render_help())

    def _render_help(self) -> str:
        lines = ["Available commands:"]
        for cmd in SLASH_COMMANDS:
            lines.append(f"  {cmd.usage:<30} {cmd.description}")
        return "\n".join(lines)

    def _quit(self, _rest: str) -> DispatchResult:
        return DispatchResult(handled=True, output="Exiting.", quit=True)

    def _clear(self, _rest: str) -> DispatchResult:
        return DispatchResult(handled=True, output="\x1b[2J\x1b[H")

    def _sessions(self, _rest: str) -> DispatchResult:
        sessions = self.services.sessions.list_sessions(
            workspace=self.services.settings.workspace
        )
        if not sessions:
            return DispatchResult(handled=True, output="No sessions in this workspace.")
        lines = [f"{s.thread_id}\t{format_session_time(s.updated_at)}\t{s.title}" for s in sessions[:20]]
        return DispatchResult(handled=True, output="\n".join(lines))

    def _resume(self, rest: str) -> DispatchResult:
        target = rest.strip() or "latest"
        if target == "latest":
            latest = self.services.sessions.get_latest_session(
                workspace=self.services.settings.workspace
            )
            if latest is None:
                return DispatchResult(handled=True, output="No sessions in this workspace.")
            return DispatchResult(
                handled=True,
                output=f"Latest: {latest.thread_id} -> {latest.title}",
            )
        session = self.services.sessions.get_session(target)
        if session is None:
            return DispatchResult(handled=True, output=f"Unknown session: {target}")
        return DispatchResult(
            handled=True,
            output=f"Resumed context for {session.thread_id} ({session.title})",
        )

    def _branch(self, rest: str) -> DispatchResult:
        source = rest.strip()
        if not source:
            latest = self.services.sessions.get_latest_session()
            source = latest.thread_id if latest else ""
        if not source:
            return DispatchResult(handled=True, output="No session to branch from.")
        new_id = self.services.sessions.branch(source)
        return DispatchResult(handled=True, output=f"Branched {source[:8]} -> {new_id}")

    def _compact(self, rest: str) -> DispatchResult:
        thread = rest.strip()
        if not thread:
            latest = self.services.sessions.get_latest_session()
            thread = latest.thread_id if latest else ""
        if not thread:
            return DispatchResult(handled=True, output="No session to compact.")
        ctx = getattr(self.services, "context", None)
        if ctx is None:
            return DispatchResult(handled=True, output="Context service unavailable.")
        removed = self.services.sessions.compact(thread, ctx)
        return DispatchResult(
            handled=True,
            output=f"Compacted session {thread[:8]}: {removed} messages removed.",
        )

    def _mode(self, rest: str) -> DispatchResult:
        auto = getattr(self.services, "auto", None)
        if auto is None:
            return DispatchResult(handled=True, output="Mode service unavailable.")
        target = rest.strip()
        if not target:
            return DispatchResult(
                handled=True,
                output=f"Current mode: {auto.get_mode().value}",
            )
        try:
            msg = auto.set_mode(target)
        except ValueError as exc:
            return DispatchResult(handled=True, output=f"Invalid mode: {exc}")
        self.services.settings.auto_mode = auto.get_mode().value
        return DispatchResult(handled=True, output=msg)

    def _model(self, rest: str) -> DispatchResult:
        settings = self.services.settings
        target = rest.strip()
        if not target:
            return DispatchResult(handled=True, output=f"Current model: {settings.model}")
        settings.model = target
        if getattr(self.services, "context", None):
            self.services.context.set_model(target)
        sessions = getattr(self.services, "sessions", None)
        if sessions is not None and hasattr(sessions, "model"):
            sessions.model = target
        return DispatchResult(handled=True, output=f"Model set to {target}")

    def _memory(self, rest: str) -> DispatchResult:
        mem = getattr(self.services, "memory", None)
        if mem is None:
            return DispatchResult(handled=True, output="Memory service unavailable.")
        parts = rest.split(maxsplit=1)
        if parts and parts[0] == "search" and len(parts) > 1:
            hits = mem.search(parts[1])
            if not hits:
                return DispatchResult(
                    handled=True,
                    output=f"No memory matches for '{parts[1]}'",
                )
            lines = [
                f"[{hit['scope']}] {hit.get('entry_key', '')}: {hit.get('content', '')[:100]}"
                for hit in hits
            ]
            return DispatchResult(handled=True, output="\n".join(lines))
        stats = mem.stats()
        return DispatchResult(handled=True, output="Memory stats:\n" + _fmt_kv(stats))

    def _profile(self, _rest: str) -> DispatchResult:
        prof = getattr(self.services, "profile", None)
        if prof is None:
            return DispatchResult(handled=True, output="Profile service unavailable.")
        block = prof.to_prompt_section(prof.load_merged())
        return DispatchResult(handled=True, output=block or "No profile configured.")

    def _cost(self, _rest: str) -> DispatchResult:
        cost = getattr(self.services, "cost", None)
        if cost is None:
            return DispatchResult(handled=True, output="Cost service unavailable.")
        return DispatchResult(
            handled=True,
            output=cost.short_summary() + "\n" + _fmt_kv(cost.stats()),
        )

    def _context(self, _rest: str) -> DispatchResult:
        ctx = getattr(self.services, "context", None)
        if ctx is None:
            return DispatchResult(handled=True, output="Context service unavailable.")
        latest = self.services.sessions.get_latest_session()
        if latest is None:
            return DispatchResult(handled=True, output="No active session.")
        messages = self.services.sessions.load_messages(latest.thread_id)
        stats = ctx.stats(messages)
        data = {
            "model": ctx.manager.model,
            "window": stats.context_window,
            "messages": stats.messages_count,
            "tokens": stats.total_tokens,
            "usage": f"{stats.usage_percentage:.1f}%",
            "near_limit": stats.is_near_limit,
            "should_compact": stats.should_compact,
        }
        return DispatchResult(handled=True, output="Context:\n" + _fmt_kv(data))

    def _tasks(self, _rest: str) -> DispatchResult:
        tracker = getattr(self.services, "task_tracker", None)
        if tracker is None:
            return DispatchResult(handled=True, output="Task tracker unavailable.")
        items = tracker.list_tasks(self.services.settings.workspace)
        if not items:
            return DispatchResult(handled=True, output="No tracked tasks.")
        lines = [f"[{item['status']}] #{item['id']} {item['title']}" for item in items[:20]]
        return DispatchResult(handled=True, output="\n".join(lines))

    def _agents(self, _rest: str) -> DispatchResult:
        collab = getattr(self.services, "collaboration", None)
        if collab is None or not hasattr(collab, "format_status"):
            return DispatchResult(handled=True, output="Collaboration service unavailable.")
        return DispatchResult(handled=True, output=collab.format_status())

    def _skills(self, _rest: str) -> DispatchResult:
        skills = getattr(self.services, "skills", None)
        if skills is None:
            return DispatchResult(handled=True, output="Skills service unavailable.")
        items = skills.list_skills()
        if not items:
            return DispatchResult(handled=True, output="No installed skills.")
        lines = [
            f"{item.get('name', '?')} (source: {item.get('source', '?')})" for item in items
        ]
        return DispatchResult(handled=True, output="\n".join(lines))

    def _mcp(self, _rest: str) -> DispatchResult:
        mcp = getattr(self.services, "mcp", None)
        if mcp is None:
            return DispatchResult(handled=True, output="MCP service unavailable.")
        servers = mcp.list_servers()
        if not servers:
            return DispatchResult(handled=True, output="No MCP servers configured.")
        lines = [
            f"{item['name']}: {item['command']} {' '.join(item.get('args', []))}"
            for item in servers
        ]
        return DispatchResult(handled=True, output="\n".join(lines))

    def _hooks(self, _rest: str) -> DispatchResult:
        hooks = getattr(self.services, "hooks", None)
        if hooks is None or not hasattr(hooks, "stats"):
            return DispatchResult(handled=True, output="Hook stats unavailable.")
        return DispatchResult(handled=True, output="Hooks:\n" + _fmt_kv(hooks.stats()))

    def _permissions(self, _rest: str) -> DispatchResult:
        perms = getattr(self.services, "permissions", None)
        if perms is None or not hasattr(perms, "summary"):
            return DispatchResult(handled=True, output="Permissions summary unavailable.")
        summary = perms.summary()
        patterns = json.dumps(summary.pop("patterns", {}), indent=2, ensure_ascii=False)
        return DispatchResult(
            handled=True,
            output="Permissions:\n" + _fmt_kv(summary) + "\n" + patterns,
        )
