from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from minicode.core.types import SessionMeta
from minicode.features.sessions.repository import checkpoint_from_messages


class Migrator:
    def __init__(
        self,
        paths: object,
        db: object,
        sessions: object,
        memory: object,
        permissions: object,
        skills: object,
        mcp: object,
        task_tracker: object,
        task_graph: object,
    ) -> None:
        self.paths = paths
        self.db = db
        self.sessions = sessions
        self.memory = memory
        self.permissions = permissions
        self.skills = skills
        self.mcp = mcp
        self.task_tracker = task_tracker
        self.task_graph = task_graph

    def migrate_once(self) -> None:
        if self.db.has_ledger("bootstrap_v1"):
            return
        self._migrate_settings()
        self._migrate_mcp()
        self._migrate_skills()
        self._migrate_permissions()
        self._migrate_memory()
        self._migrate_sessions()
        self._migrate_tasks()
        self.db.write_ledger("bootstrap_v1", {"status": "complete"})

    def _old_global_dir(self) -> Path:
        return Path(os.environ.get("MINICODE_LEGACY_HOME", Path.home() / ".mini-code")).resolve()

    def _project_old_dir(self) -> Path:
        return (Path.cwd() / ".mini-code").resolve()

    def _migrate_settings(self) -> None:
        old_dir = self._old_global_dir()
        for name in ("settings.json", "config.json"):
            candidate = old_dir / name
            if candidate.exists() and not self.paths.config_path.exists():
                self.paths.config_path.write_text(candidate.read_text(encoding="utf-8"), encoding="utf-8")
                self.db.write_ledger(f"settings:{candidate}")
                break

    def _migrate_mcp(self) -> None:
        old_dir = self._old_global_dir()
        candidates = [old_dir / "mcp_servers.json", Path.cwd() / ".mcp.json"]
        for candidate in candidates:
            if not candidate.exists():
                continue
            data = json.loads(candidate.read_text(encoding="utf-8") or "{}")
            servers = data.get("mcpServers", data)
            if isinstance(servers, dict):
                for name, payload in servers.items():
                    command = str(payload.get("command", ""))
                    args = payload.get("args", []) or []
                    env = payload.get("env", {}) or {}
                    cwd = payload.get("cwd")
                    if command:
                        self.mcp.add_server(name=name, command=command, args=list(args), env=dict(env), cwd=cwd)
            self.db.write_ledger(f"mcp:{candidate}")
            break

    def _migrate_skills(self) -> None:
        sources = [self._old_global_dir() / "skills", self._project_old_dir() / "skills"]
        for source in sources:
            if not source.exists():
                continue
            for skill_file in source.rglob("SKILL.md"):
                name = skill_file.parent.name
                self.skills.install_skill(name=name, body=skill_file.read_text(encoding="utf-8"), source=str(skill_file))
            self.db.write_ledger(f"skills:{source}")

    def _migrate_permissions(self) -> None:
        candidates = [self._old_global_dir() / "permissions.json", self._project_old_dir() / "permissions.json"]
        for candidate in candidates:
            if not candidate.exists():
                continue
            data = json.loads(candidate.read_text(encoding="utf-8") or "{}")
            if isinstance(data, dict):
                for key, decision in data.items():
                    self.permissions.store_decision(key, str(decision), {"source": str(candidate)})
            self.db.write_ledger(f"permissions:{candidate}")

    def _migrate_memory(self) -> None:
        memory_dir = self._old_global_dir() / "memory"
        if not memory_dir.exists():
            return
        for file_path in memory_dir.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in {".md", ".txt", ".json"}:
                scope = "user"
                if "project" in file_path.parts:
                    scope = "project"
                elif "local" in file_path.parts:
                    scope = "local"
                self.memory.put(
                    scope=scope,
                    workspace=str(Path.cwd().resolve()),
                    entry_key=file_path.stem,
                    content=file_path.read_text(encoding="utf-8", errors="replace"),
                )
        self.db.write_ledger(f"memory:{memory_dir}")

    def _migrate_sessions(self) -> None:
        sessions_dir = self._old_global_dir() / "sessions"
        if not sessions_dir.exists():
            return
        for file_path in sessions_dir.glob("*.json"):
            if self.db.has_ledger(f"session:{file_path.name}"):
                continue
            try:
                data = json.loads(file_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            raw_messages = data.get("messages", [])
            base_messages = []
            for message in raw_messages:
                role = str(message.get("role", ""))
                content = str(message.get("content", ""))
                if role == "system":
                    base_messages.append(SystemMessage(content=content))
                elif role == "user":
                    base_messages.append(HumanMessage(content=content))
                elif role == "assistant":
                    base_messages.append(AIMessage(content=content))
                elif role == "tool_result":
                    base_messages.append(
                        ToolMessage(
                            content=content,
                            tool_call_id=str(message.get("toolUseId", "legacy")),
                            name=message.get("toolName"),
                        )
                    )
            if not base_messages:
                continue
            thread_id = str(data.get("thread_id") or data.get("session_id") or uuid.uuid4().hex[:12])
            title = next((str(msg.get("content", ""))[:120] for msg in raw_messages if msg.get("role") == "user"), "")
            created_at = float(data.get("created_at", time.time()))
            updated_at = float(data.get("updated_at", created_at))
            workspace = str(data.get("workspace", Path.cwd().resolve()))
            self.sessions.upsert_session(
                SessionMeta(
                    thread_id=thread_id,
                    workspace=workspace,
                    created_at=created_at,
                    updated_at=updated_at,
                    model="legacy-import",
                    title=title,
                )
            )
            with self.db.checkpointer() as saver:
                saver.put(
                    {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}},
                    checkpoint_from_messages(base_messages),
                    {"source": "migration", "step": -1, "parents": {}},
                    {"messages": 1},
                )
            self.db.write_ledger(f"session:{file_path.name}")

    def _migrate_tasks(self) -> None:
        candidates = [self._old_global_dir() / "tasks.json", self._project_old_dir() / "tasks.json"]
        for candidate in candidates:
            if not candidate.exists():
                continue
            try:
                data = json.loads(candidate.read_text(encoding="utf-8") or "[]")
            except Exception:
                continue
            if isinstance(data, list):
                for item in data:
                    title = str(item.get("title", "")).strip()
                    if title:
                        self.task_tracker.add_task(title=title, note=str(item.get("note", "")), workspace=str(Path.cwd().resolve()))
            self.db.write_ledger(f"tasks:{candidate}")
