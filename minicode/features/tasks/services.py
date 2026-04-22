from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any

from minicode.platform.process import start_background_command

from .types import (
    PRIORITY_RANK,
    TERMINAL_STATES,
    TaskCycleError,
    TaskGraphError,
    TaskPriority,
    TaskState,
)


# ---------------------------------------------------------------------------
# TaskTrackerService — flat workspace-scoped TODO list
# ---------------------------------------------------------------------------


class TaskTrackerService:
    """Simple session-local TODO-style tracker backed by task_items table.

    State machine: open → in_progress → {completed, blocked, cancelled}
    Unlike TaskGraphService this is flat (no deps).
    """

    VALID_STATUSES = frozenset({"open", "in_progress", "completed", "blocked", "cancelled"})

    def __init__(self, repository: Any) -> None:
        self.repository = repository

    def add_task(self, title: str, note: str, workspace: str) -> None:
        self.repository.add(workspace, title, note)

    def list_tasks(self, workspace: str) -> list[dict[str, Any]]:
        return self.repository.list(workspace)

    def set_status(self, workspace: str, task_id: int, status: str) -> bool:
        if status not in self.VALID_STATUSES:
            raise ValueError(f"invalid status: {status}")
        return self.repository.set_status(workspace, task_id, status)

    def summary(self, workspace: str) -> dict[str, int]:
        buckets = {s: 0 for s in self.VALID_STATUSES}
        for item in self.list_tasks(workspace):
            s = item.get("status", "open")
            buckets[s] = buckets.get(s, 0) + 1
        return buckets


# ---------------------------------------------------------------------------
# TaskGraphService — persistent DAG with a small state machine
# ---------------------------------------------------------------------------


class TaskGraphService:
    """Persistent DAG over task_graph_nodes + task_graph_edges tables.

    Boundaries:
    - `link` rejects edges that would introduce a cycle (Kahn detection).
    - `ready_tasks` filters nodes whose dependencies are all completed.
    - `transition` enforces a small state machine:
        pending → queued → running → {completed, failed, cancelled}
        any non-terminal → cancelled
    """

    def __init__(self, repository: Any) -> None:
        self.repository = repository

    # --- creation ---
    def add_task_node(
        self,
        workspace: str,
        title: str,
        status: str = TaskState.PENDING.value,
        metadata: dict[str, Any] | None = None,
        priority: TaskPriority | str = TaskPriority.NORMAL,
    ) -> str:
        node_id = uuid.uuid4().hex[:12]
        meta = dict(metadata or {})
        prio = priority.value if isinstance(priority, TaskPriority) else str(priority)
        meta.setdefault("priority", prio)
        self.repository.upsert_node(node_id, workspace, title, status, meta)
        return node_id

    def link(self, parent_id: str, child_id: str, workspace: str | None = None) -> None:
        if parent_id == child_id:
            raise TaskGraphError("self-loop not allowed")
        if workspace is not None and self._would_cycle(workspace, parent_id, child_id):
            raise TaskCycleError([parent_id, child_id, parent_id])
        self.repository.add_edge(parent_id, child_id)

    def get_graph(self, workspace: str) -> dict[str, Any]:
        return self.repository.list_graph(workspace)

    # --- state transitions ---
    _ALLOWED_TRANSITIONS: dict[str, set[str]] = {
        TaskState.PENDING.value: {
            TaskState.QUEUED.value,
            TaskState.RUNNING.value,
            TaskState.SKIPPED.value,
            TaskState.CANCELLED.value,
        },
        TaskState.QUEUED.value: {
            TaskState.RUNNING.value,
            TaskState.CANCELLED.value,
            TaskState.SKIPPED.value,
        },
        TaskState.RUNNING.value: {
            TaskState.COMPLETED.value,
            TaskState.FAILED.value,
            TaskState.CANCELLED.value,
        },
    }

    def transition(self, workspace: str, node_id: str, target: TaskState | str) -> None:
        target_str = target.value if isinstance(target, TaskState) else str(target)
        node = self._find_node(workspace, node_id)
        if node is None:
            raise TaskGraphError(f"node not found: {node_id}")
        current = node["status"]
        if current in TERMINAL_STATES and target_str != current:
            raise TaskGraphError(f"cannot transition terminal state {current} -> {target_str}")
        allowed = self._ALLOWED_TRANSITIONS.get(current, set())
        if target_str not in allowed and target_str != current:
            raise TaskGraphError(f"invalid transition {current} -> {target_str}")
        metadata = self._metadata_from_row(node)
        self.repository.upsert_node(node_id, workspace, node["title"], target_str, metadata)

    # --- queries ---
    def ready_tasks(self, workspace: str) -> list[dict[str, Any]]:
        graph = self.repository.list_graph(workspace)
        nodes = {n["node_id"]: n for n in graph["nodes"]}
        completed = {nid for nid, n in nodes.items() if n["status"] == TaskState.COMPLETED.value}
        running = {nid for nid, n in nodes.items() if n["status"] == TaskState.RUNNING.value}

        parents_of: dict[str, set[str]] = {nid: set() for nid in nodes}
        for edge in graph["edges"]:
            parents_of.setdefault(edge["child_id"], set()).add(edge["parent_id"])

        ready: list[dict[str, Any]] = []
        for nid, node in nodes.items():
            if node["status"] != TaskState.PENDING.value:
                continue
            if nid in running:
                continue
            deps = parents_of.get(nid, set())
            if deps and not all(dep in completed for dep in deps):
                continue
            ready.append(node)
        ready.sort(key=lambda n: PRIORITY_RANK.get(self._priority(n), 2))
        return ready

    def is_complete(self, workspace: str) -> bool:
        graph = self.repository.list_graph(workspace)
        if not graph["nodes"]:
            return True
        return all(n["status"] == TaskState.COMPLETED.value for n in graph["nodes"])

    def progress_percentage(self, workspace: str) -> float:
        graph = self.repository.list_graph(workspace)
        nodes = graph["nodes"]
        if not nodes:
            return 0.0
        done = sum(1 for n in nodes if n["status"] == TaskState.COMPLETED.value)
        return done / len(nodes) * 100.0

    def orphans(self, workspace: str) -> list[str]:
        """Nodes with no incoming or outgoing edges (potential leftover items)."""
        graph = self.repository.list_graph(workspace)
        ids = {n["node_id"] for n in graph["nodes"]}
        linked: set[str] = set()
        for edge in graph["edges"]:
            linked.add(edge["parent_id"])
            linked.add(edge["child_id"])
        return sorted(ids - linked)

    def detect_cycle(self, workspace: str) -> list[str] | None:
        graph = self.repository.list_graph(workspace)
        nodes = {n["node_id"] for n in graph["nodes"]}
        in_degree: dict[str, int] = {nid: 0 for nid in nodes}
        adj: dict[str, list[str]] = {nid: [] for nid in nodes}
        for edge in graph["edges"]:
            if edge["parent_id"] not in nodes or edge["child_id"] not in nodes:
                continue
            in_degree[edge["child_id"]] += 1
            adj[edge["parent_id"]].append(edge["child_id"])
        queue = [nid for nid, d in in_degree.items() if d == 0]
        visited = 0
        while queue:
            nid = queue.pop()
            visited += 1
            for child in adj[nid]:
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)
        if visited == len(nodes):
            return None
        remaining = [nid for nid, d in in_degree.items() if d > 0]
        return remaining

    # --- internals ---
    def _find_node(self, workspace: str, node_id: str) -> dict[str, Any] | None:
        graph = self.repository.list_graph(workspace)
        for node in graph["nodes"]:
            if node["node_id"] == node_id:
                return node
        return None

    def _would_cycle(self, workspace: str, parent_id: str, child_id: str) -> bool:
        # BFS from child_id; if we reach parent_id, adding edge parent→child creates a cycle.
        graph = self.repository.list_graph(workspace)
        adj: dict[str, list[str]] = {}
        for edge in graph["edges"]:
            adj.setdefault(edge["parent_id"], []).append(edge["child_id"])
        stack = [child_id]
        seen: set[str] = set()
        while stack:
            nid = stack.pop()
            if nid == parent_id:
                return True
            if nid in seen:
                continue
            seen.add(nid)
            stack.extend(adj.get(nid, []))
        return False

    @staticmethod
    def _metadata_from_row(node: dict[str, Any]) -> dict[str, Any]:
        raw = node.get("metadata_json") or "{}"
        if isinstance(raw, dict):
            return raw
        try:
            return json.loads(raw)
        except (TypeError, json.JSONDecodeError):
            return {}

    @classmethod
    def _priority(cls, node: dict[str, Any]) -> TaskPriority:
        meta = cls._metadata_from_row(node)
        raw = meta.get("priority", TaskPriority.NORMAL.value)
        try:
            return TaskPriority(raw)
        except ValueError:
            return TaskPriority.NORMAL


# ---------------------------------------------------------------------------
# BackgroundTaskService — bounded pool for long-running shell tasks
# ---------------------------------------------------------------------------


class BackgroundTaskService:
    """Bounded background task runner.

    Boundaries (resource tier):
    - `max_slots` caps concurrency (default 4).
    - `timeout_s` enforces a wall-clock timeout per task (default 1800s = 30 min).
    - `max_output_bytes` caps stdout log size; overflow truncates to head+tail.
    """

    DEFAULT_MAX_SLOTS = 4
    DEFAULT_TIMEOUT_S = 1800.0
    DEFAULT_MAX_OUTPUT = 10 * 1024 * 1024  # 10 MB

    def __init__(
        self,
        repository: Any,
        logs_dir: str,
        *,
        max_slots: int = DEFAULT_MAX_SLOTS,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        max_output_bytes: int = DEFAULT_MAX_OUTPUT,
    ) -> None:
        self.repository = repository
        self.logs_dir = logs_dir
        self.max_slots = max(1, max_slots)
        self.timeout_s = timeout_s
        self.max_output_bytes = max_output_bytes
        self._processes: dict[str, Any] = {}
        self._started_at: dict[str, float] = {}

    # --- slot management ---
    def used_slots(self) -> int:
        return sum(1 for p in self._processes.values() if p.poll() is None)

    def available_slots(self) -> int:
        return max(0, self.max_slots - self.used_slots())

    def can_start(self) -> bool:
        return self.available_slots() > 0

    def slot_stats(self) -> dict[str, Any]:
        used = self.used_slots()
        return {
            "used_slots": used,
            "max_slots": self.max_slots,
            "available_slots": self.max_slots - used,
            "total_tracked": len(self._processes),
        }

    # --- lifecycle ---
    def start(self, command: str, cwd: str):
        if not self.can_start():
            raise RuntimeError(
                f"background slot pool exhausted ({self.used_slots()}/{self.max_slots})"
            )
        process, record = start_background_command(command, cwd, self.logs_dir)
        self._processes[record.task_id] = process
        self._started_at[record.task_id] = time.time()
        self.repository.upsert(record)
        return record

    def cancel(self, task_id: str) -> bool:
        process = self._processes.get(task_id)
        if process is None:
            return False
        if process.poll() is None:
            try:
                process.terminate()
            except OSError:
                pass
        return True

    def refresh(self) -> list[dict[str, Any]]:
        now = time.time()
        for task_id, process in list(self._processes.items()):
            started = self._started_at.get(task_id, now)
            if process.poll() is None:
                if now - started > self.timeout_s:
                    try:
                        process.kill()
                    except OSError:
                        pass
                    self._mark_terminated(task_id, status="timeout", return_code=-1)
                    self._processes.pop(task_id, None)
                continue
            status = "finished" if process.returncode == 0 else "failed"
            self._mark_terminated(task_id, status=status, return_code=process.returncode)
            self._processes.pop(task_id, None)
        return self.repository.list()

    # --- output ---
    def read_output(self, task_id: str) -> str:
        record = next((r for r in self.repository.list() if r["task_id"] == task_id), None)
        if not record or not record.get("output_path"):
            return ""
        path = Path(record["output_path"])
        if not path.exists():
            return ""
        try:
            raw = path.read_bytes()
        except OSError:
            return ""
        if len(raw) <= self.max_output_bytes:
            return raw.decode("utf-8", errors="replace")
        head_chunk = int(self.max_output_bytes * 0.7)
        tail_chunk = self.max_output_bytes - head_chunk
        head = raw[:head_chunk].decode("utf-8", errors="replace")
        tail = raw[-tail_chunk:].decode("utf-8", errors="replace")
        return (
            head
            + f"\n... [{len(raw) - self.max_output_bytes} bytes truncated] ...\n"
            + tail
        )

    # --- internals ---
    def _mark_terminated(self, task_id: str, *, status: str, return_code: int | None) -> None:
        existing = next(
            (item for item in self.repository.list() if item["task_id"] == task_id), None
        )
        if not existing:
            return
        existing["status"] = status
        existing["return_code"] = return_code
        existing["updated_at"] = time.time()
        record = type("Background", (), existing)
        self.repository.upsert(record)
