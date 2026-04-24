from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from minicode.core.types import ToolContext
from minicode.features.tasks import (
    TaskCycleError,
    TaskGraphError,
    TaskGraphService,
    TaskPriority,
    TaskState,
    TaskTrackerService,
)
from minicode.features.tasks.services import normalize_task_tracker_item
from minicode.features.tools.builtin import todo_write


class _FakeGraphRepo:
    def __init__(self) -> None:
        self.nodes: dict[str, dict] = {}
        self.edges: list[tuple[str, str]] = []

    def upsert_node(self, node_id, workspace, title, status, metadata):
        self.nodes[node_id] = {
            "node_id": node_id,
            "workspace": workspace,
            "title": title,
            "status": status,
            "metadata_json": json.dumps(metadata),
        }

    def add_edge(self, parent_id, child_id):
        self.edges.append((parent_id, child_id))

    def list_graph(self, workspace):
        return {
            "nodes": [n for n in self.nodes.values() if n["workspace"] == workspace],
            "edges": [{"parent_id": p, "child_id": c} for p, c in self.edges],
        }


class _FakeTrackerRepo:
    def __init__(self) -> None:
        self.items: list[dict] = []

    def add(self, workspace, title, note=""):
        self.items.append(
            {
                "id": len(self.items) + 1,
                "workspace": workspace,
                "title": title,
                "status": "open",
                "note": note,
                "created_at": 0,
                "updated_at": 0,
            }
        )

    def list(self, workspace):
        return [i for i in self.items if i["workspace"] == workspace]

    def set_status(self, workspace, task_id, status):
        for item in self.items:
            if item["workspace"] == workspace and item["id"] == task_id:
                item["status"] = status
                return True
        return False


def test_graph_add_and_list():
    svc = TaskGraphService(_FakeGraphRepo())
    a = svc.add_task_node("w", "A")
    b = svc.add_task_node("w", "B")
    svc.link(a, b, workspace="w")
    g = svc.get_graph("w")
    assert len(g["nodes"]) == 2
    assert len(g["edges"]) == 1


def test_graph_cycle_detection_rejects_edge():
    svc = TaskGraphService(_FakeGraphRepo())
    a = svc.add_task_node("w", "A")
    b = svc.add_task_node("w", "B")
    svc.link(a, b, workspace="w")
    with pytest.raises(TaskCycleError):
        svc.link(b, a, workspace="w")


def test_graph_self_loop_rejected():
    svc = TaskGraphService(_FakeGraphRepo())
    a = svc.add_task_node("w", "A")
    with pytest.raises(TaskGraphError):
        svc.link(a, a, workspace="w")


def test_graph_ready_tasks_respects_deps():
    svc = TaskGraphService(_FakeGraphRepo())
    a = svc.add_task_node("w", "A", priority=TaskPriority.LOW)
    b = svc.add_task_node("w", "B", priority=TaskPriority.HIGH)
    svc.link(a, b, workspace="w")
    # Initially only A is ready (no deps)
    ready = svc.ready_tasks("w")
    assert [n["node_id"] for n in ready] == [a]
    # After A completes, B becomes ready
    svc.transition("w", a, TaskState.RUNNING)
    svc.transition("w", a, TaskState.COMPLETED)
    ready = svc.ready_tasks("w")
    assert [n["node_id"] for n in ready] == [b]


def test_graph_invalid_transition():
    svc = TaskGraphService(_FakeGraphRepo())
    a = svc.add_task_node("w", "A")
    svc.transition("w", a, TaskState.RUNNING)
    svc.transition("w", a, TaskState.COMPLETED)
    with pytest.raises(TaskGraphError):
        svc.transition("w", a, TaskState.RUNNING)


def test_graph_progress_and_complete():
    svc = TaskGraphService(_FakeGraphRepo())
    a = svc.add_task_node("w", "A")
    b = svc.add_task_node("w", "B")
    svc.transition("w", a, TaskState.RUNNING)
    svc.transition("w", a, TaskState.COMPLETED)
    assert svc.progress_percentage("w") == pytest.approx(50.0)
    assert svc.is_complete("w") is False
    svc.transition("w", b, TaskState.RUNNING)
    svc.transition("w", b, TaskState.COMPLETED)
    assert svc.is_complete("w") is True


def test_graph_detect_cycle_none_on_dag():
    svc = TaskGraphService(_FakeGraphRepo())
    a = svc.add_task_node("w", "A")
    b = svc.add_task_node("w", "B")
    svc.link(a, b, workspace="w")
    assert svc.detect_cycle("w") is None


def test_graph_orphans():
    svc = TaskGraphService(_FakeGraphRepo())
    a = svc.add_task_node("w", "A")
    b = svc.add_task_node("w", "B")
    c = svc.add_task_node("w", "C")  # orphan
    svc.link(a, b, workspace="w")
    assert c in svc.orphans("w")
    assert a not in svc.orphans("w")


def test_tracker_set_status_validates():
    svc = TaskTrackerService(_FakeTrackerRepo())
    svc.add_task("T", "", "w")
    items = svc.list_tasks("w")
    assert svc.set_status("w", items[0]["id"], "completed") is True
    with pytest.raises(ValueError):
        svc.set_status("w", items[0]["id"], "bogus")


def test_tracker_summary_buckets():
    svc = TaskTrackerService(_FakeTrackerRepo())
    svc.add_task("A", "", "w")
    svc.add_task("B", "", "w")
    items = svc.list_tasks("w")
    svc.set_status("w", items[0]["id"], "completed")
    buckets = svc.summary("w")
    assert buckets["completed"] == 1
    assert buckets["open"] == 1


def test_normalize_task_tracker_item_accepts_string() -> None:
    assert normalize_task_tracker_item("Write README") == {
        "title": "Write README",
        "note": "",
    }


def test_normalize_task_tracker_item_accepts_mapping_aliases() -> None:
    assert normalize_task_tracker_item({"task": "Ship fix", "description": "today"}) == {
        "title": "Ship fix",
        "note": "today",
    }


def test_todo_write_accepts_string_items() -> None:
    tracker = TaskTrackerService(_FakeTrackerRepo())
    context = ToolContext(
        thread_id="thread-1",
        cwd="w",
        mode="default",
        services=SimpleNamespace(
            settings=SimpleNamespace(workspace="w"),
            task_tracker=tracker,
        ),
        emit_event=lambda _kind, _payload: None,
    )
    result = todo_write(["Write docs", "Run lint"], context=context)
    assert json.loads(result.content) == {"saved": ["Write docs", "Run lint"]}
    assert [item["title"] for item in tracker.list_tasks("w")] == ["Write docs", "Run lint"]
