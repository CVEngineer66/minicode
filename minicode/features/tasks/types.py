from __future__ import annotations

from enum import Enum


class TaskState(str, Enum):
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    CANCELLED = "cancelled"


class TaskPriority(str, Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


PRIORITY_RANK: dict[TaskPriority, int] = {
    TaskPriority.CRITICAL: 0,
    TaskPriority.HIGH: 1,
    TaskPriority.NORMAL: 2,
    TaskPriority.LOW: 3,
}


TERMINAL_STATES = frozenset(
    {
        TaskState.COMPLETED.value,
        TaskState.FAILED.value,
        TaskState.SKIPPED.value,
        TaskState.CANCELLED.value,
    }
)


class TaskGraphError(RuntimeError):
    pass


class TaskCycleError(TaskGraphError):
    def __init__(self, cycle: list[str]) -> None:
        super().__init__(f"Cycle detected in task graph: {' -> '.join(cycle)}")
        self.cycle = cycle
