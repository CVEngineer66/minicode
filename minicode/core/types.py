from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from langchain_core.messages import BaseMessage


@dataclass(slots=True)
class GraphEvent:
    kind: str
    payload: dict[str, Any] = field(default_factory=dict)
    timestamp: float = 0.0


@dataclass(slots=True)
class RunTurnResult:
    thread_id: str
    messages: list[BaseMessage]
    final_text: str | None
    interrupt: dict[str, Any] | None
    await_user: bool
    error: str | None
    events: list[GraphEvent] = field(default_factory=list)


@dataclass(slots=True)
class ToolCapability:
    concurrency_safe: bool = True
    reads_files: bool = False
    writes_files: bool = False
    shell: bool = False
    network: bool = False
    interactive: bool = False
    long_running: bool = False
    task: bool = False

    @property
    def requires_serial_execution(self) -> bool:
        return self.interactive or self.long_running or self.task

    @property
    def risk_level(self) -> str:
        if self.task or self.shell or self.writes_files:
            return "high"
        if self.network or self.long_running:
            return "medium"
        return "low"


@dataclass(slots=True)
class PermissionPolicy:
    kind: str
    persist_scope: str = "session"
    always_require_approval: bool = False


@dataclass(slots=True)
class ToolResult:
    ok: bool = True
    content: str = ""
    structured: Any = None
    await_user: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    def as_text(self) -> str:
        if self.content:
            return self.content
        if self.structured is None:
            return ""
        return str(self.structured)


class ToolExecutor(Protocol):
    def __call__(self, arguments: dict[str, Any], context: "ToolContext") -> ToolResult:
        ...


class ToolValidator(Protocol):
    def __call__(self, arguments: dict[str, Any]) -> dict[str, Any]:
        ...


@dataclass(slots=True)
class ToolSpec:
    name: str
    description: str
    input_schema: dict[str, Any]
    capability: ToolCapability
    permission_policy: PermissionPolicy
    validator: ToolValidator
    executor: ToolExecutor

    def to_model_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.input_schema,
            },
        }


@dataclass(slots=True)
class ToolContext:
    thread_id: str
    cwd: str
    mode: str
    services: "AppServices"
    emit_event: Callable[[str, dict[str, Any]], None]
    subtask_runner: Callable[[str, str, str], str] | None = None


@dataclass(slots=True)
class SessionMeta:
    thread_id: str
    workspace: str
    created_at: float
    updated_at: float
    model: str
    title: str


@dataclass(slots=True)
class BackgroundTaskRecord:
    task_id: str
    command: str
    cwd: str
    status: str
    created_at: float
    updated_at: float
    return_code: int | None = None
    output_path: str | None = None


@dataclass(slots=True)
class AgentCard:
    name: str
    description: str
    skills: list[str] = field(default_factory=list)
    status: str = "idle"


@dataclass(slots=True)
class AppServices:
    paths: Any
    db: Any
    settings: Any
    sessions: Any
    memory: Any
    permissions: Any
    tools: Any
    task_tracker: Any
    task_graph: Any
    background_tasks: Any
    mcp: Any
    skills: Any
    collaboration: Any
    hooks: Any
    runtime_events: Any
    migrator: Any
    context: Any = None
    profile: Any = None
    cost: Any = None
    execution: Any = None
    auto: Any = None
