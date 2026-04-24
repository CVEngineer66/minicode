from __future__ import annotations

import time
import uuid
from dataclasses import replace
from typing import Any

from minicode.core.types import AgentSpec, AppServices, WorkerRun
from minicode.features.tasks import TaskPriority, TaskState
from minicode.features.tools.registry import ToolRegistry


_EXPLORE_TOOLS = (
    "ask_user",
    "file_tree",
    "grep_files",
    "list_files",
    "load_skill",
    "read_file",
)
_PLAN_TOOLS = _EXPLORE_TOOLS + (
    "ask_user_choice",
    "plan_tasks",
    "todo_write",
)


def _truncate_title(text: str, fallback: str = "Untitled task") -> str:
    line = next((part.strip() for part in text.splitlines() if part.strip()), "")
    return (line or fallback)[:120]


def _contains_any(text: str, patterns: tuple[str, ...]) -> bool:
    lowered = text.lower()
    return any(pattern in lowered for pattern in patterns)


class CollaborationService:
    def __init__(self, repository: object) -> None:
        self.repository = repository
        self._agents = self._build_builtin_agents()
        self._workers: dict[str, WorkerRun] = {}
        self._worker_by_node: dict[str, str] = {}

    # --- built-in agents -------------------------------------------------
    def _build_builtin_agents(self) -> dict[str, AgentSpec]:
        return {
            "Explore": AgentSpec(
                name="Explore",
                description="Read-only research agent for gathering code and environment context.",
                system_prompt=(
                    "You are the Explore subagent. Work read-only. Gather concrete evidence from the "
                    "codebase, summarize findings, and avoid proposing speculative changes."
                ),
                allowed_tools=_EXPLORE_TOOLS,
                default_mode="default",
                max_steps=10,
                spawn_allowed=False,
            ),
            "Plan": AgentSpec(
                name="Plan",
                description=(
                    "Planning agent for decomposing work, surfacing tradeoffs, and asking the user "
                    "to choose among materially different approaches."
                ),
                system_prompt=(
                    "You are the Plan subagent. Break work into concrete tasks, use ask_user_choice "
                    "when there are materially different approaches with real tradeoffs, and do not "
                    "modify code directly."
                ),
                allowed_tools=_PLAN_TOOLS,
                default_mode="default",
                max_steps=12,
                spawn_allowed=False,
            ),
            "general-purpose": AgentSpec(
                name="general-purpose",
                description=(
                    "Execution agent for intent-driven implementation, review, or validation work."
                ),
                system_prompt=(
                    "You are the general-purpose execution subagent. Complete only the requested "
                    "task. Do not assume review or testing is required unless the user asked for it "
                    "or the current task explicitly depends on it."
                ),
                allowed_tools=("*",),
                default_mode="default",
                max_steps=16,
                spawn_allowed=False,
            ),
        }

    def list_agents(self) -> list[dict[str, Any]]:
        cards: list[dict[str, Any]] = []
        active_counts: dict[str, int] = {}
        for worker in self._workers.values():
            if worker.status not in {
                TaskState.COMPLETED.value,
                TaskState.FAILED.value,
                TaskState.CANCELLED.value,
                TaskState.SKIPPED.value,
            }:
                active_counts[worker.agent_name] = active_counts.get(worker.agent_name, 0) + 1
        for spec in self._agents.values():
            cards.append(
                {
                    "name": spec.name,
                    "description": spec.description,
                    "default_mode": spec.default_mode,
                    "max_steps": spec.max_steps,
                    "spawn_allowed": spec.spawn_allowed,
                    "allowed_tools": list(spec.allowed_tools),
                    "active_workers": active_counts.get(spec.name, 0),
                }
            )
        return cards

    def list_workers(self) -> list[dict[str, Any]]:
        rows = sorted(self._workers.values(), key=lambda worker: worker.updated_at, reverse=True)
        return [
            {
                "worker_id": worker.worker_id,
                "node_id": worker.node_id,
                "title": worker.title,
                "agent_name": worker.agent_name,
                "task_type": worker.task_type,
                "status": worker.status,
                "execution_mode": worker.execution_mode,
                "thread_id": worker.thread_id,
                "write_scope": list(worker.write_scope),
                "summary": worker.summary,
                "error": worker.error,
            }
            for worker in rows
        ]

    def get_worker(self, worker_id: str) -> WorkerRun:
        return self._workers[worker_id]

    def format_status(self) -> str:
        lines = []
        for agent in self.list_agents():
            lines.append(
                f"{agent['name']}: {agent['description']} "
                f"[mode={agent['default_mode']}, active={agent['active_workers']}]"
            )
        workers = self.list_workers()
        if workers:
            lines.append("")
            lines.append("Workers:")
            for worker in workers[:10]:
                lines.append(
                    f"- {worker['worker_id']} {worker['status']} "
                    f"{worker['agent_name']}::{worker['task_type']} -> {worker['title']}"
                )
        return "\n".join(lines) if lines else "No built-in agents configured."

    # --- task submission -------------------------------------------------
    def infer_agent(self, prompt: str, mode_hint: str = "") -> str:
        hint = (mode_hint or "").strip().lower()
        if hint in {"explore", "research", "investigate"}:
            return "Explore"
        if hint in {"plan", "planning", "design"}:
            return "Plan"
        text = f"{hint}\n{prompt}".lower()
        if _contains_any(text, ("compare", "tradeoff", "trade-off", "options", "approach", "design")):
            return "Plan"
        if _contains_any(text, ("explore", "research", "investigate", "inspect", "understand")):
            return "Explore"
        return "general-purpose"

    def infer_task_type(self, prompt: str, mode_hint: str = "") -> str:
        hint = (mode_hint or "").strip().lower()
        if hint in {"review", "code_review"}:
            return "review"
        if hint in {"test", "tests", "validation", "validate"}:
            return "validation"
        if hint in {"plan", "planning", "design"}:
            return "planning"
        if hint in {"explore", "research", "investigate"}:
            return "research"
        lowered = prompt.lower()
        if _contains_any(lowered, ("review", "audit", "inspect diff", "look for regressions")):
            return "review"
        if _contains_any(lowered, ("test", "validate", "verification", "verify")):
            return "validation"
        if _contains_any(lowered, ("plan", "design", "compare", "tradeoff", "options")):
            return "planning"
        if _contains_any(lowered, ("explore", "research", "investigate", "understand")):
            return "research"
        return "implementation"

    def submit_task(
        self,
        *,
        services: AppServices,
        prompt: str,
        parent_thread_id: str,
        current_mode: str | None = None,
        mode_hint: str = "",
        execution_mode: str = "foreground",
        write_scope: list[str] | None = None,
        depends_on: list[str] | None = None,
        title: str | None = None,
        task_type: str | None = None,
        agent_name: str | None = None,
        priority: TaskPriority | str = TaskPriority.NORMAL,
    ) -> dict[str, Any]:
        workspace = services.settings.workspace
        resolved_agent = agent_name or self.infer_agent(prompt, mode_hint)
        if resolved_agent not in self._agents:
            resolved_agent = self.infer_agent(prompt, mode_hint)
        resolved_task_type = task_type or self.infer_task_type(prompt, mode_hint)
        execution = "background" if str(execution_mode).lower() == "background" else "foreground"
        now = time.time()
        worker_id = uuid.uuid4().hex[:12]
        node_title = _truncate_title(title or prompt)
        metadata = {
            "owner_agent": resolved_agent,
            "task_type": resolved_task_type,
            "execution_mode": execution,
            "write_scope": list(write_scope or []),
            "prompt": prompt,
            "parent_thread_id": parent_thread_id,
            "worker_id": worker_id,
            "priority": priority.value if isinstance(priority, TaskPriority) else str(priority),
        }
        node_id = services.task_graph.add_task_node(
            workspace=workspace,
            title=node_title,
            status=TaskState.PENDING.value,
            metadata=metadata,
            priority=priority,
        )
        for parent_id in depends_on or []:
            services.task_graph.link(parent_id, node_id, workspace=workspace)
        worker_mode = (
            "bypass"
            if (current_mode or services.settings.auto_mode) == "bypass"
            else self._agents[resolved_agent].default_mode
        )
        worker = WorkerRun(
            worker_id=worker_id,
            node_id=node_id,
            title=node_title,
            prompt=prompt,
            agent_name=resolved_agent,
            thread_id=f"{parent_thread_id}:{worker_id}",
            mode=worker_mode,
            task_type=resolved_task_type,
            status=TaskState.PENDING.value,
            execution_mode=execution,
            write_scope=list(write_scope or []),
            created_at=now,
            updated_at=now,
        )
        self._workers[worker_id] = worker
        self._worker_by_node[node_id] = worker_id
        self.post_task(
            channel=parent_thread_id,
            sender="orchestrator",
            recipient=resolved_agent,
            content=f"submitted:{node_id}:{resolved_task_type}:{node_title}",
        )
        return self._serialize_worker(worker)

    def submit_plan(
        self,
        *,
        services: AppServices,
        items: list[dict[str, Any]],
        parent_thread_id: str,
        current_mode: str | None = None,
        edges: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        created: dict[str, dict[str, Any]] = {}
        order: list[str] = []
        for index, item in enumerate(items, start=1):
            local_id = str(item.get("id") or f"task_{index}")
            prompt = str(item.get("prompt") or item.get("title") or "").strip()
            if not prompt:
                continue
            worker = self.submit_task(
                services=services,
                prompt=prompt,
                parent_thread_id=parent_thread_id,
                current_mode=current_mode,
                mode_hint=str(item.get("agent") or item.get("mode") or ""),
                execution_mode=str(item.get("execution_mode") or "foreground"),
                write_scope=[
                    str(part) for part in list(item.get("write_scope") or item.get("paths") or [])
                ],
                title=str(item.get("title") or prompt),
                task_type=str(item.get("task_type") or "").strip() or None,
                agent_name=str(item.get("agent_name") or "").strip() or None,
                priority=str(item.get("priority") or TaskPriority.NORMAL.value),
            )
            created[local_id] = worker
            order.append(local_id)
        for index, item in enumerate(items, start=1):
            local_id = str(item.get("id") or f"task_{index}")
            current = created.get(local_id)
            if current is None:
                continue
            node_id = current["node_id"]
            for dep in list(item.get("depends_on") or []):
                parent = created.get(str(dep))
                if parent is not None:
                    services.task_graph.link(parent["node_id"], node_id, workspace=services.settings.workspace)
        for edge in list(edges or []):
            parent = created.get(str(edge.get("from") or edge.get("parent") or ""))
            child = created.get(str(edge.get("to") or edge.get("child") or ""))
            if parent is None or child is None:
                continue
            services.task_graph.link(
                parent["node_id"],
                child["node_id"],
                workspace=services.settings.workspace,
            )
        return {
            "created": [created[item_id] for item_id in order if item_id in created],
            "graph_progress": services.task_graph.progress_percentage(services.settings.workspace),
        }

    # --- scheduling / execution -----------------------------------------
    def ready_workers(self, services: AppServices) -> list[WorkerRun]:
        ready_nodes = services.task_graph.ready_tasks(services.settings.workspace)
        runs: list[WorkerRun] = []
        for node in ready_nodes:
            worker_id = self._worker_by_node.get(node["node_id"])
            if not worker_id:
                continue
            worker = self._workers.get(worker_id)
            if worker is None:
                continue
            if worker.status in {TaskState.COMPLETED.value, TaskState.FAILED.value, TaskState.CANCELLED.value}:
                continue
            runs.append(worker)
        return runs

    def can_run_in_parallel(self, worker: WorkerRun, running_workers: list[WorkerRun]) -> bool:
        if worker.execution_mode != "foreground":
            return False
        if worker.agent_name != "Explore":
            return False
        worker_scope = {item for item in worker.write_scope if item}
        for active in running_workers:
            active_scope = {item for item in active.write_scope if item}
            if worker_scope and active_scope and worker_scope.intersection(active_scope):
                return False
            if active.execution_mode != "foreground":
                return False
            if active.agent_name != "Explore":
                return False
        return True

    def start_worker(
        self,
        *,
        services: AppServices,
        worker_id: str,
        mode: str | None = None,
        resume: dict[str, Any] | None = None,
    ):
        worker = self._workers[worker_id]
        services.task_graph.transition(
            services.settings.workspace,
            worker.node_id,
            TaskState.QUEUED.value if worker.status == TaskState.PENDING.value else worker.status,
        )
        self._set_worker_status(worker_id, TaskState.RUNNING.value)
        services.task_graph.transition(
            services.settings.workspace,
            worker.node_id,
            TaskState.RUNNING.value,
        )
        self.post_task(
            channel=worker.thread_id,
            sender="orchestrator",
            recipient=worker.agent_name,
            content=f"started:{worker.node_id}:{worker.task_type}",
        )
        result = self._run_worker_turn(
            services=services,
            worker=worker,
            mode=mode or worker.mode,
            resume=resume,
        )
        self._apply_result(services, worker_id, result)
        return result

    def queue_worker(self, *, services: AppServices, worker_id: str) -> WorkerRun:
        worker = self._workers[worker_id]
        if worker.status == TaskState.PENDING.value:
            services.task_graph.transition(
                services.settings.workspace,
                worker.node_id,
                TaskState.QUEUED.value,
            )
            self._set_worker_status(worker_id, TaskState.QUEUED.value)
        return worker

    def _run_worker_turn(
        self,
        *,
        services: AppServices,
        worker: WorkerRun,
        mode: str,
        resume: dict[str, Any] | None,
    ):
        from minicode.runtime.runner import run_turn

        spec = self._agents[worker.agent_name]
        worker_services = self._build_worker_services(services, spec, mode)
        return run_turn(
            services=worker_services,
            prompt=None if resume is not None else worker.prompt,
            thread_id=worker.thread_id,
            resume=resume,
            mode=mode,
            max_steps=spec.max_steps,
            persist=True,
        )

    def _build_worker_services(
        self,
        services: AppServices,
        spec: AgentSpec,
        mode: str,
    ) -> AppServices:
        settings = replace(
            services.settings,
            auto_mode=mode,
            system_prompt=f"{services.settings.system_prompt}\n\n{spec.system_prompt}".strip(),
        )
        tools = self._filter_tools(services.tools, spec)
        return replace(services, settings=settings, tools=tools, agent_name=spec.name)

    def _filter_tools(self, registry: ToolRegistry, spec: AgentSpec) -> ToolRegistry:
        if spec.allowed_tools == ("*",):
            allowed = list(registry.list())
        else:
            names = set(spec.allowed_tools)
            allowed = [tool for tool in registry.list() if tool.name in names]
        if not spec.spawn_allowed:
            blocked = {"task", "run_ready_tasks"}
            if spec.name != "Plan":
                blocked.add("plan_tasks")
            allowed = [tool for tool in allowed if tool.name not in blocked]
        return ToolRegistry(allowed)

    def _apply_result(self, services: AppServices, worker_id: str, result: Any) -> None:
        worker = self._workers[worker_id]
        workspace = services.settings.workspace
        if getattr(result, "interrupt", None):
            self._set_worker_status(worker_id, TaskState.RUNNING.value)
            services.task_graph.update_metadata(
                workspace,
                worker.node_id,
                {
                    "awaiting_user": True,
                    "last_interrupt": getattr(result, "interrupt", None),
                },
            )
            return
        if getattr(result, "error", None):
            self._set_worker_status(
                worker_id,
                TaskState.FAILED.value,
                error=str(result.error),
            )
            services.task_graph.transition(workspace, worker.node_id, TaskState.FAILED.value)
            services.task_graph.update_metadata(
                workspace,
                worker.node_id,
                {"error": str(result.error), "awaiting_user": False},
            )
            self.post_task(
                channel=worker.thread_id,
                sender=worker.agent_name,
                recipient="orchestrator",
                content=f"failed:{worker.node_id}:{result.error}",
            )
            return
        summary = str(getattr(result, "final_text", "") or "").strip()
        self._set_worker_status(
            worker_id,
            TaskState.COMPLETED.value,
            summary=summary,
            error=None,
        )
        services.task_graph.transition(workspace, worker.node_id, TaskState.COMPLETED.value)
        services.task_graph.update_metadata(
            workspace,
            worker.node_id,
            {
                "result_summary": summary,
                "awaiting_user": False,
            },
        )
        self.post_task(
            channel=worker.thread_id,
            sender=worker.agent_name,
            recipient="orchestrator",
            content=f"completed:{worker.node_id}:{summary[:200]}",
        )

    def _set_worker_status(
        self,
        worker_id: str,
        status: str,
        *,
        summary: str | None = None,
        error: str | None = None,
    ) -> None:
        worker = self._workers[worker_id]
        worker.status = status
        worker.updated_at = time.time()
        if summary is not None:
            worker.summary = summary
        worker.error = error

    def _serialize_worker(self, worker: WorkerRun) -> dict[str, Any]:
        return {
            "worker_id": worker.worker_id,
            "node_id": worker.node_id,
            "title": worker.title,
            "agent_name": worker.agent_name,
            "task_type": worker.task_type,
            "status": worker.status,
            "execution_mode": worker.execution_mode,
            "thread_id": worker.thread_id,
            "write_scope": list(worker.write_scope),
        }

    # --- message log -----------------------------------------------------
    def post_task(self, channel: str, sender: str, recipient: str, content: str) -> None:
        self.repository.post_message(channel, sender, recipient, content)

    def route_messages(self, channel: str):
        return self.repository.list_messages(channel)
