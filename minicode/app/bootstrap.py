from __future__ import annotations

from minicode.core.events import EventBus
from minicode.core.types import AppServices
from minicode.features.auto import AutoModeService, PermissionMode
from minicode.features.collaboration import CollaborationRepository, CollaborationService
from minicode.features.context import ContextService
from minicode.features.cost import BudgetCaps, CostService
from minicode.features.execution import ExecutionService
from minicode.features.hooks import HookService
from minicode.features.mcp import McpServerRepository, McpService
from minicode.features.memory import MemoryRepository, MemoryService
from minicode.features.permissions import ApprovalBroker, DecisionStore, PolicyEngine
from minicode.features.profile import ProfileRepository, ProfileService
from minicode.features.sessions import SessionRepository, SessionService
from minicode.features.skills import SkillRepository, SkillService
from minicode.features.tasks import (
    BackgroundTaskRepository,
    BackgroundTaskService,
    TaskGraphRepository,
    TaskGraphService,
    TaskTrackerRepository,
    TaskTrackerService,
)
from minicode.features.tools import build_builtin_registry
from minicode.platform.config import ensure_config_scaffold, load_settings
from minicode.platform.database import DatabaseManager
from minicode.platform.migration import Migrator
from minicode.platform.paths import resolve_paths


def bootstrap_services(cwd: str) -> AppServices:
    paths = resolve_paths(cwd)
    ensure_config_scaffold(paths)
    db = DatabaseManager(paths.db_path)
    settings = load_settings(paths, cwd)
    runtime_events = EventBus()
    hooks = HookService(runtime_events)
    sessions = SessionService(SessionRepository(db), settings.workspace, settings.model)
    memory = MemoryService(MemoryRepository(db), settings.workspace)
    permissions = ApprovalBroker(PolicyEngine(), DecisionStore(db))
    task_tracker = TaskTrackerService(TaskTrackerRepository(db))
    task_graph = TaskGraphService(TaskGraphRepository(db))
    background_tasks = BackgroundTaskService(BackgroundTaskRepository(db), str(paths.logs_dir))
    skills = SkillService(SkillRepository(db, paths.skills_dir))
    mcp = McpService(McpServerRepository(db))
    collaboration = CollaborationService(CollaborationRepository(db))
    context = ContextService(model=settings.model)
    profile = ProfileService(
        ProfileRepository(
            global_path=paths.global_dir / "USER.md",
            project_path=paths.project_dir / "USER.md",
        )
    )
    cost = CostService(caps=BudgetCaps())
    execution = ExecutionService(
        allowed_roots=[paths.project_dir.parent, paths.global_dir],
    )
    auto = AutoModeService(PermissionMode.DEFAULT)
    services = AppServices(
        paths=paths,
        db=db,
        settings=settings,
        sessions=sessions,
        memory=memory,
        permissions=permissions,
        tools=None,
        task_tracker=task_tracker,
        task_graph=task_graph,
        background_tasks=background_tasks,
        mcp=mcp,
        skills=skills,
        collaboration=collaboration,
        hooks=hooks,
        runtime_events=runtime_events,
        migrator=None,
        context=context,
        profile=profile,
        cost=cost,
        execution=execution,
        auto=auto,
    )
    services.migrator = Migrator(paths, db, sessions, memory, permissions, skills, mcp, task_tracker, task_graph)
    services.migrator.migrate_once()
    services.tools = build_builtin_registry(services)
    collaboration.register_agent("main", "Primary coding agent", ["coding", "review"])
    return services
