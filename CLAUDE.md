# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Package layout

The repo has a single active package: `minicode/`.

| Package | Status |
| --- | --- |
| `minicode/` | **Active** — LangGraph-first, layered (`core`/`platform`/`runtime`/`features`/`ui`/`app`) |

## Commands

```bash
# install (editable, with all LangGraph/provider deps)
pip install -e .
pip install -e ".[dev]"   # adds pytest

# run the TUI (recommended entry point)
minicode-next

# headless single-turn; --jsonl streams graph events + result
minicode-next-headless --prompt "hello"
minicode-next-headless --jsonl --prompt "hello" --mode plan

# HTTP gateway (binds 127.0.0.1:7681, no auth — localhost only)
minicode-next-gateway --port 7681

# scheduled jobs from a config file
minicode-next-cron --run-once --config cron.json

# tests
pytest tests/                                 # full suite
pytest tests/test_minicode_next.py::test_name # single test
pytest -k "context and not cron"              # pattern filter
```

## Architecture — the LangGraph turn

`minicode/` is organized in five layers; dependencies flow downward only (`app → ui/runtime → features → platform/core`).

```
app/         CLI entry points (main, headless, gateway_main, cron_main) + bootstrap
ui/tui/      Textual-based TUI (app, modal screens for approval/picker, slash command dropdown, dispatcher)
runtime/     LangGraph StateGraph + system-prompt pipeline + retry + model factory
features/    Domain services (tools, memory, permissions, context, tasks, mcp, ...)
platform/    SQLite, paths, config, logging, http, process, migration
core/        Dataclasses + Protocols used by every layer (AppServices, ToolSpec, ...)
gateway/     Stdlib BaseHTTPRequestHandler wrapping run_turn
cron/        Tiny polling scheduler (every:<N>s|m|h|d)
```

**Bootstrap is the composition root.** `minicode/app/bootstrap.py::bootstrap_services(cwd)` builds every service, runs `Migrator.migrate_once()`, and returns an `AppServices` dataclass. **Every entry point (TUI, headless, gateway, cron) calls this function.** New services should be wired in here and added to `core/types.py::AppServices`.

**A turn is a compiled LangGraph.** `runtime/runner.py::run_turn` is the single path — TUI, headless, gateway, cron, and the `task` builtin tool (for sub-agents) all funnel through it. The graph nodes (defined inside `_build_graph`) are:

```
START → prompt_assembly → maybe_compact → model_call → classify_output
                                                          ├─ execute_tools → model_call
                                                          ├─ progress_continue → model_call
                                                          └─ memory_update → session_finalize → END
```

- `prompt_assembly` uses `runtime/prompts.py::assemble_system_prompt` — a paragraph-level pipeline with TTL cache and a `SYSTEM_PROMPT_DYNAMIC_BOUNDARY` sentinel so providers can cache the static prefix.
- `maybe_compact` delegates to `ContextService` (token estimate → summarize old turns).
- `model_call` wraps `retry_with_backoff` (`runtime/retry.py`) around `bound_model.stream(...)` so tokens are streamed as they arrive; usage is recorded into `CostService`.
- `classify_output` reads the `<progress>…</progress>` marker to decide whether the model wants another loop iteration versus a final answer.
- `execute_tools` goes through `ToolGraphAdapter` (see below). Sub-agent tools re-enter `run_turn` with a fresh `thread_id` suffix and `persist=False`.
- Persistence uses LangGraph's `SqliteSaver` checkpointer — `DatabaseManager.checkpointer()` opens the same SQLite file used for app state.

**Tool execution has four gates**, enforced in `features/tools/graph_adapter.py::ToolGraphAdapter` in this order — do not reorder or bypass:
1. Schema validation (`ToolSpec.validator`)
2. `ExecutionService` boundary (deny / require-approval based on path roots + command risk)
3. `AutoModeService` classification (per-mode allow / prompt / block)
4. `PolicyEngine` approval flow → `ApprovalBroker` → TUI interrupt

Whether a batch runs concurrently or serially depends on `ToolCapability.requires_serial_execution` and whether any call would trigger an approval request. `mode="plan"` forces serial.

**Tool catalog.** `features/tools/builtin.py` is a single ~580-line module of ~50+ builtin tools (file I/O, search, shell, web, archives, hashing, task graph, MCP, subagent dispatch). Each tool is a `ToolSpec` with explicit `ToolCapability` (network/shell/writes_files/…) and `PermissionPolicy`. Events emitted during execution (`assistant_token`, `tool_start`, `tool_result`, `context_compacted`, `turn_cost`, `session_finalized`, …) take two paths: graph-level events go through LangGraph's `get_stream_writer()` in custom stream mode; **tool events bypass the writer** because `execute_tools` runs tools in a `ThreadPoolExecutor` and worker threads don't have LangGraph's runnable context — they emit directly through the caller's `event_sink` instead.

**State lives in SQLite.** `~/.minicode-next/runtime.sqlite` (override via `MINICODE_NEXT_HOME`) holds session metadata, memory entries, permission decisions, tasks, task graph, background tasks, MCP servers, skills, and collaboration state. LangGraph checkpoints live in the same file. `platform/database.py::DatabaseManager._setup` is the schema source of truth — schema changes go there and need a corresponding migration in `platform/migration.py`.

## Config & settings

- Runtime config: `<global_dir>/config.json` (default `~/.minicode-next/config.json`).
- Env overrides: `MINICODE_NEXT_HOME`, `MINICODE_NEXT_PROVIDER`, `MINICODE_NEXT_MODEL`, `MINICODE_NEXT_BASE_URL`, `MINICODE_NEXT_MODE`.
- `platform/config.py::load_settings` resolves in this order: env var → `config.json` value → hard-coded default. `api_key` comes from `os.environ[api_key_env]` (e.g. `OPENAI_API_KEY`), with `config.json.api_key` as a fallback.

## Testing conventions

- `tests/test_minicode_next_*.py` — the active suite. Each maps to one feature module (`_memory`, `_permissions`, `_context`, `_tasks`, `_mcp`, `_hooks`, `_sessions`, `_tui`, `_tui_dispatcher`, `_retry`, `_cost`, `_auto`, `_execution`, `_skills`, `_profile`, `_prompts`, `_platform_cron`).
- `tests/test_minicode_next.py` exercises the graph end-to-end with a `FakeChatModel` that scripts `AIMessage` / tool-call responses — copy that pattern when writing new runtime tests instead of hitting a real model.
- Tests monkey-patch `MINICODE_NEXT_HOME` to a tmp path so they don't touch the developer's real state. Follow the same pattern for any test that calls `bootstrap_services`.
