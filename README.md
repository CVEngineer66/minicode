# MiniCode

一个基于 **LangGraph** 的终端编码助手。它不是“一个 TUI 外壳 + 几个散落脚本”，而是把 **TUI、headless CLI、HTTP gateway、cron** 四个入口都收敛到同一套运行时里：同一张状态图、同一套工具门控、同一份 SQLite 持久化、同一条事件流。

这份 README 重点回答两个问题：

- 这个项目现在应该怎么用。
- 这套代码为什么按现在的方式组织。

不展开 `tests/`。

---

## 1. 核心设计

这几条是当前实现的主轴：

1. **一个 turn 一张图。** 所有模型调用都走 `minicode/runtime/runner.py::run_turn`，没有入口侧旁路。
2. **Bootstrap 是唯一组合根。** `minicode/app/bootstrap.py::bootstrap_services(cwd)` 负责装配配置、数据库、服务与工具注册表。
3. **工具执行有固定四关。** 顺序是：参数校验 -> 执行边界 -> auto 风险分类 -> 权限审批。
4. **状态和业务数据共用一个 SQLite。** LangGraph checkpoint 和业务表一起落到 `~/.minicode/runtime.sqlite`。
5. **Prompt 显式拆成静态前缀与动态尾部。** 中间插入 `SYSTEM_PROMPT_DYNAMIC_BOUNDARY`，便于模型供应商命中前缀缓存。
6. **外部失败尽量局部化。** hooks 超时、MCP 异常、单个 skill 解析失败都会被局部吞掉，不让整个 turn 连带崩掉。

---

## 2. 安装与首次启动

```bash
pip install -e .
```

运行要求：

- Python `>=3.10`
- 运行期依赖见 `pyproject.toml`
- 主要依赖：`langgraph`、`langgraph-checkpoint-sqlite`、`langchain-core`、`langchain-openai`、`langchain-anthropic`、`textual`

首次启动时会自动创建：

- `~/.minicode/config.json`
- `~/.minicode/config.example.json`
- `~/.minicode/runtime.sqlite`
- `~/.minicode/skills/`
- `~/.minicode/exports/`
- `~/.minicode/logs/`
- `<workspace>/.minicode/`
- `<workspace>/.minicode-local/`

---

## 3. 快速开始

### 3.1 最小配置

当前配置不是旧版那种扁平 `provider/model/api_key` 结构，而是 **provider catalog + current_model**。最小可用配置示例：

```json
{
  "providers": {
    "openai": {
      "kind": "openai",
      "api_key": "sk-REPLACE-ME",
      "models": ["gpt-4o-mini", "gpt-4o"]
    },
    "anthropic": {
      "kind": "anthropic",
      "api_key": "sk-ant-REPLACE-ME",
      "models": ["claude-sonnet-4-20250514"]
    }
  },
  "current_model": "openai:gpt-4o-mini",
  "default_mode": "default",
  "system_prompt": "You are MiniCode, a coding assistant focused on safe, concrete execution.",
  "thinking": "auto",
  "thinking_budget_tokens": 2048
}
```

说明：

- `providers` 的 key 是 provider 名称，值里至少要有 `kind`、`api_key`、`models`。
- `kind` 只认 `"openai"` 或 `"anthropic"`，决定底层 SDK 分支。
- `current_model` 形如 `provider:model`。
- 如果 `current_model` 缺失或失效，会回退到 **第一个 provider 的第一个 model**。
- `/model` 选择器会把当前选择写回 `config.json` 的 `current_model`。
- `/mode` 只影响当前进程；启动默认 mode 来自 `default_mode`。

### 3.2 常用命令

```bash
minicode
minicode -c
minicode --resume 1a2b3c
minicode sessions list
minicode mcp list
minicode skills list

minicode-headless "总结当前仓库结构"
minicode-headless --jsonl --mode plan "只读分析当前架构"
minicode-headless --list-sessions
minicode-headless --resume latest "继续上一次任务"

minicode-gateway --port 7681
minicode-cron --run-once
```

---

## 4. 入口命令

`pyproject.toml` 暴露了四个控制台脚本：

| 命令 | 入口文件 | 用途 |
| --- | --- | --- |
| `minicode` | `minicode/app/main.py` | Textual TUI 主入口 |
| `minicode-headless` | `minicode/app/headless.py` | 单次 turn / 恢复审批 / JSONL 流式输出 |
| `minicode-gateway` | `minicode/app/gateway_main.py` | 本地 HTTP gateway |
| `minicode-cron` | `minicode/app/cron_main.py` | 轮询式本地定时执行 |

### 4.1 `minicode`

支持：

- `-c`, `--continue`：恢复当前工作区最近一次会话
- `--resume <thread_id>`：按完整 id 或前缀恢复会话
- `sessions list`
- `mcp list`
- `mcp add <name> <command> [args...]`
- `skills list`
- `skills install <path>`

### 4.2 `minicode-headless`

核心参数：

- 位置参数 `prompt`
- `--resume <thread_id|prefix|latest>`
- `--list-sessions`
- `--decision-json <json>`
- `--mode {default,auto,bypass,plan}`
- `--jsonl`
- `--max-steps <n>`
- `--cwd <path>`

行为说明：

- 普通模式下输出最终文本，或在审批中断时输出 `__interrupt__` JSON。
- `--jsonl` 会把事件流和最终结果都逐行输出，适合脚本接入。

### 4.3 `minicode-gateway`

默认监听 `127.0.0.1:7681`，支持：

- `GET /health`
- `GET /sessions`
- `POST /turn`
- `POST /resume`

### 4.4 `minicode-cron`

支持：

- `--config <path>`：默认 `<global_dir>/cron.json`
- `--run-once`
- `--tick-seconds`
- `--max-parallel`
- `--cwd`
- `--log-level`
- `--structured-logs`

---

## 5. 配置与环境变量

当前实现里，**只有目录相关环境变量**在配置加载阶段生效：

| 环境变量 | 用途 |
| --- | --- |
| `MINICODE_HOME` | 全局目录，默认 `~/.minicode` |
| `MINICODE_LEGACY_HOME` | 旧版 `~/.mini-code` 迁移来源目录 |

注意：

- `provider/model/mode/thinking` 不再从环境变量覆盖。
- 当前生效模型来自 `config.json` 里的 `providers + current_model`。
- 启动默认 mode 来自 `default_mode`。
- `config.example.json` 会在首次启动时生成，但不会覆盖已有文件。

---

## 6. 分层结构

依赖方向保持单向：

```text
app -> ui/runtime -> features -> platform/core
```

目录职责：

```text
minicode/app/         四个入口 + bootstrap
minicode/ui/tui/      Textual TUI、slash 命令、审批呈现、picker/modal
minicode/runtime/     LangGraph runner、prompt pipeline、retry、model factory
minicode/features/    领域服务（sessions/memory/tools/mcp/context/...）
minicode/gateway/     HTTP gateway
minicode/cron/        轻量轮询 scheduler
minicode/platform/    config/database/logging/http/process/paths/migration
minicode/core/        GraphEvent、ToolSpec、ToolResult、AppServices 等共享类型
```

### 6.1 Bootstrap 装配内容

`minicode/app/bootstrap.py` 当前装配：

- `DatabaseManager`
- `Settings`
- `EventBus`
- `HookService`
- `SessionService`
- `MemoryService`
- `ApprovalBroker`
- `TaskTrackerService`
- `TaskGraphService`
- `BackgroundTaskService`
- `SkillService`
- `McpService`
- `CollaborationService`
- `ContextService`
- `ProfileService`
- `CostService`
- `ExecutionService`
- `AutoModeService`
- `Migrator`
- `ToolRegistry`（最后构建，允许工具依赖前面所有服务）

原则：

- 新服务只在 bootstrap 里 new。
- 其它层只通过 `AppServices` 拿依赖。
- 入口层只负责“怎么交互”，不负责“有哪些能力”。

---

## 7. Turn 运行图

`minicode/runtime/runner.py::_build_graph` 当前图节点如下：

```text
START
  -> prompt_assembly
  -> maybe_compact
  -> model_call
  -> classify_output
     -> execute_tools
     -> progress_continue
     -> memory_update
     -> session_finalize
  -> END
```

节点职责：

| 节点 | 作用 |
| --- | --- |
| `prompt_assembly` | 组装静态 prompt、动态 prompt、上下文信息 |
| `maybe_compact` | 接近上下文上限时压缩历史消息 |
| `model_call` | 调模型，流式累积 token，并记录成本 |
| `classify_output` | 判断输出是继续、调工具、收尾还是需要 progress nudge |
| `progress_continue` | 注入继续提示，再回到 `model_call` |
| `execute_tools` | 进入 `ToolGraphAdapter`，跑一批工具 |
| `memory_update` | 写入长期记忆 |
| `session_finalize` | 触发结束事件，收尾 |

运行期边界：

- `run_turn` 开始前会重跑一次迁移检查。
- 输入 prompt 会先经过 prompt injection 检测。
- turn 结束后会做输出安全检测和成本结算。
- `persist=False` 时不会写 checkpoint，子任务可复用同图但不污染主会话。

---

## 8. 工具执行四关

`minicode/features/tools/graph_adapter.py::ToolGraphAdapter` 的顺序是固定的：

```text
1. ToolSpec.validator(args)
2. ExecutionService.check_path_access / check_command
3. AutoModeService.assess(tool, args)
4. PolicyEngine + ApprovalBroker
5. spec.executor(validated, context)
```

含义：

1. **参数校验**：把模型给的参数规范化成 executor 可直接消费的结构。
2. **执行边界**：检查路径逃逸、敏感文件、命令风险等级。
3. **auto 风险分类**：在 `auto` / `plan` 模式下决定 approve / prompt / block。
4. **权限审批**：命中缓存则直接放行，否则通过 LangGraph interrupt 向前端请求决策。

并发策略：

- `plan` 模式下一律串行。
- 只要某个工具是 `interactive` / `long_running` / `task`，整批串行。
- 只要某个工具需要审批，整批串行。
- 否则最多 `max_workers=4` 并发。

---

## 9. Mode 与权限

当前支持四种 mode：

| Mode | 语义 |
| --- | --- |
| `default` | 风险工具默认走审批 |
| `auto` | 低风险工具自动放行，中高风险继续提示或阻断 |
| `bypass` | 跳过审批，直接执行 |
| `plan` | 只读导向；执行类工具会被限制或阻断 |

切换方式：

- TUI：`/mode`
- Headless：`--mode`
- Gateway：`POST /turn` 里传 `mode`
- 启动默认值：`config.json` 里的 `default_mode`

权限层当前明确持久化的是：

- `allow_always`
- `deny_always`

turn 内缓存由 `ApprovalBroker` 维护，turn 结束会清空。

执行层安全边界由 `ExecutionService` 提供：

- 路径必须落在允许根目录内
- 敏感文件会升级为需要审批
- 命令按风险级别分类
- 高风险命令可走临时 worktree 隔离执行

---

## 10. Prompt Pipeline

`minicode/runtime/prompts.py` 现在不是旧版的“几段字符串拼接”，而是一个带 TTL 缓存的 `PromptPipeline`。

静态前缀：

```text
role
system
doing_tasks
actions
tools
turns
tone_and_style
```

动态尾部：

```text
mode
env
profile
skills
memory
language
session_guidance
closer
```

两段之间插入：

```text
__SYSTEM_PROMPT_DYNAMIC_BOUNDARY__
```

当前动态段特点：

- `mode`：60s TTL
- `env`：300s TTL
- `profile`：300s TTL
- `skills`：120s TTL
- `memory`：60s TTL，且只有 `latest_user_query` 非空时才注入
- `language` / `closer`：常量段

补充：

- `read_file_cached(path, ttl)` 是 mtime-aware 文件缓存，用于 `USER.md` / `SKILL.md` 这类读取。
- `assemble_system_prompt_split()` 会把静态和动态部分拆开，供 Anthropic 前缀缓存路径使用。

---

## 11. 关键子系统

这里只列当前代码里最重要、最稳定的那些边界。

| 子系统 | 位置 | 当前职责 |
| --- | --- | --- |
| 会话 | `features/sessions/` | 线程创建、恢复、branch、compact、archive |
| 长期记忆 | `features/memory/` | SQLite 持久化、TF-IDF 召回、prompt block 构建 |
| 上下文 | `features/context/` | token 估算、接近上限时压缩、子代理 sandbox |
| 权限 | `features/permissions/` | 审批请求构造、决策缓存、危险命令识别 |
| 执行安全 | `features/execution/` | 路径白名单、敏感文件升级、命令风险评估、隔离执行 |
| Auto mode | `features/auto/` | 工具风险分类、prompt injection 检测、输出安全检测 |
| Hooks | `features/hooks/` | `pre_turn` / `post_turn` / `pre_tool` / `post_tool` / `on_error`，单 handler 默认 5 秒超时 |
| MCP | `features/mcp/` | server 注册、连接池、工具列表 60 秒缓存、软失败返回 |
| Skills | `features/skills/` | 已安装 skill 目录与 SQLite catalog，同步写入 `~/.minicode/skills/` |
| Cost | `features/cost/` | token/cost ledger、预算上限、按模型计费 |
| Profile | `features/profile/` | 合并全局与项目级 `USER.md` |
| Tasks | `features/tasks/` | 简单任务列表、DAG 任务图、后台任务记账 |
| Collaboration | `features/collaboration/` | agent card 和消息通道的存储层 |

---

## 12. TUI 结构

`minicode/ui/tui/app.py` 当前是一个 Textual 应用，核心组成是：

- `Header`
- `TranscriptScroll`
- 底部 `ComposerInput`
- `SlashSuggest`
- `Footer`

UI 交互相关组件：

- `ApprovalEntry`：审批请求的内联展示与决策事件
- `PickerScreen`：`/mode`、`/model`、`/resume` 选择器
- `CommandOutputScreen`：slash 命令输出覆盖层

Slash 命令定义在 `minicode/ui/tui/commands.py`，当前共 24 条，覆盖：

- 会话：`/sessions`、`/resume`、`/branch`、`/compact`
- 运行模式：`/mode`、`/model`
- 观察类：`/memory`、`/profile`、`/cost`、`/context`
- 管理类：`/tasks`、`/agents`、`/skills`、`/mcp`、`/hooks`、`/permissions`
- UI：`/help`、`/clear`、`/quit`

---

## 13. 存储布局

路径解析见 `minicode/platform/paths.py`：

```text
$MINICODE_HOME (默认 ~/.minicode)
├── runtime.sqlite
├── config.json
├── config.example.json
├── mcp_servers.json        (兼容/迁移路径，当前主存储仍是 SQLite)
├── skills/
├── exports/
└── logs/

<workspace>/.minicode/
<workspace>/.minicode-local/
```

`runtime.sqlite` 里当前主要表包括：

- `session_meta`
- `memory_entries`
- `permission_decisions`
- `task_items`
- `task_graph_nodes`
- `task_graph_edges`
- `background_tasks`
- `mcp_servers`
- `skills`
- `collaboration_agents`
- `collaboration_messages`
- `migration_ledger`
- 以及 LangGraph `SqliteSaver` 自己的 checkpoint 表

迁移逻辑在 `minicode/platform/migration.py`：

- 旧目录默认是 `~/.mini-code`
- 会尝试迁移 settings、MCP、skills、permissions、memory、sessions、tasks
- 迁移通过 `migration_ledger` 做幂等控制

---

## 14. HTTP Gateway

`minicode/gateway/server.py` 当前暴露的是一个极简 JSON API：

### `GET /health`

返回：

```json
{
  "status": "ok",
  "workspace": "...",
  "provider": "...",
  "model": "...",
  "mode": "..."
}
```

### `GET /sessions`

返回当前工作区的会话列表。

### `POST /turn`

请求体：

```json
{
  "prompt": "帮我检查当前目录",
  "thread_id": "optional",
  "mode": "optional",
  "max_steps": 40
}
```

### `POST /resume`

请求体：

```json
{
  "thread_id": "abc123",
  "decision": {
    "decision": "allow_once"
  }
}
```

边界：

- body 上限 1 MiB
- 统一返回 JSON，不吐 HTML traceback
- 默认只监听 `127.0.0.1`
- **没有鉴权，也没有 CORS 方案，不要直接暴露公网**

---

## 15. Cron

`minicode/cron/runner.py` 是一个轻量轮询调度器，不支持标准 crontab 语法。

配置文件示例：

```json
{
  "jobs": [
    {
      "name": "daily-summary",
      "schedule": "every:4h",
      "prompt": "总结当前项目变更",
      "enabled": true
    }
  ]
}
```

支持的 schedule：

- `every:30m`
- `every:4h`
- `every:300s`
- 纯数字秒数，如 `300`

当前边界：

- 内部执行固定走 `mode="auto"`
- `max_parallel` 默认 1
- handler 异常记到 job 上，不会把循环打停
- 如果需要复杂调度，交给系统 cron 去调 `minicode-headless`

---

## 16. 内置工具

`minicode/features/tools/builtin.py` 当前注册了 **53 个内置工具**，并在构建注册表时动态追加 MCP 工具。

注册方式：

- 使用 `@tools.register(...)`
- 从函数签名和 docstring 推导 JSON Schema
- 统一收敛成 `ToolSpec`

主要类别：

| 类别 | 代表工具 |
| --- | --- |
| 文件读取 | `list_files`, `file_tree`, `grep_files`, `read_file` |
| 文件写入 | `write_file`, `modify_file`, `edit_file`, `patch_file` |
| 批量文件操作 | `batch_copy`, `batch_move`, `batch_delete` |
| Shell / 诊断 | `run_command`, `git`, `code_review`, `diff_viewer`, `test_runner` |
| 网络 | `web_fetch`, `web_search`, `http_request` |
| 数据格式 | `json_format`, `json_parse`, `csv_parse`, `csv_create` |
| 文本处理 | `regex_test`, `regex_replace`, `text_sort`, `text_dedupe`, `text_join` |
| 压缩归档 | `gzip_*`, `tar_*`, `zip_*` |
| 编码与摘要 | `base64_*`, `url_*`, `hash`, `hmac` |
| 时间与标识 | `current_time`, `timestamp`, `uuid_generate`, `random_string` |
| 任务协作 | `todo_write`, `task`, `ask_user` |
| Skills / MCP | `load_skill` + 动态 `mcp.<server>.<tool>` |

---

## 17. 代码索引

如果你要改某一块，优先从这些文件进入：

```text
minicode/app/
  bootstrap.py            组合根，装配 AppServices
  main.py                 TUI CLI 入口
  headless.py             Headless CLI 入口
  gateway_main.py         Gateway 启动入口
  cron_main.py            Cron 启动入口

minicode/runtime/
  runner.py               run_turn + LangGraph 编排
  prompts.py              PromptPipeline + 动静态 prompt 拼装
  model_factory.py        OpenAI / Anthropic 模型构造
  retry.py                API 重试分类与退避
  graph_state.py          GraphState 定义

minicode/features/tools/
  builtin.py              53 个内置工具 + MCP 动态工具
  graph_adapter.py        工具执行四关 + 并发策略
  decorator.py            register 装饰器与 schema 推导
  registry.py             ToolRegistry

minicode/features/permissions/
  service.py              PolicyEngine + ApprovalBroker
  repository.py           决策存储
  graph_adapter.py        LangGraph interrupt 审批桥

minicode/features/models/
  catalog.py              provider:model catalog，当前模型选择源

minicode/features/sessions/
  service.py              ensure/resume/branch/compact/archive

minicode/features/context/
  service.py              token 统计、压缩、子代理 sandbox

minicode/features/memory/
  service.py              长期记忆与 prompt block

minicode/features/mcp/
  service.py              server 管理、工具缓存、调用封装

minicode/features/skills/
  service.py              skill 安装、加载、目录镜像

minicode/features/tasks/
  services.py             task tracker / task graph / background tasks
  repository.py           task 持久化
  types.py                task 状态与错误类型

minicode/ui/tui/
  app.py                  MiniCodeApp
  dispatcher.py           slash 命令分发
  commands.py             slash 命令声明
  screens.py              picker / command output modal
  approval_entry.py       审批 UI 条目

minicode/platform/
  config.py               config.json 读取、保存当前模型、首次脚手架
  database.py             SQLite schema + checkpointer
  paths.py                全局/项目路径解析
  migration.py            旧版数据迁移
  logging.py              gateway/cron 日志初始化
```

---

## 18. 当前边界与非目标

这些限制是有意保留的：

- **不做多模型 fan-out。** 一个 turn 只跑一个模型。
- **不支持标准 crontab。** 复杂调度交给系统 cron。
- **Gateway 不做鉴权和 CORS。** 默认假设只在本机或受控网络使用。
- **不做 SSE / WebSocket gateway。** 流式主要给 TUI 和 headless `--jsonl`。
- **默认存储只有 SQLite。** 暂不引入外部数据库。

---

## 19. 一句话总结

MiniCode 当前的重点不是“功能拼得多”，而是把 **入口、运行图、工具门控、上下文、持久化** 收敛成一套统一机制。你要改入口、加工具、调 prompt、接 MCP、改审批，都应该先回到这几个文件：`app/bootstrap.py`、`runtime/runner.py`、`runtime/prompts.py`、`features/tools/graph_adapter.py`、`platform/database.py`。
