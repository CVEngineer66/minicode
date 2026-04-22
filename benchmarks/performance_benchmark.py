"""Performance benchmark suite for MiniCode Python.

Measures performance across key areas:
1. Rendering performance (terminal UI)
2. Tool execution performance (file operations, commands)
3. Memory usage patterns
4. Context management (token estimation, compaction)
5. Agent loop throughput
"""

from __future__ import annotations

import cProfile
import io
import os
import pstats
import sys
import tempfile
import timeit
import tracemalloc
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

# Add parent directory to path to import minicode
sys.path.insert(0, str(Path(__file__).parent.parent))

# Isolate benchmark state so we don't touch the developer's real ~/.minicode.
_BENCH_HOME = Path(tempfile.mkdtemp(prefix="minicode_bench_home_"))
os.environ.setdefault("MINICODE_HOME", str(_BENCH_HOME))
os.environ.setdefault("MINICODE_PROVIDER", "openai")
os.environ.setdefault("MINICODE_MODEL", "gpt-4o-mini")
os.environ.setdefault("OPENAI_API_KEY", "bench-key")

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage  # noqa: E402
from rich.console import Console  # noqa: E402
from rich.markdown import Markdown  # noqa: E402
from rich.text import Text  # noqa: E402

from minicode.app.bootstrap import bootstrap_services  # noqa: E402
from minicode.core.types import ToolContext  # noqa: E402
from minicode.features.context.token_estimator import (  # noqa: E402
    estimate_message_tokens,
    estimate_messages_tokens,
    estimate_tokens,
)
from minicode.runtime.runner import run_turn  # noqa: E402
from minicode.ui.tui.commands import find_matching_slash_commands  # noqa: E402
from minicode.ui.tui.dispatcher import _fmt_kv  # noqa: E402
from minicode.ui.tui.parser import parse_input  # noqa: E402


@dataclass
class BenchmarkResult:
    name: str
    duration_ms: float
    ops_per_sec: float
    memory_mb: float = 0.0
    details: str = ""


def _run_timed(fn: Callable[[], Any], iterations: int) -> tuple[float, float]:
    """Return (avg_ms_per_op, ops_per_sec)."""
    total = timeit.timeit(fn, number=iterations)
    avg_ms = total * 1000 / iterations
    ops = iterations / total if total > 0 else float("inf")
    return avg_ms, ops


def format_result(result: BenchmarkResult) -> str:
    return (
        f"{result.name:<44} {result.duration_ms:>10.3f} ms  "
        f"({result.ops_per_sec:>10.1f} ops/sec)  "
        f"{result.memory_mb:>6.2f} MB  {result.details}"
    )


# ---------------------------------------------------------------------------
# Shared services / workspace
# ---------------------------------------------------------------------------

class _BenchWorkspace:
    """Owns a shared tempdir workspace + bootstrapped AppServices."""

    def __init__(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory(prefix="minicode_bench_ws_")
        self.workspace = Path(self._tmpdir.name)
        self._seed_files()
        self.services = bootstrap_services(str(self.workspace))

    def _seed_files(self) -> None:
        for i in range(50):
            (self.workspace / f"file_{i}.txt").write_text(
                f"Content line {i}\n" * 100, encoding="utf-8"
            )
        for i in range(5):
            sub = self.workspace / f"subdir_{i}"
            sub.mkdir()
            for j in range(10):
                (sub / f"sub_{j}.txt").write_text(
                    f"Sub content {j}\n" * 50, encoding="utf-8"
                )

    def tool_context(self, mode: str = "bypass") -> ToolContext:
        return ToolContext(
            thread_id="bench",
            cwd=str(self.workspace),
            mode=mode,
            services=self.services,
            emit_event=lambda *_args, **_kwargs: None,
        )

    def cleanup(self) -> None:
        try:
            self._tmpdir.cleanup()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# 1. Rendering benchmarks  (Textual/Rich + TUI helpers)
# ---------------------------------------------------------------------------


def benchmark_rendering() -> list[BenchmarkResult]:
    results: list[BenchmarkResult] = []

    # 1a. Slash command parsing
    inputs = [
        "hello world",
        "/help",
        "/resume latest",
        "/memory search langgraph",
        "/mode bypass",
    ]

    def parse_bench() -> None:
        for text in inputs:
            parse_input(text)

    avg_ms, ops = _run_timed(parse_bench, iterations=5000)
    results.append(
        BenchmarkResult(
            name="tui.parse_input (5 inputs)",
            duration_ms=avg_ms,
            ops_per_sec=ops,
            details=f"{len(inputs)} samples",
        )
    )

    # 1b. Slash command fuzzy matching
    queries = ["/", "/h", "/mem", "/mode b", "/resume", "/context"]

    def match_bench() -> None:
        for q in queries:
            find_matching_slash_commands(q)

    avg_ms, ops = _run_timed(match_bench, iterations=5000)
    results.append(
        BenchmarkResult(
            name="tui.find_matching_slash_commands",
            duration_ms=avg_ms,
            ops_per_sec=ops,
            details=f"{len(queries)} queries",
        )
    )

    # 1c. Rich Markdown rendering (EntryView uses Markdown for assistant bodies)
    console = Console(record=True, width=100, file=io.StringIO())
    md_source = (
        "# Heading\n\n"
        "Some **bold** text with `inline code` and a list:\n\n"
        "- item one with 你好世界 CJK\n"
        "- item two\n"
        "- item three\n\n"
        "```python\ndef foo(x):\n    return x + 1\n```\n"
    )

    def render_markdown() -> None:
        console.print(Markdown(md_source))

    avg_ms, ops = _run_timed(render_markdown, iterations=500)
    results.append(
        BenchmarkResult(
            name="rich.Markdown render (mixed)",
            duration_ms=avg_ms,
            ops_per_sec=ops,
            details=f"{len(md_source)} chars",
        )
    )

    # 1d. Rich Text rendering (tool call header / status lines)
    text = Text.from_markup(
        "[bold cyan]tool[/] [green]read_file[/] [dim]→[/] "
        "[magenta]Content line 0[/]"
    )

    def render_text() -> None:
        console.print(text)

    avg_ms, ops = _run_timed(render_text, iterations=5000)
    results.append(
        BenchmarkResult(
            name="rich.Text render (markup header)",
            duration_ms=avg_ms,
            ops_per_sec=ops,
        )
    )

    # 1e. Dispatcher key/value formatting (used in /cost, /context, /stats)
    payload = {
        "model": "claude-sonnet-4-20250514",
        "thread_id": "abc123-def",
        "tokens": 12_345,
        "cost_usd": 0.0137,
        "messages": 42,
        "mode": "default",
    }

    def fmt_bench() -> None:
        _fmt_kv(payload)

    avg_ms, ops = _run_timed(fmt_bench, iterations=20_000)
    results.append(
        BenchmarkResult(
            name="dispatcher._fmt_kv (6 keys)",
            duration_ms=avg_ms,
            ops_per_sec=ops,
        )
    )

    return results


# ---------------------------------------------------------------------------
# 2. Context management benchmarks  (token estimation + compaction)
# ---------------------------------------------------------------------------


def benchmark_context(workspace: _BenchWorkspace) -> list[BenchmarkResult]:
    results: list[BenchmarkResult] = []

    test_cases = [
        ("ASCII only", "Hello World " * 100),
        ("Chinese only", "你好世界" * 100),
        ("Mixed CJK/ASCII", "Hello 你好 World 世界 " * 50),
        ("Code sample", "def foo(x): return x + 1\n" * 50),
        ("Long prose", "Lorem ipsum dolor sit amet " * 200),
    ]

    for name, text in test_cases:
        tokens = estimate_tokens(text)
        avg_ms, ops = _run_timed(lambda t=text: estimate_tokens(t), iterations=5000)
        results.append(
            BenchmarkResult(
                name=f"estimate_tokens ({name})",
                duration_ms=avg_ms,
                ops_per_sec=ops,
                details=f"{len(text)} chars -> {tokens} tokens",
            )
        )

    # Message-list token sum (what maybe_compact calls every turn).
    messages = []
    for i in range(100):
        if i % 3 == 0:
            messages.append(HumanMessage(content=f"question {i} " * 30))
        elif i % 3 == 1:
            messages.append(AIMessage(content=f"answer {i} " * 30))
        else:
            messages.append(SystemMessage(content=f"note {i} " * 10))

    avg_ms, ops = _run_timed(
        lambda: estimate_messages_tokens(messages), iterations=500
    )
    total_tokens = estimate_messages_tokens(messages)
    results.append(
        BenchmarkResult(
            name="estimate_messages_tokens (100 msgs)",
            duration_ms=avg_ms,
            ops_per_sec=ops,
            details=f"{total_tokens} tokens",
        )
    )

    # ContextService.should_compact — the hot path inside maybe_compact.
    ctx = workspace.services.context

    avg_ms, ops = _run_timed(
        lambda: ctx.should_compact(messages), iterations=2000
    )
    results.append(
        BenchmarkResult(
            name="ContextService.should_compact",
            duration_ms=avg_ms,
            ops_per_sec=ops,
        )
    )

    # Compaction on an oversized context — push above the threshold.
    heavy_messages = []
    for i in range(400):
        heavy_messages.append(HumanMessage(content="Q " + "x " * 400))
        heavy_messages.append(AIMessage(content="A " + "y " * 400))

    def compact_once() -> None:
        # Rebuild every call so compaction_level doesn't advance permanently.
        from minicode.features.context.manager import ContextManager

        cm = ContextManager(model=workspace.services.settings.model)
        cm.compact(heavy_messages)

    avg_ms, ops = _run_timed(compact_once, iterations=20)
    results.append(
        BenchmarkResult(
            name="ContextManager.compact (800 msgs)",
            duration_ms=avg_ms,
            ops_per_sec=ops,
            details=f"{estimate_messages_tokens(heavy_messages)} tokens in",
        )
    )

    return results


# ---------------------------------------------------------------------------
# 3. Tool execution benchmarks
# ---------------------------------------------------------------------------


def benchmark_tools(workspace: _BenchWorkspace) -> list[BenchmarkResult]:
    results: list[BenchmarkResult] = []
    registry = workspace.services.tools
    ctx = workspace.tool_context()

    def run_tool(name: str, args: dict) -> None:
        spec = registry.get(name)
        spec.executor(dict(args), ctx)

    # 3a. list_files over ~100 files
    avg_ms, ops = _run_timed(
        lambda: run_tool("list_files", {"path": ".", "recursive": True}),
        iterations=200,
    )
    results.append(
        BenchmarkResult(
            name="tool.list_files (recursive, 100 files)",
            duration_ms=avg_ms,
            ops_per_sec=ops,
        )
    )

    # 3b. read_file
    avg_ms, ops = _run_timed(
        lambda: run_tool("read_file", {"path": "file_0.txt"}),
        iterations=500,
    )
    results.append(
        BenchmarkResult(
            name="tool.read_file (100 lines)",
            duration_ms=avg_ms,
            ops_per_sec=ops,
        )
    )

    # 3c. grep_files
    avg_ms, ops = _run_timed(
        lambda: run_tool(
            "grep_files", {"path": ".", "pattern": "Content", "max_matches": 200}
        ),
        iterations=50,
    )
    results.append(
        BenchmarkResult(
            name="tool.grep_files (100 files)",
            duration_ms=avg_ms,
            ops_per_sec=ops,
        )
    )

    # 3d. file_tree
    avg_ms, ops = _run_timed(
        lambda: run_tool("file_tree", {"path": ".", "max_depth": 3}),
        iterations=200,
    )
    results.append(
        BenchmarkResult(
            name="tool.file_tree (depth 3)",
            duration_ms=avg_ms,
            ops_per_sec=ops,
        )
    )

    # 3e. pure-CPU tool — regex_test (no I/O)
    avg_ms, ops = _run_timed(
        lambda: run_tool(
            "regex_test", {"pattern": r"\bContent\b", "text": "Content line 0\n" * 50}
        ),
        iterations=5000,
    )
    results.append(
        BenchmarkResult(
            name="tool.regex_test (pure CPU)",
            duration_ms=avg_ms,
            ops_per_sec=ops,
        )
    )

    # 3f. Full gate stack via ToolGraphAdapter for a safe read tool
    from minicode.features.tools.graph_adapter import ToolGraphAdapter

    adapter = ToolGraphAdapter(registry, workspace.services.permissions)
    call = {"id": "call_1", "name": "read_file", "args": {"path": "file_1.txt"}}

    avg_ms, ops = _run_timed(
        lambda: adapter.execute([call], ctx),
        iterations=500,
    )
    results.append(
        BenchmarkResult(
            name="ToolGraphAdapter.execute (read_file)",
            duration_ms=avg_ms,
            ops_per_sec=ops,
            details="4 gates applied",
        )
    )

    return results


# ---------------------------------------------------------------------------
# 4. Memory usage benchmarks  (tracemalloc peak allocations)
# ---------------------------------------------------------------------------


def _measure_peak(fn: Callable[[], Any]) -> float:
    tracemalloc.start()
    try:
        fn()
        _current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return peak / (1024 * 1024)


def benchmark_memory(workspace: _BenchWorkspace) -> list[BenchmarkResult]:
    results: list[BenchmarkResult] = []

    # 4a. Bootstrapping a fresh services bundle.
    def bootstrap_once() -> None:
        tmp = tempfile.mkdtemp(prefix="minicode_bench_boot_")
        try:
            bootstrap_services(tmp)
        finally:
            import shutil

            shutil.rmtree(tmp, ignore_errors=True)

    start = timeit.default_timer()
    peak = _measure_peak(bootstrap_once)
    elapsed_ms = (timeit.default_timer() - start) * 1000
    results.append(
        BenchmarkResult(
            name="bootstrap_services (cold)",
            duration_ms=elapsed_ms,
            ops_per_sec=1000 / elapsed_ms if elapsed_ms > 0 else 0.0,
            memory_mb=peak,
            details="peak RSS delta",
        )
    )

    # 4b. Building a 1000-message list + token estimation.
    def build_messages() -> None:
        msgs = []
        for i in range(1000):
            msgs.append(HumanMessage(content=f"m{i} " * 40))
            msgs.append(AIMessage(content=f"r{i} " * 40))
        estimate_messages_tokens(msgs)

    start = timeit.default_timer()
    peak = _measure_peak(build_messages)
    elapsed_ms = (timeit.default_timer() - start) * 1000
    results.append(
        BenchmarkResult(
            name="2000 messages + token scan",
            duration_ms=elapsed_ms,
            ops_per_sec=2000 / elapsed_ms * 1000 if elapsed_ms > 0 else 0.0,
            memory_mb=peak,
        )
    )

    # 4c. Compacting a large context.
    heavy = []
    for i in range(400):
        heavy.append(HumanMessage(content="u " * 400))
        heavy.append(AIMessage(content="a " * 400))

    def compact() -> None:
        from minicode.features.context.manager import ContextManager

        ContextManager(model="default").compact(heavy)

    start = timeit.default_timer()
    peak = _measure_peak(compact)
    elapsed_ms = (timeit.default_timer() - start) * 1000
    results.append(
        BenchmarkResult(
            name="ContextManager.compact (peak)",
            duration_ms=elapsed_ms,
            ops_per_sec=1000 / elapsed_ms if elapsed_ms > 0 else 0.0,
            memory_mb=peak,
        )
    )

    return results


# ---------------------------------------------------------------------------
# 5. Agent loop throughput  (run_turn with a scripted FakeChatModel)
# ---------------------------------------------------------------------------


class _FakeChatModel:
    """Reproduces the test-suite fake: scripted responses, supports bind_tools + stream."""

    def __init__(self, responses: list[AIMessage]) -> None:
        self._template = list(responses)
        self._queue: list[AIMessage] = []
        self.bound_tools = None

    def bind_tools(self, tools):  # noqa: ANN001
        self.bound_tools = tools
        return self

    def _refill(self) -> None:
        self._queue = list(self._template)

    def invoke(self, messages):  # noqa: ANN001
        if not self._queue:
            self._refill()
        return self._queue.pop(0)

    def stream(self, messages):  # noqa: ANN001
        yield self.invoke(messages)


def benchmark_agent_loop(workspace: _BenchWorkspace) -> list[BenchmarkResult]:
    results: list[BenchmarkResult] = []
    services = workspace.services

    # 5a. Single-shot turn: model responds with a final answer, no tool calls.
    single_model = _FakeChatModel([AIMessage(content="<final> done")])

    def single_shot() -> None:
        run_turn(
            services=services,
            prompt="ping",
            chat_model=single_model,
            persist=False,
        )

    avg_ms, ops = _run_timed(single_shot, iterations=20)
    results.append(
        BenchmarkResult(
            name="run_turn (final answer, no tools)",
            duration_ms=avg_ms,
            ops_per_sec=ops,
            details="START -> assembly -> model -> finalize",
        )
    )

    # 5b. Progress-then-final: classify_output must recycle once.
    progress_model = _FakeChatModel(
        [
            AIMessage(content="<progress> thinking..."),
            AIMessage(content="<final> done"),
        ]
    )

    def progress_cycle() -> None:
        run_turn(
            services=services,
            prompt="ping",
            chat_model=progress_model,
            persist=False,
            max_steps=6,
        )

    avg_ms, ops = _run_timed(progress_cycle, iterations=20)
    results.append(
        BenchmarkResult(
            name="run_turn (progress + final)",
            duration_ms=avg_ms,
            ops_per_sec=ops,
            details="progress_continue loop once",
        )
    )

    return results


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def run_all_benchmarks() -> list[BenchmarkResult]:
    print("=" * 90)
    print("MiniCode Python Performance Benchmark")
    print("=" * 90)
    print()

    workspace = _BenchWorkspace()
    all_results: list[BenchmarkResult] = []

    groups: list[tuple[str, Callable[[], list[BenchmarkResult]]]] = [
        ("Rendering (TUI + Rich)", benchmark_rendering),
        ("Context Management", lambda: benchmark_context(workspace)),
        ("Tool Execution", lambda: benchmark_tools(workspace)),
        ("Memory Usage", lambda: benchmark_memory(workspace)),
        ("Agent Loop Throughput", lambda: benchmark_agent_loop(workspace)),
    ]

    try:
        for name, func in groups:
            print(f"[*] Running {name}...")
            try:
                results = func()
                all_results.extend(results)
                print(f"    -> {len(results)} benchmarks completed\n")
            except Exception as exc:
                print(f"    !! Failed: {exc!r}\n")
    finally:
        workspace.cleanup()

    return all_results


def print_results(results: list[BenchmarkResult]) -> None:
    print("=" * 90)
    print("Benchmark Results")
    print("=" * 90)
    print(
        f"{'Test':<44} {'Duration':>14} {'Ops/sec':>14} {'Memory':>8} {'Details'}"
    )
    print("-" * 90)
    for r in results:
        print(format_result(r))
    print("-" * 90)
    print(f"\nTotal benchmarks: {len(results)}")


def profile_key_functions() -> None:
    print("\n" + "=" * 90)
    print("Profiling Key Functions")
    print("=" * 90)

    profiler = cProfile.Profile()
    profiler.enable()
    large_text = "Hello 你好 " * 10_000
    for _ in range(1000):
        estimate_tokens(large_text)
    messages = [
        HumanMessage(content="Q " + "w " * 80) if i % 2 == 0 else AIMessage(content="A " + "z " * 80)
        for i in range(200)
    ]
    for _ in range(200):
        estimate_messages_tokens(messages)
    profiler.disable()

    buf = io.StringIO()
    pstats.Stats(profiler, stream=buf).sort_stats("cumulative").print_stats(20)
    print(buf.getvalue())


if __name__ == "__main__":
    results = run_all_benchmarks()
    print_results(results)
    profile_key_functions()

    output_file = Path(__file__).parent / "benchmark_results.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("MiniCode Python Benchmark Results\n")
        f.write("=" * 90 + "\n\n")
        f.write(
            f"{'Test':<44} {'Duration':>14} {'Ops/sec':>14} {'Memory':>8} {'Details'}\n"
        )
        f.write("-" * 90 + "\n")
        for r in results:
            f.write(format_result(r) + "\n")

    print(f"\nResults saved to {output_file}")
