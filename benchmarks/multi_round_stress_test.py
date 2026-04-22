"""Multi-round stress test for MiniCode Python performance.

Runs five rounds each against the five strategic areas so regressions that
only surface after warmup (JIT-like effects in regex/tokeniser caches, SQLite
page cache warm-up, LangGraph compiled-graph reuse) still show up.
"""

from __future__ import annotations

import os
import sys
import tempfile
import timeit
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# Isolate benchmark state before any minicode import resolves paths.
_BENCH_HOME = Path(tempfile.mkdtemp(prefix="minicode_stress_home_"))
os.environ.setdefault("MINICODE_HOME", str(_BENCH_HOME))
os.environ.setdefault("MINICODE_PROVIDER", "openai")
os.environ.setdefault("MINICODE_MODEL", "gpt-4o-mini")
os.environ.setdefault("OPENAI_API_KEY", "stress-key")

from langchain_core.messages import AIMessage, HumanMessage  # noqa: E402

from minicode.app.bootstrap import bootstrap_services  # noqa: E402
from minicode.core.types import ToolContext  # noqa: E402
from minicode.features.context.token_estimator import (  # noqa: E402
    estimate_messages_tokens,
    estimate_tokens,
)
from minicode.runtime.runner import run_turn  # noqa: E402
from minicode.ui.tui.commands import find_matching_slash_commands  # noqa: E402

ROUNDS = 5


# ---------------------------------------------------------------------------
# Shared workspace
# ---------------------------------------------------------------------------


def _build_workspace() -> tuple[Path, object, "tempfile.TemporaryDirectory[str]"]:
    tmpdir = tempfile.TemporaryDirectory(prefix="minicode_stress_ws_")
    workspace = Path(tmpdir.name)
    for i in range(20):
        (workspace / f"file_{i}.txt").write_text(
            f"Content line {i}\n" * 50, encoding="utf-8"
        )
    services = bootstrap_services(str(workspace))
    return workspace, services, tmpdir


# ---------------------------------------------------------------------------
# FakeChatModel for agent-loop rounds
# ---------------------------------------------------------------------------


class _FakeChatModel:
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


# ---------------------------------------------------------------------------
# 1. Token estimation (context management)
# ---------------------------------------------------------------------------


def test_token_estimation() -> list[tuple[int, list[tuple[str, float, int]]]]:
    cases = [
        ("ASCII", "Hello World " * 100),
        ("Chinese", "你好世界" * 100),
        ("Mixed", "Hello 你好 World 世界 " * 50),
    ]
    all_rounds = []
    for round_num in range(1, ROUNDS + 1):
        round_results: list[tuple[str, float, int]] = []
        for name, text in cases:
            elapsed = timeit.timeit(lambda t=text: estimate_tokens(t), number=10_000)
            ops = 10_000 / elapsed if elapsed > 0 else float("inf")
            round_results.append((name, ops, estimate_tokens(text)))
        all_rounds.append((round_num, round_results))
    return all_rounds


# ---------------------------------------------------------------------------
# 2. TUI / rendering helpers
# ---------------------------------------------------------------------------


def test_slash_command_matching() -> list[tuple[int, list[tuple[str, float]]]]:
    queries = ["/", "/h", "/mem search auth", "/mode b", "/resume latest"]
    all_rounds = []
    for round_num in range(1, ROUNDS + 1):
        round_results: list[tuple[str, float]] = []
        for q in queries:
            elapsed = timeit.timeit(
                lambda text=q: find_matching_slash_commands(text), number=20_000
            )
            ops = 20_000 / elapsed if elapsed > 0 else float("inf")
            round_results.append((q, ops))
        all_rounds.append((round_num, round_results))
    return all_rounds


# ---------------------------------------------------------------------------
# 3. Tool execution
# ---------------------------------------------------------------------------


def test_tool_execution(
    workspace: Path, services: object
) -> list[tuple[int, list[tuple[str, float]]]]:
    ctx = ToolContext(
        thread_id="stress",
        cwd=str(workspace),
        mode="bypass",
        services=services,
        emit_event=lambda *_a, **_kw: None,
    )
    registry = services.tools  # type: ignore[attr-defined]

    list_spec = registry.get("list_files")
    read_spec = registry.get("read_file")
    grep_spec = registry.get("grep_files")

    all_rounds = []
    for round_num in range(1, ROUNDS + 1):
        round_results: list[tuple[str, float]] = []

        elapsed = timeit.timeit(
            lambda: list_spec.executor({"path": ".", "recursive": False}, ctx),
            number=500,
        )
        round_results.append(("list_files", 500 / elapsed if elapsed else 0.0))

        elapsed = timeit.timeit(
            lambda: read_spec.executor({"path": "file_0.txt"}, ctx),
            number=1000,
        )
        round_results.append(("read_file", 1000 / elapsed if elapsed else 0.0))

        elapsed = timeit.timeit(
            lambda: grep_spec.executor(
                {"path": ".", "pattern": "Content", "max_matches": 100}, ctx
            ),
            number=50,
        )
        round_results.append(("grep_files", 50 / elapsed if elapsed else 0.0))

        all_rounds.append((round_num, round_results))
    return all_rounds


# ---------------------------------------------------------------------------
# 4. Context compaction
# ---------------------------------------------------------------------------


def test_context_compaction(services: object) -> list[tuple[int, list[tuple[str, float, int]]]]:
    messages = []
    for i in range(300):
        messages.append(HumanMessage(content="q " + "x " * 300))
        messages.append(AIMessage(content="a " + "y " * 300))
    total_tokens = estimate_messages_tokens(messages)
    model = services.settings.model  # type: ignore[attr-defined]

    from minicode.features.context.manager import ContextManager

    all_rounds = []
    for round_num in range(1, ROUNDS + 1):
        round_results: list[tuple[str, float, int]] = []

        elapsed = timeit.timeit(
            lambda: estimate_messages_tokens(messages), number=50
        )
        round_results.append(
            ("estimate_messages (600 msgs)", 50 / elapsed if elapsed else 0.0, total_tokens)
        )

        def run_compact() -> None:
            cm = ContextManager(model=model)
            cm.compact(messages)

        elapsed = timeit.timeit(run_compact, number=5)
        round_results.append(
            ("compact (600 msgs)", 5 / elapsed if elapsed else 0.0, total_tokens)
        )

        all_rounds.append((round_num, round_results))
    return all_rounds


# ---------------------------------------------------------------------------
# 5. Agent loop throughput
# ---------------------------------------------------------------------------


def test_agent_loop(services: object) -> list[tuple[int, list[tuple[str, float]]]]:
    final_model = _FakeChatModel([AIMessage(content="<final> done")])
    progress_model = _FakeChatModel(
        [
            AIMessage(content="<progress> ..."),
            AIMessage(content="<final> done"),
        ]
    )

    all_rounds = []
    for round_num in range(1, ROUNDS + 1):
        round_results: list[tuple[str, float]] = []

        elapsed = timeit.timeit(
            lambda: run_turn(
                services=services,
                prompt="ping",
                chat_model=final_model,
                persist=False,
            ),
            number=10,
        )
        round_results.append(("run_turn (final)", 10 / elapsed if elapsed else 0.0))

        elapsed = timeit.timeit(
            lambda: run_turn(
                services=services,
                prompt="ping",
                chat_model=progress_model,
                persist=False,
                max_steps=6,
            ),
            number=10,
        )
        round_results.append(
            ("run_turn (progress+final)", 10 / elapsed if elapsed else 0.0)
        )

        all_rounds.append((round_num, round_results))
    return all_rounds


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _print_ops_rounds(
    title: str, rounds: list[tuple[int, list[tuple[str, float]]]]
) -> None:
    print(f"\n[*] {title} (ops/sec)")
    print("-" * 70)
    for round_num, items in rounds:
        print(f"Round {round_num}:")
        for name, ops in items:
            print(f"  {name:<30} {ops:>14,.0f} ops/sec")


def _print_token_rounds(
    rounds: list[tuple[int, list[tuple[str, float, int]]]]
) -> None:
    print("\n[*] Token Estimation (ops/sec)")
    print("-" * 70)
    for round_num, items in rounds:
        print(f"Round {round_num}:")
        for name, ops, tokens in items:
            print(f"  {name:<12} {ops:>14,.0f} ops/sec  -> {tokens} tokens")


def _print_compaction_rounds(
    rounds: list[tuple[int, list[tuple[str, float, int]]]]
) -> None:
    print("\n[*] Context Compaction (ops/sec)")
    print("-" * 70)
    for round_num, items in rounds:
        print(f"Round {round_num}:")
        for name, ops, tokens in items:
            print(f"  {name:<32} {ops:>12,.2f} ops/sec  ({tokens} tokens)")


def main() -> None:
    print("=" * 80)
    print("MiniCode Python Multi-Round Performance Stress Test")
    print(f"Rounds per test: {ROUNDS}")
    print("=" * 80)

    workspace, services, tmpdir = _build_workspace()
    try:
        token_rounds = test_token_estimation()
        slash_rounds = test_slash_command_matching()
        tool_rounds = test_tool_execution(workspace, services)
        compact_rounds = test_context_compaction(services)
        agent_rounds = test_agent_loop(services)
    finally:
        try:
            tmpdir.cleanup()
        except Exception:
            pass

    _print_token_rounds(token_rounds)
    _print_ops_rounds("Slash Command Matching", slash_rounds)
    _print_ops_rounds("Tool Execution", tool_rounds)
    _print_compaction_rounds(compact_rounds)
    _print_ops_rounds("Agent Loop (run_turn)", agent_rounds)

    # ---------- summary ----------
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)

    def _avg_first(rounds, col_index=1) -> float:
        return sum(r[1][0][col_index] for r in rounds) / max(1, len(rounds))

    avg_token_ascii = _avg_first(token_rounds, col_index=1)
    avg_slash = _avg_first(slash_rounds, col_index=1)
    avg_list = _avg_first(tool_rounds, col_index=1)
    avg_compact = _avg_first(compact_rounds, col_index=1)
    avg_agent = _avg_first(agent_rounds, col_index=1)

    print(f"Token Estimation (ASCII avg):       {avg_token_ascii:>14,.0f} ops/sec")
    print(f"Slash Command Matching (avg):       {avg_slash:>14,.0f} ops/sec")
    print(f"list_files Tool (avg):              {avg_list:>14,.0f} ops/sec")
    print(f"estimate_messages_tokens (avg):     {avg_compact:>14,.2f} ops/sec")
    print(f"run_turn final-answer (avg):        {avg_agent:>14,.2f} ops/sec")

    print(f"\n[OK] Completed {ROUNDS} rounds across 5 areas.")

    output_file = Path(__file__).parent / "stress_test_results.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("Multi-Round Performance Stress Test Results\n")
        f.write("=" * 70 + "\n\n")

        f.write("Token Estimation:\n")
        for round_num, items in token_rounds:
            f.write(f"  Round {round_num}:\n")
            for name, ops, tokens in items:
                f.write(f"    {name}: {ops:,.0f} ops/sec -> {tokens} tokens\n")

        f.write("\nSlash Command Matching:\n")
        for round_num, items in slash_rounds:
            f.write(f"  Round {round_num}:\n")
            for name, ops in items:
                f.write(f"    {name}: {ops:,.0f} ops/sec\n")

        f.write("\nTool Execution:\n")
        for round_num, items in tool_rounds:
            f.write(f"  Round {round_num}:\n")
            for name, ops in items:
                f.write(f"    {name}: {ops:,.0f} ops/sec\n")

        f.write("\nContext Compaction:\n")
        for round_num, items in compact_rounds:
            f.write(f"  Round {round_num}:\n")
            for name, ops, tokens in items:
                f.write(f"    {name}: {ops:,.2f} ops/sec ({tokens} tokens)\n")

        f.write("\nAgent Loop (run_turn):\n")
        for round_num, items in agent_rounds:
            f.write(f"  Round {round_num}:\n")
            for name, ops in items:
                f.write(f"    {name}: {ops:,.2f} ops/sec\n")

    print(f"\n[OK] Results saved to: {output_file}")


if __name__ == "__main__":
    main()
