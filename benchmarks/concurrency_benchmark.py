"""Level 1 concurrency benchmark: ToolGraphAdapter batch parallelism.

Measures the speedup of submitting N concurrency-safe tool calls as a single
batch (one `adapter.execute([...N calls...])`) versus running them as N separate
single-call executes. This exercises the ThreadPoolExecutor path in
`features/tools/graph_adapter.py:77` and is the main designed concurrency point
of the system.

Reports for each N:
- serial wall-clock (sum of N single executes)
- parallel wall-clock (one execute with N calls)
- speedup ratio = serial / parallel
- per-call throughput (calls/sec)

`max_workers` is capped at `min(4, len(tool_calls))`, so speedup should plateau
around 4× once N >= 4 for I/O-bound tools. CPU-bound tools will hit the GIL
ceiling earlier.
"""

from __future__ import annotations

import os
import statistics
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# Isolate state before importing minicode.
_BENCH_HOME = Path(tempfile.mkdtemp(prefix="minicode_concurrency_home_"))
os.environ.setdefault("MINICODE_HOME", str(_BENCH_HOME))
os.environ.setdefault("MINICODE_PROVIDER", "openai")
os.environ.setdefault("MINICODE_MODEL", "gpt-4o-mini")
os.environ.setdefault("OPENAI_API_KEY", "concurrency-key")

from minicode.app.bootstrap import bootstrap_services  # noqa: E402
from minicode.core.types import ToolContext  # noqa: E402
from minicode.features.tools.graph_adapter import ToolGraphAdapter  # noqa: E402

# Scan N values up to 16 so we see the plateau past max_workers=4.
BATCH_SIZES = [1, 2, 4, 8, 16]
TRIALS = 5
WARMUP = 2


# ---------------------------------------------------------------------------
# Workspace setup
# ---------------------------------------------------------------------------


@dataclass
class BenchEnv:
    workspace: Path
    services: object
    adapter: ToolGraphAdapter
    context: ToolContext


def build_env(num_files: int = 80, lines_per_file: int = 400) -> BenchEnv:
    """I/O-heavy workspace so grep_files has real wall-clock per call."""
    tmp = Path(tempfile.mkdtemp(prefix="minicode_concurrency_ws_"))
    for i in range(num_files):
        lines = [f"line {j} of file {i} - needle{i % 7}" for j in range(lines_per_file)]
        (tmp / f"file_{i:03d}.txt").write_text("\n".join(lines), encoding="utf-8")
    services = bootstrap_services(str(tmp))
    adapter = ToolGraphAdapter(services.tools, services.permissions)
    context = ToolContext(
        thread_id="concurrency-bench",
        cwd=str(tmp),
        mode="bypass",
        services=services,
        emit_event=lambda *_a, **_kw: None,
    )
    return BenchEnv(workspace=tmp, services=services, adapter=adapter, context=context)


# ---------------------------------------------------------------------------
# Call builders
# ---------------------------------------------------------------------------


def grep_call(call_id: int) -> dict:
    return {
        "id": f"call_{call_id}",
        "name": "grep_files",
        "args": {"path": ".", "pattern": f"needle{call_id % 7}", "max_matches": 200},
    }


def read_call(call_id: int) -> dict:
    return {
        "id": f"call_{call_id}",
        "name": "read_file",
        "args": {"path": f"file_{call_id % 80:03d}.txt"},
    }


# ---------------------------------------------------------------------------
# Measurement primitives
# ---------------------------------------------------------------------------


def time_parallel(env: BenchEnv, calls: list[dict]) -> float:
    start = time.perf_counter()
    env.adapter.execute(calls, env.context)
    return time.perf_counter() - start


def time_serial(env: BenchEnv, calls: list[dict]) -> float:
    start = time.perf_counter()
    for call in calls:
        env.adapter.execute([call], env.context)
    return time.perf_counter() - start


def median_of(fn, trials: int, warmup: int) -> float:
    for _ in range(warmup):
        fn()
    samples = [fn() for _ in range(trials)]
    return statistics.median(samples)


# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------


def assert_parallel_path(env: BenchEnv, calls: list[dict]) -> None:
    """Confirm the batch actually goes through the ThreadPoolExecutor branch."""
    serial = env.adapter._should_run_serially(calls, env.context.mode)
    if serial:
        raise RuntimeError(
            "Expected parallel path; got serial. Check tool capabilities or mode."
        )


# ---------------------------------------------------------------------------
# Benchmark scenarios
# ---------------------------------------------------------------------------


@dataclass
class Row:
    scenario: str
    n: int
    serial_ms: float
    parallel_ms: float
    speedup: float
    parallel_throughput: float

    def format(self) -> str:
        return (
            f"{self.scenario:<18} N={self.n:<3} "
            f"serial={self.serial_ms:>8.2f} ms  "
            f"parallel={self.parallel_ms:>8.2f} ms  "
            f"speedup={self.speedup:>5.2f}x  "
            f"throughput={self.parallel_throughput:>8.1f} calls/s"
        )


def run_scenario(env: BenchEnv, scenario: str, build_calls) -> list[Row]:
    rows: list[Row] = []
    for n in BATCH_SIZES:
        calls = [build_calls(i) for i in range(n)]
        assert_parallel_path(env, calls)

        parallel_s = median_of(lambda: time_parallel(env, calls), TRIALS, WARMUP)
        serial_s = median_of(lambda: time_serial(env, calls), TRIALS, WARMUP)
        speedup = serial_s / parallel_s if parallel_s > 0 else 0.0
        throughput = n / parallel_s if parallel_s > 0 else 0.0
        rows.append(
            Row(
                scenario=scenario,
                n=n,
                serial_ms=serial_s * 1000,
                parallel_ms=parallel_s * 1000,
                speedup=speedup,
                parallel_throughput=throughput,
            )
        )
        print(f"  {rows[-1].format()}")
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print("=" * 96)
    print("MiniCode Python - Level 1 Concurrency Benchmark")
    print("ToolGraphAdapter.execute() - batch parallel vs sequential single-call")
    print(f"Batch sizes: {BATCH_SIZES}   Trials/N: {TRIALS}   Warmup: {WARMUP}")
    print("max_workers cap: min(4, len(tool_calls))  (see graph_adapter.py:77)")
    print("=" * 96)

    env = build_env()
    all_rows: list[Row] = []
    try:
        print("\n[A] grep_files x N  (Python-heavy, GIL-bound regex scan per call)")
        print("-" * 96)
        all_rows += run_scenario(env, "grep_files", grep_call)

        print("\n[B] read_file x N  (light I/O, GIL released during read_text)")
        print("-" * 96)
        all_rows += run_scenario(env, "read_file", read_call)
    finally:
        import shutil

        shutil.rmtree(env.workspace, ignore_errors=True)

    # ---------- summary ----------
    print("\n" + "=" * 96)
    print("Summary")
    print("=" * 96)
    print(
        f"{'Scenario':<12} {'N':>3}  {'Serial (ms)':>12}  {'Parallel (ms)':>14}  "
        f"{'Speedup':>8}  {'Calls/sec':>10}"
    )
    print("-" * 96)
    for r in all_rows:
        print(
            f"{r.scenario:<12} {r.n:>3}  {r.serial_ms:>12.2f}  {r.parallel_ms:>14.2f}  "
            f"{r.speedup:>7.2f}x  {r.parallel_throughput:>10.1f}"
        )

    # ---------- interpretation hints ----------
    print("\nInterpretation hints")
    print("-" * 96)
    print("  * speedup ~= 1.0x at N=1: single-call path, no thread overhead expected")
    print("  * speedup should rise toward ~min(N, 4) for I/O-bound tools")
    print("  * if grep_files speedup saturates well below 4x, the GIL is the ceiling")
    print("    (regex scan + Python loop hold the GIL; threading can't parallelize)")
    print("  * if read_file shows speedup < 1.0 (slower parallel!), thread overhead")
    print("    exceeds per-call work - expected for sub-millisecond tools")

    # ---------- save ----------
    out = Path(__file__).parent / "concurrency_results.txt"
    with open(out, "w", encoding="utf-8") as f:
        f.write("ToolGraphAdapter Concurrency Benchmark Results\n")
        f.write("=" * 96 + "\n\n")
        f.write(
            f"{'Scenario':<12} {'N':>3}  {'Serial (ms)':>12}  {'Parallel (ms)':>14}  "
            f"{'Speedup':>8}  {'Calls/sec':>10}\n"
        )
        f.write("-" * 96 + "\n")
        for r in all_rows:
            f.write(
                f"{r.scenario:<12} {r.n:>3}  {r.serial_ms:>12.2f}  {r.parallel_ms:>14.2f}  "
                f"{r.speedup:>7.2f}x  {r.parallel_throughput:>10.1f}\n"
            )
    print(f"\n[OK] Results saved to: {out}")


if __name__ == "__main__":
    main()
