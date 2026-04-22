from __future__ import annotations

import argparse
import logging
from pathlib import Path

from minicode.app.bootstrap import bootstrap_services
from minicode.cron import CronConfig, CronJob, CronRunner
from minicode.platform.logging import setup_logging
from minicode.platform.paths import resolve_paths
from minicode.runtime.runner import run_turn

_log = logging.getLogger("minicode.cron")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MiniCode Next cron runner")
    parser.add_argument(
        "--config",
        default=None,
        help="Path to cron.json (defaults to <global_dir>/cron.json).",
    )
    parser.add_argument("--cwd", default=".")
    parser.add_argument(
        "--run-once",
        action="store_true",
        help="Run any due jobs once and exit, instead of looping.",
    )
    parser.add_argument("--tick-seconds", type=float, default=5.0)
    parser.add_argument("--max-parallel", type=int, default=1)
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--structured-logs", action="store_true")
    return parser


def _make_runner(services) -> "callable":  # type: ignore[type-arg]
    def _run(job: CronJob) -> str | None:
        result = run_turn(services=services, prompt=job.prompt, mode="auto")
        if result.error:
            _log.warning("cron job %s returned error: %s", job.name, result.error)
        return result.final_text

    return _run


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = resolve_paths(args.cwd)
    setup_logging(
        paths.logs_dir,
        level=args.log_level,
        structured=args.structured_logs,
        log_name="cron.log",
    )
    config_path = Path(args.config) if args.config else (paths.global_dir / "cron.json")
    if not config_path.exists():
        _log.warning("No cron config at %s; nothing to do.", config_path)
        return 0
    services = bootstrap_services(args.cwd)
    config = CronConfig.load(config_path)
    runner = CronRunner(
        config,
        _make_runner(services),
        tick_seconds=args.tick_seconds,
        max_parallel=args.max_parallel,
    )
    if args.run_once:
        fired = runner.run_once()
        _log.info("run-once fired %d job(s)", len(fired))
        return 0
    try:
        runner.run_forever()
    except KeyboardInterrupt:
        _log.info("cron runner interrupted")
        runner.stop()
    return 0
