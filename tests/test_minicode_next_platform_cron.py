from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from minicode.cron import CronConfig, CronJob, CronRunner
from minicode.platform.logging import StructuredFormatter, get_logger, setup_logging


def test_cron_job_due_honours_interval():
    job = CronJob(name="x", schedule="every:30s", prompt="hi")
    assert job.due(1000) is True
    job.last_run_at = 1000
    assert job.due(1020) is False
    assert job.due(1031) is True


def test_cron_job_disabled_never_due():
    job = CronJob(name="x", schedule="every:10s", prompt="hi", enabled=False)
    assert job.due(9999) is False


def test_cron_job_bad_schedule_not_due():
    job = CronJob(name="x", schedule="bogus", prompt="hi")
    assert job.due(1) is False


def test_cron_config_load_from_json(tmp_path: Path):
    path = tmp_path / "cron.json"
    path.write_text(
        '{"jobs":[{"name":"a","schedule":"every:1m","prompt":"p","enabled":true}]}',
        encoding="utf-8",
    )
    cfg = CronConfig.load(path)
    assert len(cfg.jobs) == 1
    assert cfg.jobs[0].name == "a"


def test_cron_config_load_missing_path_empty(tmp_path: Path):
    assert CronConfig.load(tmp_path / "missing.json").jobs == []


def test_cron_runner_run_once_fires_due_jobs():
    cfg = CronConfig(
        jobs=[
            CronJob(name="ready", schedule="every:1s", prompt="p"),
            CronJob(name="later", schedule="every:99d", prompt="p", last_run_at=9e18),
        ]
    )
    calls: list[str] = []

    def runner(job: CronJob) -> None:
        calls.append(job.name)

    r = CronRunner(cfg, runner)
    fired = r.run_once()
    assert [j.name for j in fired] == ["ready"]
    assert calls == ["ready"]


def test_cron_runner_captures_errors_without_stopping():
    cfg = CronConfig(jobs=[CronJob(name="boom", schedule="every:1s", prompt="p")])

    def runner(job: CronJob) -> None:
        raise RuntimeError("bang")

    r = CronRunner(cfg, runner)
    r.run_once()
    assert cfg.jobs[0].error_count == 1
    assert "bang" in (cfg.jobs[0].last_error or "")


def test_structured_formatter_includes_extras():
    import logging

    fmt = StructuredFormatter()
    record = logging.LogRecord(
        name="minicode.x",
        level=logging.INFO,
        pathname="x.py",
        lineno=10,
        msg="hi",
        args=(),
        exc_info=None,
    )
    record.tool_name = "read_file"  # type: ignore[attr-defined]
    import json

    out = fmt.format(record)
    parsed = json.loads(out)
    assert parsed["level"] == "INFO"
    assert parsed["tool_name"] == "read_file"


def test_setup_logging_is_idempotent(tmp_path: Path):
    setup_logging(tmp_path, level="DEBUG", log_to_console=False)
    logger = setup_logging(tmp_path, level="DEBUG", log_to_console=False)
    assert logger.name == "minicode"
    # Single rotating file handler retained after re-setup
    assert len([h for h in logger.handlers]) == 1


def test_get_logger_namespaced():
    assert get_logger("gateway").name == "minicode.gateway"
