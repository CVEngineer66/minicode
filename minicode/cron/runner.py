from __future__ import annotations

import json
import logging
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

_log = logging.getLogger("minicode.cron")


@dataclass
class CronJob:
    """A single scheduled job.

    Supports two schedule formats for simplicity (no third-party dep):
    - `every:<N><unit>` — e.g. "every:30m", "every:4h", "every:300s"
    - `<seconds>` — integer number of seconds between runs
    Full 5-field crontab syntax is intentionally out of scope for the first cut;
    operators who need it can schedule via the system cron calling
    `minicode-next-headless`.
    """

    name: str
    schedule: str
    prompt: str
    enabled: bool = True
    last_run_at: float = 0.0
    run_count: int = 0
    error_count: int = 0
    last_error: str | None = None

    def interval_seconds(self) -> float:
        s = self.schedule.strip().lower()
        if s.startswith("every:"):
            s = s[len("every:"):]
        m = re.fullmatch(r"(\d+)([smhd]?)", s)
        if not m:
            raise ValueError(f"unsupported schedule: {self.schedule!r}")
        value = int(m.group(1))
        unit = m.group(2) or "s"
        return value * {"s": 1, "m": 60, "h": 3600, "d": 86400}[unit]

    def due(self, now: float) -> bool:
        if not self.enabled:
            return False
        try:
            interval = self.interval_seconds()
        except ValueError as exc:
            _log.warning("bad schedule for %s: %s", self.name, exc)
            return False
        return (now - self.last_run_at) >= interval


@dataclass
class CronConfig:
    jobs: list[CronJob] = field(default_factory=list)

    @classmethod
    def load(cls, path: Path) -> "CronConfig":
        if not path.exists():
            return cls()
        data = json.loads(path.read_text(encoding="utf-8"))
        raw_jobs = data.get("jobs", [])
        jobs = [
            CronJob(
                name=str(j.get("name", "")),
                schedule=str(j.get("schedule", "")),
                prompt=str(j.get("prompt", "")),
                enabled=bool(j.get("enabled", True)),
            )
            for j in raw_jobs
            if isinstance(j, dict) and j.get("name")
        ]
        return cls(jobs=jobs)


JobRunner = Callable[[CronJob], str | None]


class CronRunner:
    """Bounded scheduler loop.

    Boundaries:
    - `max_parallel` caps concurrent job runs (default 1 → serial).
    - Runner exceptions are captured on the job record; the loop keeps running.
    - `tick_seconds` controls the minimum poll interval (default 5s).
    """

    def __init__(
        self,
        config: CronConfig,
        runner: JobRunner,
        *,
        tick_seconds: float = 5.0,
        max_parallel: int = 1,
    ) -> None:
        self.config = config
        self.runner = runner
        self.tick_seconds = tick_seconds
        self.max_parallel = max(1, max_parallel)
        self._stop = threading.Event()
        self._active: set[str] = set()
        self._lock = threading.Lock()

    def stop(self) -> None:
        self._stop.set()

    def run_forever(self) -> None:
        _log.info("cron loop starting with %d job(s)", len(self.config.jobs))
        while not self._stop.is_set():
            now = time.time()
            for job in list(self.config.jobs):
                if self._stop.is_set():
                    break
                if not job.due(now):
                    continue
                with self._lock:
                    if len(self._active) >= self.max_parallel:
                        continue
                    if job.name in self._active:
                        continue
                    self._active.add(job.name)
                thread = threading.Thread(
                    target=self._run_job,
                    args=(job,),
                    name=f"cron:{job.name}",
                    daemon=True,
                )
                thread.start()
            self._stop.wait(self.tick_seconds)
        _log.info("cron loop stopped")

    def run_once(self) -> list[CronJob]:
        """Run every job whose interval has elapsed once. Returns jobs that fired."""
        fired: list[CronJob] = []
        now = time.time()
        for job in self.config.jobs:
            if not job.due(now):
                continue
            self._run_job(job)
            fired.append(job)
        return fired

    def _run_job(self, job: CronJob) -> None:
        start = time.time()
        _log.info("cron fire %s (%s)", job.name, job.schedule)
        try:
            self.runner(job)
            job.run_count += 1
            job.last_error = None
        except BaseException as exc:
            job.error_count += 1
            job.last_error = f"{type(exc).__name__}: {exc}"
            _log.exception("cron job %s failed", job.name)
        finally:
            job.last_run_at = start
            with self._lock:
                self._active.discard(job.name)
