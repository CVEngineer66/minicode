from __future__ import annotations

import json
import logging
import logging.handlers
import sys
import time
from pathlib import Path
from typing import Any

LOG_MAX_BYTES = 10 * 1024 * 1024  # 10 MB per file
LOG_BACKUP_COUNT = 5
CONSOLE_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
FILE_FORMAT = "%(asctime)s [%(levelname)s] %(name)s %(filename)s:%(lineno)d: %(message)s"


class StructuredFormatter(logging.Formatter):
    """JSON-per-line formatter for machine-parseable logs."""

    _STRUCTURED_KEYS = (
        "tool_name",
        "model",
        "duration_ms",
        "tokens_in",
        "tokens_out",
        "cost",
        "error_category",
        "session_id",
        "thread_id",
        "workspace",
    )

    def format(self, record: logging.LogRecord) -> str:
        entry: dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(record.created)),
            "level": record.levelname,
            "module": record.name,
            "msg": record.getMessage(),
            "file": f"{record.filename}:{record.lineno}",
        }
        for key in self._STRUCTURED_KEYS:
            value = getattr(record, key, None)
            if value is not None:
                entry[key] = value
        if record.exc_info and record.exc_info[1] is not None:
            entry["exception"] = str(record.exc_info[1])
            entry["exc_type"] = type(record.exc_info[1]).__name__
        return json.dumps(entry, ensure_ascii=False, default=str)


def setup_logging(
    log_dir: Path,
    *,
    level: str = "WARNING",
    log_to_file: bool = True,
    log_to_console: bool = True,
    structured: bool = False,
    log_name: str = "minicode.log",
) -> logging.Logger:
    """Configure the `minicode` logger hierarchy.

    Returns the root application logger. Idempotent within a process: existing
    handlers are cleared on reconfigure.
    """
    if log_to_file:
        log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("minicode")
    logger.setLevel(getattr(logging, level.upper(), logging.WARNING))
    logger.handlers.clear()

    file_formatter: logging.Formatter
    console_formatter: logging.Formatter
    if structured:
        file_formatter = StructuredFormatter()
        console_formatter = StructuredFormatter()
    else:
        file_formatter = logging.Formatter(FILE_FORMAT)
        console_formatter = logging.Formatter(CONSOLE_FORMAT)

    if log_to_file:
        file_handler = logging.handlers.RotatingFileHandler(
            log_dir / log_name,
            maxBytes=LOG_MAX_BYTES,
            backupCount=LOG_BACKUP_COUNT,
            encoding="utf-8",
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    if log_to_console:
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setLevel(getattr(logging, level.upper(), logging.WARNING))
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

    for noisy in ("urllib3", "httpx", "openai", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    return logger


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"minicode.{name}")


def log_stats(log_dir: Path, log_name: str = "minicode.log") -> dict[str, Any]:
    log_file = log_dir / log_name
    stats: dict[str, Any] = {"log_file": str(log_file), "exists": log_file.exists()}
    if log_file.exists():
        size = log_file.stat().st_size
        stats["size_bytes"] = size
        stats["size_mb"] = round(size / (1024 * 1024), 2)
        stats["max_size_mb"] = LOG_MAX_BYTES / (1024 * 1024)
        stats["rotation_pct"] = round(size / LOG_MAX_BYTES * 100, 1)
    rotated = list(log_dir.glob(f"{log_name}.*"))
    stats["rotated_files"] = len(rotated)
    stats["max_rotated"] = LOG_BACKUP_COUNT
    return stats
