from __future__ import annotations

import random
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

MAX_RETRIES = 3
BASE_BACKOFF = 1.0
MAX_BACKOFF = 60.0
JITTER_FACTOR = 0.5
RETRYABLE_STATUS = {429, 500, 502, 503, 504, 529}


class ErrorCategory(str, Enum):
    RATE_LIMIT = "rate_limit"
    SERVER_ERROR = "server_error"
    NETWORK_ERROR = "network_error"
    AUTH_ERROR = "auth_error"
    INPUT_ERROR = "input_error"
    OVERLOAD = "overload"
    UNKNOWN = "unknown"


_CATEGORY_BACKOFF: dict[ErrorCategory, float] = {
    ErrorCategory.RATE_LIMIT: 2.0,
    ErrorCategory.SERVER_ERROR: 1.0,
    ErrorCategory.NETWORK_ERROR: 0.5,
    ErrorCategory.OVERLOAD: 3.0,
    ErrorCategory.UNKNOWN: 1.0,
}

_CATEGORY_MAX_RETRIES: dict[ErrorCategory, int] = {
    ErrorCategory.NETWORK_ERROR: 5,
    ErrorCategory.OVERLOAD: 5,
    ErrorCategory.RATE_LIMIT: 4,
}

_OVERLOAD_RE = re.compile(
    r"(?:overloaded|overload|capacity|too many requests|temporarily unavailable|"
    r"please try again later|service is currently unavailable|api is temporarily|"
    r"capacity exceeded|high demand)",
    re.IGNORECASE,
)

_NETWORK_RE = re.compile(
    r"(?:connection\s*(?:refused|reset|timeout|aborted)|timed?\s*out|"
    r"dns\s*resolution|name\s*resolution|network\s*(?:error|unreachable|down)|"
    r"socket\s*(?:error|closed)|eof\s*occurred|ssl\s*error|handshake\s*failed)",
    re.IGNORECASE,
)


class HTTPError(Exception):
    def __init__(
        self,
        message: str,
        status_code: int,
        retry_after: float | None = None,
        response: Any = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.retry_after = retry_after
        self.response = response


class APIRetryExhaustedError(Exception):
    def __init__(
        self,
        message: str,
        attempts: int,
        last_error: Exception | None = None,
        category: ErrorCategory = ErrorCategory.UNKNOWN,
    ) -> None:
        super().__init__(message)
        self.attempts = attempts
        self.last_error = last_error
        self.category = category


def classify_error(error: Exception) -> ErrorCategory:
    status_code = getattr(error, "status_code", None)
    if status_code is not None:
        if status_code == 429:
            return ErrorCategory.RATE_LIMIT
        if status_code == 529:
            return ErrorCategory.OVERLOAD
        if status_code in (401, 403):
            return ErrorCategory.AUTH_ERROR
        if status_code in (400, 404, 405, 409, 413, 415, 422):
            return ErrorCategory.INPUT_ERROR
        if status_code in (500, 502, 503, 504):
            if _OVERLOAD_RE.search(str(error)):
                return ErrorCategory.OVERLOAD
            return ErrorCategory.SERVER_ERROR
    msg = str(error)
    if _NETWORK_RE.search(msg):
        return ErrorCategory.NETWORK_ERROR
    if _OVERLOAD_RE.search(msg):
        return ErrorCategory.OVERLOAD
    if any(k in type(error).__name__.lower() for k in ("timeout", "connection", "socket")):
        return ErrorCategory.NETWORK_ERROR
    return ErrorCategory.UNKNOWN


def is_retryable(category: ErrorCategory) -> bool:
    return category in {
        ErrorCategory.RATE_LIMIT,
        ErrorCategory.SERVER_ERROR,
        ErrorCategory.NETWORK_ERROR,
        ErrorCategory.OVERLOAD,
        ErrorCategory.UNKNOWN,
    }


def calculate_backoff(
    attempt: int,
    retry_after: float | None = None,
    base: float = BASE_BACKOFF,
    max_wait: float = MAX_BACKOFF,
    jitter: float = JITTER_FACTOR,
    category: ErrorCategory | None = None,
) -> float:
    effective_base = base * (_CATEGORY_BACKOFF.get(category, 1.0) if category else 1.0)
    if retry_after is not None and retry_after > 0:
        min_wait = effective_base * (2 ** min(attempt, 2))
        return max(min(retry_after, max_wait), min_wait)
    backoff = effective_base * (2 ** attempt)
    jitter_range = backoff * jitter
    backoff += random.uniform(-jitter_range, jitter_range)
    return max(0.1, min(backoff, max_wait))


@dataclass
class RetryState:
    attempts: int = 0
    max_attempts: int = MAX_RETRIES
    total_wait_time: float = 0.0
    last_error: str | None = None
    last_category: ErrorCategory = ErrorCategory.UNKNOWN
    category_history: list[ErrorCategory] = field(default_factory=list)
    succeeded: bool = False


def retry_with_backoff(
    func: Callable[..., Any],
    *args: Any,
    max_retries: int = MAX_RETRIES,
    base_backoff: float = BASE_BACKOFF,
    max_backoff: float = MAX_BACKOFF,
    on_retry: Callable[[RetryState], None] | None = None,
    sleep: Callable[[float], None] = time.sleep,
    **kwargs: Any,
) -> Any:
    """Retry a callable on transient failures with adaptive backoff."""
    state = RetryState(max_attempts=max_retries)
    last_error: Exception | None = None
    last_category = ErrorCategory.UNKNOWN
    for attempt in range(max_retries + 1):
        try:
            result = func(*args, **kwargs)
            state.succeeded = True
            state.attempts = attempt + 1
            return result
        except Exception as exc:
            category = classify_error(exc)
            state.last_category = category
            state.category_history.append(category)
            if not is_retryable(category):
                raise
            state.attempts = attempt + 1
            state.last_error = str(exc)
            last_error = exc
            last_category = category
            cat_max = _CATEGORY_MAX_RETRIES.get(category)
            effective_max = cat_max if cat_max is not None else max_retries
            if attempt >= effective_max:
                raise APIRetryExhaustedError(
                    f"API call failed after {attempt + 1} attempts "
                    f"(category: {category.value}): {exc}",
                    attempts=attempt + 1,
                    last_error=exc,
                    category=category,
                ) from exc
            wait = calculate_backoff(
                attempt,
                retry_after=getattr(exc, "retry_after", None),
                base=base_backoff,
                max_wait=max_backoff,
                category=category,
            )
            state.total_wait_time += wait
            if on_retry:
                on_retry(state)
            sleep(wait)
    # Unreachable: the loop exits only via return or raise
    raise APIRetryExhaustedError(
        f"retry budget exhausted: {last_error}",
        attempts=max_retries + 1,
        last_error=last_error,
        category=last_category,
    )
