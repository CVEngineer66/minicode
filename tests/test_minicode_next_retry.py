from __future__ import annotations

import pytest

from minicode.runtime.retry import (
    APIRetryExhaustedError,
    ErrorCategory,
    HTTPError,
    calculate_backoff,
    classify_error,
    is_retryable,
    retry_with_backoff,
)


def test_classify_rate_limit_and_overload():
    assert classify_error(HTTPError("rate", 429)) == ErrorCategory.RATE_LIMIT
    assert classify_error(HTTPError("overload", 529)) == ErrorCategory.OVERLOAD


def test_classify_network_from_message():
    assert classify_error(RuntimeError("Connection refused")) == ErrorCategory.NETWORK_ERROR


def test_classify_auth_not_retryable():
    assert is_retryable(classify_error(HTTPError("no", 401))) is False


def test_calculate_backoff_monotonic():
    b1 = calculate_backoff(0, base=1.0, jitter=0.0)
    b2 = calculate_backoff(2, base=1.0, jitter=0.0)
    assert b2 > b1


def test_calculate_backoff_respects_retry_after():
    b = calculate_backoff(0, retry_after=5.0, base=1.0, jitter=0.0)
    assert b >= 1.0


def test_retry_with_backoff_succeeds_after_retries():
    calls = {"n": 0}

    def flaky() -> str:
        calls["n"] += 1
        if calls["n"] < 3:
            raise HTTPError("transient", 503)
        return "ok"

    result = retry_with_backoff(flaky, max_retries=5, base_backoff=0.001, sleep=lambda _s: None)
    assert result == "ok"
    assert calls["n"] == 3


def test_retry_with_backoff_exhausts():
    def always_fail() -> str:
        raise HTTPError("boom", 500)

    with pytest.raises(APIRetryExhaustedError):
        retry_with_backoff(always_fail, max_retries=2, base_backoff=0.001, sleep=lambda _s: None)


def test_retry_does_not_retry_non_retryable():
    def auth_fail() -> str:
        raise HTTPError("nope", 401)

    with pytest.raises(HTTPError):
        retry_with_backoff(auth_fail, max_retries=3, base_backoff=0.001, sleep=lambda _s: None)
