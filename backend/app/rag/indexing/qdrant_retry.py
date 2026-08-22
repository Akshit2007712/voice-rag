"""Bounded, read-operation retry support for transient Qdrant Cloud failures."""

from __future__ import annotations

import logging
import socket
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar

try:
    import httpx
except ImportError:  # pragma: no cover - dependency is required in production.
    httpx = None


logger = logging.getLogger("uvicorn.error")
Result = TypeVar("Result")


@dataclass(frozen=True)
class QdrantRetryPolicy:
    """Bound retries to two backoffs after the original network request."""

    max_attempts: int = 3
    backoff_seconds: tuple[float, ...] = (0.5, 1.0)

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        if len(self.backoff_seconds) < self.max_attempts - 1:
            raise ValueError("backoff_seconds must provide one delay per retry")


@dataclass(frozen=True)
class QdrantOperationMetrics:
    """Timing and retry facts for one Qdrant operation."""

    operation_ms: float
    retry_wait_ms: float
    wall_ms: float
    retry_count: int


def _status_code(error: BaseException) -> int | None:
    """Find an HTTP status code without depending on a specific client exception."""
    for candidate in (error, getattr(error, "response", None)):
        value = getattr(candidate, "status_code", None)
        if isinstance(value, int):
            return value
    return None


def is_transient_qdrant_error(error: BaseException) -> bool:
    """Retry only DNS, connection, and timeout failures; never HTTP client errors."""
    status = _status_code(error)
    # HTTP responses, including 5xx, reached Qdrant successfully.  This utility is
    # deliberately limited to DNS/connection/timeout resilience, not server-error
    # retry policy, so deterministic API/configuration errors remain visible.
    if status is not None:
        return False
    if isinstance(error, (socket.gaierror, TimeoutError, ConnectionError)):
        return True
    if httpx is not None and isinstance(error, (httpx.ConnectError, httpx.TimeoutException, httpx.NetworkError)):
        return True
    # qdrant-client wraps some httpx transport exceptions in
    # ResponseHandlingException without consistently preserving ``__cause__``.
    # Match only known transport text, never arbitrary API response bodies.
    if type(error).__name__ == "ResponseHandlingException":
        message = str(error).lower()
        if "getaddrinfo failed" in message or "errno 11001" in message or "timed out" in message:
            return True
    cause = error.__cause__ or error.__context__
    return bool(cause is not None and cause is not error and is_transient_qdrant_error(cause))


def transient_reason(error: BaseException) -> str:
    """Return a concise safe reason label, never a URL/key/error body."""
    if isinstance(error, socket.gaierror):
        return "dns_error"
    if isinstance(error, TimeoutError) or (httpx is not None and isinstance(error, httpx.TimeoutException)):
        return "timeout"
    return "connection_error"


def call_with_qdrant_retry(
    operation: str,
    call: Callable[[], Result],
    policy: QdrantRetryPolicy,
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[Result, QdrantOperationMetrics, int]:
    """Execute one remote read with bounded retry and safe telemetry.

    The returned integer is the number of underlying failed requests, including
    failures that later recover.  Callers aggregate it for benchmark visibility.
    """
    started_at = time.perf_counter()
    operation_seconds = 0.0
    retry_wait_seconds = 0.0
    failed_requests = 0
    for attempt in range(1, policy.max_attempts + 1):
        attempt_started_at = time.perf_counter()
        try:
            result = call()
            operation_seconds += time.perf_counter() - attempt_started_at
            return result, QdrantOperationMetrics(
                operation_ms=operation_seconds * 1_000,
                retry_wait_ms=retry_wait_seconds * 1_000,
                wall_ms=(time.perf_counter() - started_at) * 1_000,
                retry_count=attempt - 1,
            ), failed_requests
        except BaseException as error:
            operation_seconds += time.perf_counter() - attempt_started_at
            failed_requests += 1
            if not is_transient_qdrant_error(error) or attempt == policy.max_attempts:
                raise RuntimeError(f"Qdrant remote {operation} failed after {attempt} attempt(s).") from error
            logger.warning("QDRANT_RETRY attempt=%d reason=%s operation=%s", attempt, transient_reason(error), operation)
            delay = policy.backoff_seconds[attempt - 1]
            sleep(delay)
            retry_wait_seconds += delay
    raise AssertionError("unreachable")
