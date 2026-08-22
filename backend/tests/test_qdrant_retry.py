"""Unit tests for bounded, transport-only Qdrant retry behavior."""

from __future__ import annotations

import logging
import socket
import unittest

from app.rag.indexing.qdrant_retry import QdrantRetryPolicy, call_with_qdrant_retry


class StatusError(RuntimeError):
    def __init__(self, status_code: int) -> None:
        self.status_code = status_code


class QdrantRetryTests(unittest.TestCase):
    def test_transient_dns_failure_then_success_retries_once(self) -> None:
        calls = 0

        def call():
            nonlocal calls
            calls += 1
            if calls == 1:
                raise socket.gaierror(11001, "host not found")
            return "ok"

        result, metrics, failures = call_with_qdrant_retry("query_points", call, QdrantRetryPolicy(), sleep=lambda _seconds: None)
        self.assertEqual(result, "ok")
        self.assertEqual(calls, 2)
        self.assertEqual(metrics.retry_count, 1)
        self.assertEqual(failures, 1)

    def test_qdrant_response_wrapper_with_dns_text_is_retried(self) -> None:
        wrapper = type("ResponseHandlingException", (RuntimeError,), {})
        calls = 0

        def call():
            nonlocal calls
            calls += 1
            if calls == 1:
                raise wrapper("[Errno 11001] getaddrinfo failed")
            return "ok"

        result, metrics, _failures = call_with_qdrant_retry("query_points", call, QdrantRetryPolicy(), sleep=lambda _seconds: None)
        self.assertEqual(result, "ok")
        self.assertEqual(metrics.retry_count, 1)

    def test_retry_exhaustion_is_bounded(self) -> None:
        calls = 0

        def call():
            nonlocal calls
            calls += 1
            raise socket.gaierror(11001, "host not found")

        with self.assertRaisesRegex(RuntimeError, "failed after 3 attempt"):
            call_with_qdrant_retry("count", call, QdrantRetryPolicy(), sleep=lambda _seconds: None)
        self.assertEqual(calls, 3)

    def test_400_is_not_retried(self) -> None:
        calls = 0

        def call():
            nonlocal calls
            calls += 1
            raise StatusError(400)

        with self.assertRaises(RuntimeError):
            call_with_qdrant_retry("query_points", call, QdrantRetryPolicy(), sleep=lambda _seconds: None)
        self.assertEqual(calls, 1)

    def test_auth_failure_is_not_retried_or_logged_with_secrets(self) -> None:
        calls = 0
        captured: list[str] = []
        logger = logging.getLogger("uvicorn.error")
        handler = logging.Handler()
        handler.emit = lambda record: captured.append(record.getMessage())  # type: ignore[method-assign]
        logger.addHandler(handler)
        try:
            def call():
                nonlocal calls
                calls += 1
                raise StatusError(401)

            with self.assertRaises(RuntimeError):
                call_with_qdrant_retry("collection_exists", call, QdrantRetryPolicy(), sleep=lambda _seconds: None)
        finally:
            logger.removeHandler(handler)
        self.assertEqual(calls, 1)
        self.assertFalse(any("secret" in message.lower() for message in captured))


if __name__ == "__main__":
    unittest.main()
