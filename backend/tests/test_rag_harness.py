"""Focused orchestration tests that keep frozen text-RAG execution unchanged."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from app.services.rag_harness import (
    HarnessFailure,
    HarnessSettings,
    HarnessSuccess,
    RAGHarness,
    RAGRequestContext,
)
from app.services.text_rag import RagLatency, TextRagResult


def _app() -> SimpleNamespace:
    return SimpleNamespace(state=SimpleNamespace())


def _result(input_mode: str = "text", retry_count: int = 0) -> TextRagResult:
    return TextRagResult(
        query="not retained by the harness",
        answer=SimpleNamespace(),
        input_mode=input_mode,
        latency=RagLatency(1.0, 2.0, 3.0, 4.0, 5.0, 0.0, 6.0, 7.0),
        qdrant_retry_count=retry_count,
    )


class _Hooks:
    def __init__(self) -> None:
        self.events = []

    def pre_rag(self, context) -> None:
        self.events.append(("pre", context.request_id))

    def post_retrieval(self, context, result) -> None:
        self.events.append(("post", result.input_mode))


class RAGHarnessTests(unittest.TestCase):
    def test_typed_executes_shared_service_once_and_records_success_once(self) -> None:
        app = _app()
        hooks = _Hooks()
        harness = RAGHarness(HarnessSettings(request_timeout_s=30), hooks)
        context = RAGRequestContext("typed-id", "text", "en", 10.0, "private typed query")
        with patch("app.services.rag_harness.run_text_rag", return_value=_result("text", retry_count=2)) as shared_service, patch(
            "app.services.rag_harness.time.perf_counter", return_value=10.1
        ):
            outcome = harness.execute(context, app)
        self.assertIsInstance(outcome, HarnessSuccess)
        shared_service.assert_called_once_with("private typed query", "en", "text", app)
        self.assertEqual(outcome.request_id, "typed-id")
        self.assertIsNone(outcome.total_voice_pipeline_ms)
        self.assertEqual(hooks.events, [("pre", "typed-id"), ("post", "text")])
        snapshot = app.state.diagnostics_registry.snapshot()
        self.assertEqual(snapshot["requests"]["successful"], 1)
        self.assertEqual(snapshot["requests"]["failed"], 0)
        self.assertEqual(snapshot["requests"]["last_language"], "en")
        self.assertNotIn("private typed query", str(snapshot))

    def test_voice_after_stt_uses_same_service_and_preserves_stt_outside_rag(self) -> None:
        app = _app()
        harness = RAGHarness(HarnessSettings(request_timeout_s=30))
        context = RAGRequestContext("voice-id", "voice", "hi", 20.0).with_query("transcribed query", 400.0)
        with patch("app.services.rag_harness.run_text_rag", return_value=_result("voice")) as shared_service, patch(
            # ``time`` is a shared module: deterministic pre/post guardrails now
            # consume four local clock reads before voice-total/deadline timing.
            "app.services.rag_harness.time.perf_counter", side_effect=(20.0, 20.0, 20.0, 20.0, 20.7, 20.7)
        ):
            outcome = harness.execute(context, app)
        self.assertIsInstance(outcome, HarnessSuccess)
        shared_service.assert_called_once_with("transcribed query", "hi", "voice", app)
        self.assertEqual(outcome.result.latency.rag_total_ms, 7.0)
        self.assertEqual(outcome.context.stt_latency_ms, 400.0)
        self.assertAlmostEqual(outcome.total_voice_pipeline_ms, 700.0, delta=0.001)

    def test_failure_is_normalized_and_counted_once_without_query_retention(self) -> None:
        app = _app()
        harness = RAGHarness(HarnessSettings(request_timeout_s=30))
        context = RAGRequestContext("failure-id", "text", "hi", 10.0, "private query")
        with patch("app.services.rag_harness.run_text_rag", side_effect=RuntimeError("internal detail private query")) as shared_service:
            outcome = harness.execute(context, app)
        self.assertIsInstance(outcome, HarnessFailure)
        self.assertEqual(outcome.error_category, "internal_error")
        self.assertEqual(outcome.status_code, 500)
        self.assertEqual(outcome.error_code, "RAG_INTERNAL_ERROR")
        shared_service.assert_called_once()
        snapshot = app.state.diagnostics_registry.snapshot()
        self.assertEqual(snapshot["requests"]["successful"], 0)
        self.assertEqual(snapshot["requests"]["failed"], 1)
        self.assertNotIn("private query", str(snapshot))

    def test_timeout_maps_safely_without_a_broad_retry(self) -> None:
        app = _app()
        harness = RAGHarness(HarnessSettings(request_timeout_s=1.0))
        context = RAGRequestContext("timeout-id", "text", "hi", 0.0, "query")
        with patch("app.services.rag_harness.run_text_rag", return_value=_result()) as shared_service, patch(
            "app.services.rag_harness.time.perf_counter", return_value=2.0
        ):
            outcome = harness.execute(context, app)
        self.assertIsInstance(outcome, HarnessFailure)
        self.assertEqual(outcome.error_category, "timeout")
        self.assertEqual(outcome.status_code, 504)
        self.assertEqual(outcome.error_code, "RAG_TIMEOUT")
        shared_service.assert_called_once()
        snapshot = app.state.diagnostics_registry.snapshot()
        self.assertEqual(snapshot["requests"]["failed"], 1)
        self.assertEqual(snapshot["requests"]["last_error_category"], "timeout")

    def test_validation_status_is_preserved_with_a_safe_error_body(self) -> None:
        class UploadValidationError(ValueError):
            status_code = 415

        outcome = RAGHarness(HarnessSettings()).fail(
            RAGRequestContext("validation-id", "voice", "hi", 1.0),
            _app(),
            UploadValidationError("no user data is retained"),
        )
        self.assertEqual(outcome.status_code, 415)
        self.assertEqual(outcome.error_code, "INVALID_REQUEST")
        self.assertNotIn("no user data", outcome.message)

    def test_stt_timeout_uses_timeout_category_for_diagnostics(self) -> None:
        class STTTimeoutError(Exception):
            status_code = 504

        app = _app()
        outcome = RAGHarness(HarnessSettings()).fail(
            RAGRequestContext("stt-timeout", "voice", "hi", 1.0), app, STTTimeoutError()
        )
        self.assertEqual(outcome.error_category, "timeout")
        self.assertEqual(app.state.diagnostics_registry.snapshot()["requests"]["last_error_category"], "timeout")


if __name__ == "__main__":
    unittest.main()
