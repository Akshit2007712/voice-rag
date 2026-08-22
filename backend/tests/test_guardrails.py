"""Focused deterministic guardrail coverage around the unchanged one-shot RAG path."""

import unittest
import time
from types import SimpleNamespace
from unittest.mock import patch

from app.rag.generation.answer_composer import AnswerEvidence, ComposedAnswer, NO_ANSWER_TEXT
from app.services.guardrails import DeterministicGuardrails
from app.services.rag_harness import HarnessFailure, HarnessSettings, HarnessSuccess, RAGHarness, RAGRequestContext
from app.services.text_rag import RagLatency, TextRagResult


def _app() -> SimpleNamespace:
    return SimpleNamespace(state=SimpleNamespace())


def _result(answer: ComposedAnswer, input_mode: str = "text") -> TextRagResult:
    return TextRagResult(
        query="private query",
        answer=answer,
        input_mode=input_mode,
        latency=RagLatency(1.0, 2.0, 3.0, 4.0, 5.0, 0.0, 6.0, 7.0),
    )


def _accepted_answer() -> ComposedAnswer:
    return ComposedAnswer(
        "grounded answer",
        [AnswerEvidence(1, 2, 0, 0.9, "private evidence")],
        0.9,
        0.1,
        False,
    )


class DeterministicGuardrailTests(unittest.TestCase):
    def setUp(self) -> None:
        self.app = _app()
        self.harness = RAGHarness(HarnessSettings(request_timeout_s=1_000))

    def _execute(self, query: str, language: str = "hi", input_mode: str = "text"):
        return self.harness.execute(
            RAGRequestContext("request-id", input_mode, language, time.perf_counter(), query),
            self.app,
        )

    def test_empty_whitespace_and_symbols_are_rejected_before_shared_rag(self) -> None:
        for query in ("", "   \t\n", "!!!", "---", "🙂🙂"):
            with self.subTest(query=query), patch("app.services.rag_harness.run_text_rag") as shared_rag:
                outcome = self._execute(query)
            self.assertIsInstance(outcome, HarnessFailure)
            self.assertEqual(outcome.status_code, 422)
            self.assertEqual(outcome.error_code, "QUERY_INVALID")
            shared_rag.assert_not_called()

    def test_short_hindi_and_english_queries_are_allowed(self) -> None:
        for query, language in (("AI?", "en"), ("GST?", "en"), ("CPU?", "en"), ("क्या?", "hi")):
            with self.subTest(query=query), patch(
                "app.services.rag_harness.run_text_rag", return_value=_result(_accepted_answer(), "text")
            ) as shared_rag:
                outcome = self._execute(query, language)
            self.assertIsInstance(outcome, HarnessSuccess)
            shared_rag.assert_called_once()

    def test_unsupported_language_and_oversized_typed_query_reject_without_retrieval(self) -> None:
        for query, language in (("question", "ta"), ("x" * 4_001, "en")):
            with self.subTest(language=language), patch("app.services.rag_harness.run_text_rag") as shared_rag:
                outcome = self._execute(query, language)
            self.assertIsInstance(outcome, HarnessFailure)
            self.assertEqual(outcome.status_code, 422)
            self.assertIn(outcome.error_code, {"UNSUPPORTED_LANGUAGE", "QUERY_INVALID"})
            shared_rag.assert_not_called()

    def test_no_evidence_becomes_existing_no_answer_without_another_retrieval(self) -> None:
        unsafe_answer = ComposedAnswer("unsafe answer", [], 0.9, 0.1, False)
        with patch("app.services.rag_harness.run_text_rag", return_value=_result(unsafe_answer)) as shared_rag:
            outcome = self._execute("valid question", "en")
        self.assertIsInstance(outcome, HarnessSuccess)
        self.assertTrue(outcome.result.answer.is_no_answer)
        self.assertEqual(outcome.result.answer.text, NO_ANSWER_TEXT)
        shared_rag.assert_called_once()
        snapshot = self.app.state.diagnostics_registry.snapshot()
        self.assertEqual(snapshot["guardrails"]["insufficient_evidence"], 1)
        self.assertEqual(snapshot["guardrails"]["last_code"], "INSUFFICIENT_EVIDENCE")

    def test_accepted_evidence_and_voice_text_equivalence_are_preserved(self) -> None:
        answer = _accepted_answer()
        with patch("app.services.rag_harness.run_text_rag", side_effect=[_result(answer, "text"), _result(answer, "voice")]) as shared_rag:
            typed = self._execute("same question", "en", "text")
            voice = self._execute("same question", "en", "voice")
        self.assertIsInstance(typed, HarnessSuccess)
        self.assertIsInstance(voice, HarnessSuccess)
        self.assertEqual(typed.result.answer, voice.result.answer)
        self.assertEqual(shared_rag.call_count, 2)
        snapshot = self.app.state.diagnostics_registry.snapshot()
        self.assertEqual(snapshot["guardrails"]["allowed"], 2)
        self.assertNotIn("same question", str(snapshot))
        self.assertNotIn("private evidence", str(snapshot))

    def test_guardrail_primitive_never_performs_network_or_model_work(self) -> None:
        guardrails = DeterministicGuardrails()
        result = guardrails.pre_rag(RAGRequestContext("id", "text", "hi", 0.0, "AI?"))
        self.assertTrue(result.decision.allowed)
        self.assertGreaterEqual(result.elapsed_ms, 0.0)


if __name__ == "__main__":
    unittest.main()
