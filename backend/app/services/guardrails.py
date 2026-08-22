"""Deterministic, local request guardrails for the one-shot RAG harness."""

from __future__ import annotations

import time
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from app.rag.generation.answer_composer import AnswerEvidence, ComposedAnswer, NO_ANSWER_TEXT
from app.rag.language_config import LANGUAGE_CONFIG
from app.services.text_rag import MAX_TYPED_QUERY_CHARS, TextRagResult

if TYPE_CHECKING:
    from app.services.rag_harness import RAGRequestContext


SUPPORTED_APPLICATION_LANGUAGES = frozenset(LANGUAGE_CONFIG)


@dataclass(frozen=True)
class GuardrailDecision:
    """A content-free guardrail decision safe to store in diagnostics."""

    allowed: bool
    code: str | None = None
    user_safe_message: str | None = None
    diagnostics_category: str | None = None


@dataclass(frozen=True)
class GuardrailEvaluation:
    """Decision plus local timing; neither field retains request content."""

    decision: GuardrailDecision
    elapsed_ms: float


_ALLOW = GuardrailDecision(True)
_QUERY_INVALID = GuardrailDecision(False, "QUERY_INVALID", "The request is invalid.", "validation_error")
_UNSUPPORTED_LANGUAGE = GuardrailDecision(
    False,
    "UNSUPPORTED_LANGUAGE",
    "The requested language is not supported.",
    "validation_error",
)
_INSUFFICIENT_EVIDENCE = GuardrailDecision(
    False,
    "INSUFFICIENT_EVIDENCE",
    None,
    "insufficient_evidence",
)


class DeterministicGuardrails:
    """Apply cheap syntax and evidence checks without extra model or network calls."""

    def pre_rag(self, context: "RAGRequestContext") -> GuardrailEvaluation:
        """Reject invalid requests before E5, Qdrant, and BM25 are called."""
        started_at = time.perf_counter()
        query = context.query_text
        if context.language not in SUPPORTED_APPLICATION_LANGUAGES:
            decision = _UNSUPPORTED_LANGUAGE
        elif not isinstance(query, str):
            decision = _QUERY_INVALID
        else:
            normalized = query.strip()
            if not normalized:
                decision = _QUERY_INVALID
            elif context.input_mode == "text" and len(" ".join(normalized.split())) > MAX_TYPED_QUERY_CHARS:
                decision = _QUERY_INVALID
            elif not any(character.isalnum() for character in normalized):
                # This rejects punctuation/emoji-only input while allowing short
                # useful queries such as AI?, GST?, CPU?, and Hindi text.
                decision = _QUERY_INVALID
            else:
                decision = _ALLOW
        return GuardrailEvaluation(decision, (time.perf_counter() - started_at) * 1_000)

    def post_retrieval(self, _context: "RAGRequestContext", result: TextRagResult) -> GuardrailEvaluation:
        """Recognize no-evidence/malformed final output without another retrieval pass."""
        started_at = time.perf_counter()
        answer = result.answer
        if not isinstance(answer, ComposedAnswer):
            # The production composer always returns ComposedAnswer. Keep focused
            # test doubles compatible while avoiding unsafe introspection.
            decision = _ALLOW
        elif answer.is_no_answer:
            decision = _INSUFFICIENT_EVIDENCE
        elif not isinstance(answer.text, str) or not answer.text.strip():
            decision = _INSUFFICIENT_EVIDENCE
        elif not answer.evidence or not all(isinstance(item, AnswerEvidence) for item in answer.evidence):
            decision = _INSUFFICIENT_EVIDENCE
        else:
            decision = _ALLOW
        return GuardrailEvaluation(decision, (time.perf_counter() - started_at) * 1_000)


def no_answer_result(result: TextRagResult) -> TextRagResult:
    """Replace only malformed final output with the existing grounded no-answer result."""
    answer = result.answer
    if isinstance(answer, ComposedAnswer) and answer.is_no_answer:
        return result
    latency_ms = float(getattr(answer, "latency_ms", 0.0) or 0.0)
    return replace(
        result,
        answer=ComposedAnswer(NO_ANSWER_TEXT, [], None, max(0.0, latency_ms), True),
    )
