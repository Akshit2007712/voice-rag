"""One-shot request orchestration around the frozen text-RAG service."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, replace
from typing import Literal, Protocol

from app.services.diagnostics import error_category, get_diagnostics_registry
from app.services.guardrails import DeterministicGuardrails, GuardrailDecision, GuardrailEvaluation, no_answer_result
from app.services.text_rag import TextRagResult, run_text_rag


InputMode = Literal["text", "voice"]


@dataclass(frozen=True)
class HarnessSettings:
    """Startup-loaded, conservative one-shot deadline configuration."""

    request_timeout_s: float = 30.0

    @classmethod
    def from_environment(cls) -> "HarnessSettings":
        """Load configuration once during application startup, not per request."""
        value = os.getenv("RAG_REQUEST_TIMEOUT_S", "30")
        try:
            timeout = float(value)
        except ValueError as error:
            raise RuntimeError("RAG_REQUEST_TIMEOUT_S must be a positive number") from error
        if timeout <= 0:
            raise RuntimeError("RAG_REQUEST_TIMEOUT_S must be a positive number")
        return cls(request_timeout_s=timeout)


@dataclass(frozen=True)
class RAGRequestContext:
    """Transient one-shot context; query text is never copied into diagnostics."""

    request_id: str
    input_mode: InputMode
    language: str
    started_at: float
    query_text: str | None = None
    stt_latency_ms: float | None = None

    def with_query(self, query_text: str, stt_latency_ms: float | None = None) -> "RAGRequestContext":
        """Attach usable text after the voice adapter completes STT."""
        return replace(self, query_text=query_text, stt_latency_ms=stt_latency_ms)


class GuardrailHooks(Protocol):
    """Local guardrail hooks around the unchanged shared text-RAG service."""

    def pre_rag(self, context: RAGRequestContext) -> GuardrailEvaluation | None: ...

    def post_retrieval(self, context: RAGRequestContext, result: TextRagResult) -> GuardrailEvaluation | None: ...


class NoOpGuardrailHooks:
    """Preserve today’s behavior while making guardrail ownership explicit."""

    def pre_rag(self, context: RAGRequestContext) -> None:
        return None

    def post_retrieval(self, context: RAGRequestContext, result: TextRagResult) -> None:
        return None


@dataclass(frozen=True)
class HarnessSuccess:
    """A completed frozen-RAG result with request-lifecycle timings."""

    request_id: str
    context: RAGRequestContext
    result: TextRagResult
    total_voice_pipeline_ms: float | None


@dataclass(frozen=True)
class HarnessFailure:
    """Safe failure mapping with no exception object or message retained."""

    request_id: str
    error_category: str
    status_code: int
    error_code: str
    message: str
    retryable: bool


HarnessOutcome = HarnessSuccess | HarnessFailure


class RAGHarness:
    """Own one-shot execution policy, diagnostics, and safe failure normalization."""

    def __init__(self, settings: HarnessSettings, hooks: GuardrailHooks | None = None) -> None:
        self.settings = settings
        self.hooks = hooks or DeterministicGuardrails()

    def execute(self, context: RAGRequestContext, app) -> HarnessOutcome:
        """Invoke the shared service once; no retries or retrieval internals live here."""
        if isinstance(context.language, str):
            context = replace(context, language=context.language.strip().lower())
        if context.query_text is None:
            return self.fail(context, app, ValueError("query text is required"))
        pre_guardrail: GuardrailEvaluation | None = None
        post_guardrail: GuardrailEvaluation | None = None
        try:
            pre_guardrail = self.hooks.pre_rag(context)
            if pre_guardrail is not None and not pre_guardrail.decision.allowed:
                return self._pre_guardrail_failure(context, app, pre_guardrail)
            result = run_text_rag(context.query_text, context.language, context.input_mode, app)
            post_guardrail = self.hooks.post_retrieval(context, result)
            if post_guardrail is not None and not post_guardrail.decision.allowed:
                result = no_answer_result(result)
            total_voice_pipeline_ms = self._voice_total_ms(context)
            if (time.perf_counter() - context.started_at) > self.settings.request_timeout_s:
                return self.fail(
                    context,
                    app,
                    TimeoutError("one-shot request deadline exceeded"),
                    guardrail_metadata=_guardrail_metadata(pre_guardrail, post_guardrail),
                )
        except Exception as error:
            return self.fail(context, app, error, guardrail_metadata=_guardrail_metadata(pre_guardrail, post_guardrail))
        latency = dict(result.latency.__dict__)
        latency.update(stt_ms=context.stt_latency_ms, total_voice_pipeline_ms=total_voice_pipeline_ms)
        get_diagnostics_registry(app).record_success(
            context.request_id,
            context.input_mode,
            context.language,
            latency,
            qdrant_retry_count=result.qdrant_retry_count,
            guardrail_metadata=_guardrail_metadata(pre_guardrail, post_guardrail),
        )
        return HarnessSuccess(context.request_id, context, result, total_voice_pipeline_ms)

    def _pre_guardrail_failure(
        self,
        context: RAGRequestContext,
        app,
        evaluation: GuardrailEvaluation,
    ) -> HarnessFailure:
        """Reject locally before retrieval, retaining only a stable guardrail code."""
        decision = evaluation.decision
        code = decision.code or "QUERY_INVALID"
        get_diagnostics_registry(app).record_failure_category(
            context.request_id,
            context.input_mode,
            context.language,
            decision.diagnostics_category or "validation_error",
            guardrail_metadata=_guardrail_metadata(evaluation, None),
        )
        return HarnessFailure(
            context.request_id,
            decision.diagnostics_category or "validation_error",
            422,
            code,
            decision.user_safe_message or "The request is invalid.",
            False,
        )

    def fail(
        self,
        context: RAGRequestContext,
        app,
        error: BaseException,
        guardrail_metadata: dict[str, object] | None = None,
    ) -> HarnessFailure:
        """Normalize and count a one-shot failure exactly once."""
        category = error_category(error)
        failure = _map_failure(context.request_id, category, error)
        get_diagnostics_registry(app).record_failure_category(
            context.request_id,
            context.input_mode,
            context.language,
            failure.error_category,
            guardrail_metadata=guardrail_metadata,
        )
        return failure

    @staticmethod
    def _voice_total_ms(context: RAGRequestContext) -> float | None:
        if context.input_mode != "voice":
            return None
        return (time.perf_counter() - context.started_at) * 1_000


def get_rag_harness(app) -> RAGHarness:
    """Return the startup-created harness; focused tests receive a safe fallback."""
    harness = getattr(app.state, "rag_harness", None)
    if harness is None:
        harness = RAGHarness(HarnessSettings())
        app.state.rag_harness = harness
    return harness


def _map_failure(request_id: str, category: str, error: BaseException) -> HarnessFailure:
    """Map only safe categories/status facts to a stable user-facing contract."""
    if category == "timeout":
        return HarnessFailure(request_id, category, 504, "RAG_TIMEOUT", "The request could not be completed in time.", True)
    if category == "validation_error":
        status_code = getattr(error, "status_code", 422)
        status_code = status_code if isinstance(status_code, int) and 400 <= status_code < 500 else 422
        return HarnessFailure(request_id, category, status_code, "INVALID_REQUEST", "The request is invalid.", False)
    if category == "stt_error":
        status_code = getattr(error, "status_code", 502)
        if status_code == 504:
            return HarnessFailure(request_id, "timeout", 504, "RAG_TIMEOUT", "The request could not be completed in time.", True)
        status_code = status_code if status_code in {422, 502, 503} else 502
        err_msg = str(error) if str(error) and not str(error).startswith("<") else "Voice transcription is currently unavailable."
        return HarnessFailure(request_id, category, status_code, "STT_UNAVAILABLE", err_msg, status_code == 503)

    if category in {"qdrant_connection_error", "qdrant_query_error"}:
        return HarnessFailure(request_id, category, 503, "RAG_UNAVAILABLE", "The answer service is temporarily unavailable.", True)
    if category == "internal_error" and isinstance(error, RuntimeError) and "unavailable" in str(error).lower():
        return HarnessFailure(request_id, category, 503, "RAG_UNAVAILABLE", "The answer service is temporarily unavailable.", True)
    return HarnessFailure(request_id, category, 500, "RAG_INTERNAL_ERROR", "The request could not be completed.", False)


def _guardrail_metadata(
    pre_guardrail: GuardrailEvaluation | None,
    post_guardrail: GuardrailEvaluation | None,
) -> dict[str, object] | None:
    """Return bounded safe observability fields without request/evidence content."""
    if pre_guardrail is None and post_guardrail is None:
        return None
    final_decision: GuardrailDecision | None = None
    if post_guardrail is not None and not post_guardrail.decision.allowed:
        final_decision = post_guardrail.decision
    elif pre_guardrail is not None and not pre_guardrail.decision.allowed:
        final_decision = pre_guardrail.decision
    return {
        "pre_guardrail_ms": pre_guardrail.elapsed_ms if pre_guardrail is not None else None,
        "post_guardrail_ms": post_guardrail.elapsed_ms if post_guardrail is not None else None,
        "guardrail_code": final_decision.code if final_decision is not None else None,
    }
