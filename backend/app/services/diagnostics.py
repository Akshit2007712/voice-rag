"""Thread-safe, data-minimized operational metrics for production diagnostics."""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from threading import RLock
from typing import Any, Mapping

logger = logging.getLogger("uvicorn.error")
_LATENCY_FIELDS = (
    "embedding_ms",
    "qdrant_ms",
    "bm25_ms",
    "post_embedding_parallel_ms",
    "rrf_ms",
    "maturity_ms",
    "composer_ms",
    "rag_total_ms",
    "stt_ms",
    "total_voice_pipeline_ms",
)
_SUPPORTED_LANGUAGES = frozenset({"hi", "en"})
_GUARDRAIL_CODES = frozenset({"QUERY_INVALID", "UNSUPPORTED_LANGUAGE", "INSUFFICIENT_EVIDENCE"})


def new_request_id() -> str:
    """Create an opaque identifier that contains no request or user data."""
    return uuid.uuid4().hex


def safe_language(language: object) -> str | None:
    """Allow only the two configured application language codes in diagnostics."""
    return language if isinstance(language, str) and language in _SUPPORTED_LANGUAGES else None


def error_category(error: BaseException) -> str:
    """Classify errors without retaining their message, traceback, or payload."""
    name = type(error).__name__.lower()
    module = type(error).__module__.lower()
    message = str(error).lower()
    if "stt" in name or "sarvam" in name or "stt" in module:
        return "stt_error"
    status_code = getattr(error, "status_code", None)
    if isinstance(status_code, int):
        return "validation_error" if 400 <= status_code < 500 else "internal_error"
    if isinstance(error, (TimeoutError, asyncio.TimeoutError)):
        return "timeout"
    if "qdrant" in name or "qdrant" in module or "qdrant" in message:
        if any(term in message for term in ("connection", "transport", "dns", "timeout")):
            return "qdrant_connection_error"
        return "qdrant_query_error"
    if "bm25" in name or "bm25" in module:
        return "bm25_error"
    if "embedding" in name or "embedder" in name or "torch" in module:
        return "embedding_error"
    if "composer" in name or "answer_composer" in module:
        return "composition_error"
    if isinstance(error, ValueError):
        return "validation_error"
    return "retrieval_error" if "retriev" in name or "retriev" in module else "internal_error"


class DiagnosticsRegistry:
    """Keep bounded operational state only; it never stores request content."""

    def __init__(self) -> None:
        self._started_at = time.monotonic()
        self._lock = RLock()
        self._successful = 0
        self._failed = 0
        self._last_request_id: str | None = None
        self._last_input_mode: str | None = None
        self._last_language: str | None = None
        self._last_success: bool | None = None
        self._last_error_category: str | None = None
        self._last_latency: dict[str, float | None] = self._empty_latency()
        self._guardrail_allowed = 0
        self._guardrail_pre_rejected = 0
        self._guardrail_insufficient_evidence = 0
        self._last_guardrail_code: str | None = None
        self._last_pre_guardrail_ms: float | None = None
        self._last_post_guardrail_ms: float | None = None

    @staticmethod
    def _empty_latency() -> dict[str, float | None]:
        return {field: None for field in _LATENCY_FIELDS}

    @staticmethod
    def _safe_latency(latency: Mapping[str, object] | None) -> dict[str, float | None]:
        safe = DiagnosticsRegistry._empty_latency()
        if latency is None:
            return safe
        for field in _LATENCY_FIELDS:
            value = latency.get(field)
            if isinstance(value, (int, float)) and not isinstance(value, bool) and value >= 0:
                safe[field] = float(value)
        return safe

    @staticmethod
    def _safe_guardrail_metadata(metadata: Mapping[str, object] | None) -> dict[str, float | str | None]:
        safe: dict[str, float | str | None] = {
            "pre_guardrail_ms": None,
            "post_guardrail_ms": None,
            "guardrail_code": None,
        }
        if metadata is None:
            return safe
        for field in ("pre_guardrail_ms", "post_guardrail_ms"):
            value = metadata.get(field)
            if isinstance(value, (int, float)) and not isinstance(value, bool) and value >= 0:
                safe[field] = float(value)
        code = metadata.get("guardrail_code")
        if isinstance(code, str) and code in _GUARDRAIL_CODES:
            safe["guardrail_code"] = code
        return safe

    def _record_guardrail(self, metadata: Mapping[str, object] | None) -> None:
        safe = self._safe_guardrail_metadata(metadata)
        pre_guardrail_ms = safe["pre_guardrail_ms"]
        post_guardrail_ms = safe["post_guardrail_ms"]
        guardrail_code = safe["guardrail_code"]
        self._last_pre_guardrail_ms = pre_guardrail_ms if isinstance(pre_guardrail_ms, float) else None
        self._last_post_guardrail_ms = post_guardrail_ms if isinstance(post_guardrail_ms, float) else None
        self._last_guardrail_code = guardrail_code if isinstance(guardrail_code, str) else None
        code = self._last_guardrail_code
        if code == "INSUFFICIENT_EVIDENCE":
            self._guardrail_insufficient_evidence += 1
        elif code in {"QUERY_INVALID", "UNSUPPORTED_LANGUAGE"}:
            self._guardrail_pre_rejected += 1
        elif metadata is not None:
            self._guardrail_allowed += 1

    def record_success(
        self,
        request_id: str,
        input_mode: str,
        language: object,
        latency: Mapping[str, object],
        qdrant_retry_count: int = 0,
        guardrail_metadata: Mapping[str, object] | None = None,
    ) -> None:
        """Record one completed request using only safe operational fields."""
        safe_latency = self._safe_latency(latency)
        safe_guardrail = self._safe_guardrail_metadata(guardrail_metadata)
        safe_mode = input_mode if input_mode in {"text", "voice", "voice_stream"} else "unknown"
        with self._lock:
            self._successful += 1
            self._last_request_id = request_id
            self._last_input_mode = safe_mode
            self._last_language = safe_language(language)
            self._last_success = True
            self._last_error_category = None
            self._last_latency = safe_latency
            self._record_guardrail(safe_guardrail)
        self._log_completed(
            request_id, safe_mode, safe_language(language), True, safe_latency,
            qdrant_retry_count, None, safe_guardrail,
        )

    def record_failure(
        self,
        request_id: str,
        input_mode: str,
        language: object,
        error: BaseException,
        guardrail_metadata: Mapping[str, object] | None = None,
    ) -> None:
        """Record a failed request without retaining an exception message or data."""
        self.record_failure_category(request_id, input_mode, language, error_category(error), guardrail_metadata)

    def record_failure_category(
        self,
        request_id: str,
        input_mode: str,
        language: object,
        category: str,
        guardrail_metadata: Mapping[str, object] | None = None,
    ) -> None:
        """Record an already-normalized safe category without retaining an exception."""
        safe_guardrail = self._safe_guardrail_metadata(guardrail_metadata)
        safe_mode = input_mode if input_mode in {"text", "voice", "voice_stream"} else "unknown"
        with self._lock:
            self._failed += 1
            self._last_request_id = request_id
            self._last_input_mode = safe_mode
            self._last_language = safe_language(language)
            self._last_success = False
            self._last_error_category = category
            self._last_latency = self._empty_latency()
            self._record_guardrail(safe_guardrail)
        self._log_completed(
            request_id, safe_mode, safe_language(language), False, self._empty_latency(),
            0, category, safe_guardrail,
        )

    def snapshot(self) -> dict[str, object]:
        """Return a copy suitable for serialization without mutating counters."""
        with self._lock:
            return {
                "uptime_s": time.monotonic() - self._started_at,
                "requests": {
                    "successful": self._successful,
                    "failed": self._failed,
                    "last_request_id": self._last_request_id,
                    "last_input_mode": self._last_input_mode,
                    "last_language": self._last_language,
                    "last_success": self._last_success,
                    "last_error_category": self._last_error_category,
                },
                "last_latency": dict(self._last_latency),
                "guardrails": {
                    "allowed": self._guardrail_allowed,
                    "pre_rejected": self._guardrail_pre_rejected,
                    "insufficient_evidence": self._guardrail_insufficient_evidence,
                    "last_code": self._last_guardrail_code,
                    "last_pre_guardrail_ms": self._last_pre_guardrail_ms,
                    "last_post_guardrail_ms": self._last_post_guardrail_ms,
                },
            }

    @staticmethod
    def _log_completed(
        request_id: str,
        input_mode: str,
        language: str | None,
        success: bool,
        latency: Mapping[str, float | None],
        qdrant_retry_count: int,
        category: str | None,
        guardrail: Mapping[str, float | str | None],
    ) -> None:
        """Emit one normal, structured, content-free request completion log."""
        logger.info(
            "RAG_REQUEST request_id=%s input_mode=%s language=%s success=%s "
            "embedding_ms=%s qdrant_ms=%s bm25_ms=%s post_embedding_parallel_ms=%s "
            "rrf_ms=%s maturity_ms=%s composer_ms=%s rag_total_ms=%s stt_ms=%s "
            "total_voice_pipeline_ms=%s qdrant_retry_count_for_request=%d error_category=%s "
            "pre_guardrail_ms=%s post_guardrail_ms=%s guardrail_code=%s",
            request_id,
            input_mode,
            language,
            success,
            latency["embedding_ms"], latency["qdrant_ms"], latency["bm25_ms"],
            latency["post_embedding_parallel_ms"], latency["rrf_ms"], latency["maturity_ms"],
            latency["composer_ms"], latency["rag_total_ms"], latency["stt_ms"],
            latency["total_voice_pipeline_ms"], max(0, int(qdrant_retry_count)), category,
            guardrail["pre_guardrail_ms"], guardrail["post_guardrail_ms"], guardrail["guardrail_code"],
        )


def get_diagnostics_registry(app: Any) -> DiagnosticsRegistry:
    """Obtain the startup registry, with a safe fallback for focused route tests."""
    registry = getattr(app.state, "diagnostics_registry", None)
    if registry is None:
        registry = DiagnosticsRegistry()
        app.state.diagnostics_registry = registry
    return registry
