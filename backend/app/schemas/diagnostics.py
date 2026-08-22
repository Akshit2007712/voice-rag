"""Infrastructure-only schemas; intentionally separate from user API responses."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: Literal["ok"]


class ReadyResponse(BaseModel):
    ready: bool


class ReadinessDiagnostics(BaseModel):
    qdrant_ready: bool
    collection_verified: bool
    payload_indexes_verified: bool
    e5_loaded: bool
    e5_warmed: bool
    bm25_hi_ready: bool
    bm25_en_ready: bool
    hybrid_workers_ready: bool
    application_ready: bool


class E5Diagnostics(BaseModel):
    loaded: bool
    warmed: bool
    device: str | None


class QdrantDiagnostics(BaseModel):
    mode: Literal["remote"]
    verified_at_startup: bool
    collection_verified: bool
    expected_point_count: int | None = Field(default=None, ge=0)
    verified_point_count: int | None = Field(default=None, ge=0)
    vector_size: int | None = Field(default=None, ge=0)
    language_index_verified: bool
    target_lang_index_verified: bool
    retry_count: int = Field(ge=0)
    failed_request_count: int = Field(ge=0)


class Bm25Diagnostics(BaseModel):
    hi_ready: bool
    en_ready: bool


class HybridDiagnostics(BaseModel):
    workers_ready: bool


class RequestDiagnostics(BaseModel):
    successful: int = Field(ge=0)
    failed: int = Field(ge=0)
    last_request_id: str | None
    last_input_mode: str | None
    last_language: str | None
    last_success: bool | None
    last_error_category: str | None


class LastLatencyDiagnostics(BaseModel):
    embedding_ms: float | None = Field(default=None, ge=0)
    qdrant_ms: float | None = Field(default=None, ge=0)
    bm25_ms: float | None = Field(default=None, ge=0)
    post_embedding_parallel_ms: float | None = Field(default=None, ge=0)
    rrf_ms: float | None = Field(default=None, ge=0)
    maturity_ms: float | None = Field(default=None, ge=0)
    composer_ms: float | None = Field(default=None, ge=0)
    rag_total_ms: float | None = Field(default=None, ge=0)
    stt_ms: float | None = Field(default=None, ge=0)
    total_voice_pipeline_ms: float | None = Field(default=None, ge=0)


class GuardrailDiagnostics(BaseModel):
    """Safe counters/timings for local deterministic guardrails only."""

    allowed: int = Field(ge=0)
    pre_rejected: int = Field(ge=0)
    insufficient_evidence: int = Field(ge=0)
    last_code: str | None
    last_pre_guardrail_ms: float | None = Field(default=None, ge=0)
    last_post_guardrail_ms: float | None = Field(default=None, ge=0)


class BenchmarkDiagnostics(BaseModel):
    status: Literal["pending", "validated"]
    p50_ms: float | None = Field(default=None, ge=0)
    p70_ms: float | None = Field(default=None, ge=0)
    p100_ms: float | None = Field(default=None, ge=0)
    sample_count: int | None = Field(default=None, ge=1)
    benchmark_scope: str | None
    benchmark_environment: str | None


class DiagnosticsResponse(BaseModel):
    status: Literal["ready", "not_ready"]
    uptime_s: float = Field(ge=0)
    readiness: ReadinessDiagnostics
    e5: E5Diagnostics
    qdrant: QdrantDiagnostics
    bm25: Bm25Diagnostics
    hybrid: HybridDiagnostics
    requests: RequestDiagnostics
    last_latency: LastLatencyDiagnostics
    guardrails: GuardrailDiagnostics
    benchmark: BenchmarkDiagnostics
