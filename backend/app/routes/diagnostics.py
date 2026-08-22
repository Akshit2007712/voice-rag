"""Cheap public probes and an authenticated, in-memory operator report."""

from __future__ import annotations

import os
from typing import Any

from fastapi import APIRouter, Depends, Request, Response, status

from app.rag.bilingual_cloud_runtime import EXPECTED_BILINGUAL_POINT_COUNT
from app.schemas.diagnostics import DiagnosticsResponse, HealthResponse, ReadyResponse
from app.services.benchmark_latency import pending_benchmark_latency
from app.services.diagnostics import get_diagnostics_registry
from app.services.diagnostics_auth import require_diagnostics_access


router = APIRouter(tags=["operations"])
_REQUIRED_READINESS = (
    "qdrant_ready", "collection_verified", "payload_indexes_verified",
    "e5_loaded", "e5_warmed", "bm25_hi_ready", "bm25_en_ready",
    "hybrid_workers_ready", "application_ready",
)


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Process liveness probe",
    description="Cheap process liveness only. It does not call E5, Qdrant, BM25, or STT.",
)
async def health() -> HealthResponse:
    """Liveness only: no state reads, dependency initialization, or I/O."""
    return HealthResponse(status="ok")


@router.get(
    "/ready",
    response_model=ReadyResponse,
    summary="Deployment readiness probe",
    description="Returns ready only after startup has verified Cloud retrieval and initialized persistent RAG resources.",
)
async def ready(request: Request, response: Response) -> ReadyResponse:
    """Read the startup-populated readiness map without touching dependencies."""
    readiness = getattr(request.app.state, "rag_readiness", {})
    is_ready = all(readiness.get(component) is True for component in _REQUIRED_READINESS)
    if not is_ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return ReadyResponse(ready=is_ready)


@router.get(
    "/diagnostics",
    response_model=DiagnosticsResponse,
    dependencies=[Depends(require_diagnostics_access)],
    summary="Protected operator diagnostics",
    description="Authenticated operator-only in-memory diagnostics. Browser clients must not call this endpoint.",
    responses={401: {"description": "Missing or invalid diagnostics credential."}, 503: {"description": "Diagnostics credential is not configured."}},
)
async def diagnostics(request: Request) -> DiagnosticsResponse:
    """Return authenticated in-memory state only; this endpoint makes no network calls."""
    readiness = _readiness(request.app)
    registry_snapshot = get_diagnostics_registry(request.app).snapshot()
    runtime = getattr(request.app.state, "rag_runtime", None)
    store = getattr(runtime, "vector_store", None)
    verification = getattr(request.app.state, "diagnostics_verification", {})
    benchmark = getattr(request.app.state, "benchmark_latency", pending_benchmark_latency())
    return DiagnosticsResponse(
        status="ready" if readiness["application_ready"] else "not_ready",
        uptime_s=registry_snapshot["uptime_s"],
        readiness=readiness,
        e5={
            "loaded": readiness["e5_loaded"],
            "warmed": readiness["e5_warmed"],
            "device": _safe_device(runtime) if readiness["e5_loaded"] else None,
        },
        qdrant={
            "mode": "remote",
            "verified_at_startup": bool(verification.get("verified_at_startup", False)),
            "collection_verified": readiness["collection_verified"],
            "expected_point_count": verification.get("expected_point_count", _expected_point_count()),
            "verified_point_count": verification.get("verified_point_count"),
            "vector_size": verification.get("vector_size"),
            "language_index_verified": bool(verification.get("language_index_verified", False)),
            "target_lang_index_verified": bool(verification.get("target_lang_index_verified", False)),
            "retry_count": int(getattr(store, "qdrant_retry_count", 0)),
            "failed_request_count": int(getattr(store, "qdrant_failed_request_count", 0)),
        },
        bm25={"hi_ready": readiness["bm25_hi_ready"], "en_ready": readiness["bm25_en_ready"]},
        hybrid={"workers_ready": readiness["hybrid_workers_ready"]},
        requests=registry_snapshot["requests"],
        last_latency=registry_snapshot["last_latency"],
        guardrails=registry_snapshot["guardrails"],
        benchmark={
            "status": "validated" if benchmark.p50_ms is not None else "pending",
            "p50_ms": benchmark.p50_ms,
            "p70_ms": benchmark.p70_ms,
            "p100_ms": benchmark.p100_ms,
            "sample_count": benchmark.sample_count,
            "benchmark_scope": benchmark.benchmark_scope if benchmark.p50_ms is not None else None,
            "benchmark_environment": benchmark.benchmark_environment if benchmark.p50_ms is not None else None,
        },
    )


def _readiness(app: Any) -> dict[str, bool]:
    current = getattr(app.state, "rag_readiness", {})
    return {component: current.get(component) is True for component in _REQUIRED_READINESS}


def _expected_point_count() -> int:
    value = os.getenv("QDRANT_EXPECTED_POINT_COUNT", str(EXPECTED_BILINGUAL_POINT_COUNT))
    try:
        return int(value)
    except ValueError:
        return EXPECTED_BILINGUAL_POINT_COUNT


def _safe_device(runtime: object | None) -> str | None:
    device = getattr(getattr(runtime, "embedder", None), "device", None)
    return str(device) if isinstance(device, str) else None
