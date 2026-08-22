"""Validated benchmark metadata for API responses, separate from live timings."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class BenchmarkLatency:
    """A completed deployment benchmark, or an explicit pending placeholder."""

    p50_ms: float | None
    p70_ms: float | None
    p100_ms: float | None
    sample_count: int | None
    benchmark_scope: str
    benchmark_environment: str

    def as_dict(self) -> dict[str, object]:
        """Return the stable API payload shape."""
        return asdict(self)


def pending_benchmark_latency() -> BenchmarkLatency:
    """Represent the absence of a validated deployed benchmark without inventing data."""
    return BenchmarkLatency(
        p50_ms=None,
        p70_ms=None,
        p100_ms=None,
        sample_count=None,
        benchmark_scope="final_bilingual_text_rag",
        benchmark_environment="pending",
    )


def load_benchmark_latency(backend_root: Path) -> BenchmarkLatency:
    """Load optional validated metadata from ``RAG_BENCHMARK_ARTIFACT``.

    The configured path may be project-relative. An unset variable is an explicit
    pending state; a configured but malformed artifact fails startup rather than
    exposing unverified percentile values as production facts.
    """
    configured_path = os.getenv("RAG_BENCHMARK_ARTIFACT")
    if not configured_path:
        return pending_benchmark_latency()
    artifact_path = Path(configured_path)
    if not artifact_path.is_absolute():
        artifact_path = backend_root / artifact_path
    try:
        payload: dict[str, Any] = json.loads(artifact_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"RAG_BENCHMARK_ARTIFACT could not be loaded: {artifact_path}") from error
    required = ("p50_ms", "p70_ms", "p100_ms", "sample_count", "benchmark_scope", "benchmark_environment")
    missing = [field for field in required if field not in payload]
    if missing:
        raise RuntimeError(f"RAG_BENCHMARK_ARTIFACT is missing fields: {', '.join(missing)}")
    values = {field: payload[field] for field in required}
    for field in ("p50_ms", "p70_ms", "p100_ms"):
        value = values[field]
        if value is not None and (not isinstance(value, (int, float)) or isinstance(value, bool) or value < 0):
            raise RuntimeError(f"RAG_BENCHMARK_ARTIFACT field {field} must be a non-negative number or null")
    sample_count = values["sample_count"]
    if sample_count is not None and (not isinstance(sample_count, int) or isinstance(sample_count, bool) or sample_count < 1):
        raise RuntimeError("RAG_BENCHMARK_ARTIFACT field sample_count must be a positive integer or null")
    for field in ("benchmark_scope", "benchmark_environment"):
        if not isinstance(values[field], str) or not values[field].strip():
            raise RuntimeError(f"RAG_BENCHMARK_ARTIFACT field {field} must be a non-empty string")
    return BenchmarkLatency(**values)
