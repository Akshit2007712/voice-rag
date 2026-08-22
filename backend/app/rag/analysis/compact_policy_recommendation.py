"""Threshold-free interpretation of compact-policy benchmark measurements."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


LABELED_RELEVANCE_METRICS = (
    "expected_evidence_in_top_k_rate",
    "top1_expected_evidence_rate",
    "correct_evidence_family_rate",
)
LOCAL_LATENCY_PERCENTILES = ("p50", "p70", "p95", "p100")


def _as_float(value: object, name: str) -> float:
    if not isinstance(value, (int, float)):
        raise ValueError(f"Benchmark metric {name!r} must be numeric")
    return float(value)


def policy_pareto_dominates(
    candidate: Mapping[str, Any],
    other: Mapping[str, Any],
) -> bool:
    """Return whether candidate is no worse on every decision dimension.

    This intentionally does not treat zero-selected self-query coverage as
    relevance: those MSMARCO rows have no selected-passage ground truth.
    """
    candidate_values = [_as_float(candidate[key], key) for key in LABELED_RELEVANCE_METRICS]
    other_values = [_as_float(other[key], key) for key in LABELED_RELEVANCE_METRICS]
    candidate_latency = candidate.get("latency_ms")
    other_latency = other.get("latency_ms")
    if not isinstance(candidate_latency, Mapping) or not isinstance(other_latency, Mapping):
        raise ValueError("Benchmark metrics must include latency_ms mappings")

    candidate_chunks = _as_float(candidate.get("chunk_count"), "chunk_count")
    other_chunks = _as_float(other.get("chunk_count"), "chunk_count")
    candidate_latencies = [_as_float(candidate_latency[key], f"latency_ms.{key}") for key in LOCAL_LATENCY_PERCENTILES]
    other_latencies = [_as_float(other_latency[key], f"latency_ms.{key}") for key in LOCAL_LATENCY_PERCENTILES]

    no_worse = (
        all(value >= baseline for value, baseline in zip(candidate_values, other_values, strict=True))
        and candidate_chunks <= other_chunks
        and all(value <= baseline for value, baseline in zip(candidate_latencies, other_latencies, strict=True))
    )
    strictly_better = (
        any(value > baseline for value, baseline in zip(candidate_values, other_values, strict=True))
        or candidate_chunks < other_chunks
        or any(value < baseline for value, baseline in zip(candidate_latencies, other_latencies, strict=True))
    )
    return no_worse and strictly_better


def recommend_compact_policy(results: Mapping[str, Mapping[str, Any]]) -> str:
    """Recommend the sole Pareto-dominant policy, else report a real trade-off."""
    if not results:
        raise ValueError("At least one policy result is required")
    dominant = [
        policy
        for policy, metrics in results.items()
        if all(policy == other_policy or policy_pareto_dominates(metrics, other_metrics)
               for other_policy, other_metrics in results.items())
    ]
    if len(dominant) == 1:
        policy_prefix = "_".join(dominant[0].split("_", maxsplit=2)[:2])
        return f"{policy_prefix}_RECOMMENDED"
    return "COMPACT_POLICY_TRADE_OFF_UNRESOLVED"


def recommendation_interpretation(
    results: Mapping[str, Mapping[str, Any]],
    recommendation: str,
) -> dict[str, object]:
    """Attach non-gating coverage and deployment-latency caveats to the report."""
    return {
        "method": "pareto_dominance_without_acceptance_thresholds",
        "labeled_relevance_metrics": list(LABELED_RELEVANCE_METRICS),
        "relative_engineering_metrics": {
            "chunk_count": "smaller_is_better",
            "local_embedded_qdrant_latency_ms": list(LOCAL_LATENCY_PERCENTILES),
        },
        "zero_selected_coverage": {
            policy: metrics.get("zero_selected", {}) for policy, metrics in results.items()
        },
        "zero_selected_coverage_is_labeled_relevance": False,
        "zero_selected_coverage_caveat": recommendation == "POLICY_A_RECOMMENDED",
        "zero_selected_coverage_caveat_reason": (
            "Policy A intentionally excludes passages from rows with no MSMARCO "
            "selected evidence, so those rows do not receive self-query coverage."
            if recommendation == "POLICY_A_RECOMMENDED" else None
        ),
        "final_deployment_latency_validated": False,
        "local_latency_note": (
            "Embedded/local Qdrant query_points timings exclude E5 query embedding "
            "and are not final deployed end-to-end RAG latency."
        ),
    }
