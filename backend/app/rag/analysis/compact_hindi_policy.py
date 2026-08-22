"""Deterministic passage policies for compact Hindi corpus analysis only."""

from __future__ import annotations

from collections.abc import Mapping

from app.rag.ingestion.preprocessor import RetrievalDocument, preprocess_msmarco_xi_record


POLICY_A = "POLICY_A_SELECTED_ONLY"
POLICY_B = "POLICY_B_SELECTED_PLUS_1"
POLICY_C = "POLICY_C_SELECTED_PLUS_2"
POLICY_D = "POLICY_D_SELECTED_PLUS_QUERY_COVERAGE"
POLICY_E = "POLICY_E_SELECTED_PLUS_1_WITH_ZERO_SELECTED_FALLBACK"
POLICIES = (POLICY_A, POLICY_B, POLICY_C, POLICY_D, POLICY_E)


def is_selected(value: object) -> bool:
    """Interpret the MSMARCO-XI relevance labels without changing their values."""
    return value is True or value == 1


def selected_documents_for_policy(record: Mapping[str, object], policy: str) -> list[RetrievalDocument]:
    """Choose non-empty prepared passages according to one documented policy."""
    if policy not in POLICIES:
        raise ValueError(f"Unsupported compact-corpus policy: {policy}")
    documents = preprocess_msmarco_xi_record(record)
    selected = [document for document in documents if is_selected(document.metadata["is_selected"])]
    non_selected = [document for document in documents if not is_selected(document.metadata["is_selected"])]
    if policy == POLICY_A:
        return selected
    if policy == POLICY_D:
        return selected or non_selected[:1]
    if policy in (POLICY_B, POLICY_E):
        return [*selected, *_deterministic_extras(selected, non_selected, 1)]
    return [*selected, *_deterministic_extras(selected, non_selected, 2)]


def _deterministic_extras(
    selected: list[RetrievalDocument],
    non_selected: list[RetrievalDocument],
    limit: int,
) -> list[RetrievalDocument]:
    """Prefer nearest unselected passage; otherwise preserve original order."""
    selected_indices = [int(document.metadata["passage_index"]) for document in selected]
    ordered = sorted(
        non_selected,
        key=lambda document: (
            min(abs(int(document.metadata["passage_index"]) - index) for index in selected_indices)
            if selected_indices
            else int(document.metadata["passage_index"]),
            int(document.metadata["passage_index"]),
        ),
    )
    return ordered[:limit]


def storage_estimate(chunks: int, bytes_per_point: float, storage_price_per_gb: float | None = None) -> dict[str, float | None]:
    """Return measured-overhead storage extrapolations, never raw-vector guesses."""
    estimated_gb = chunks * bytes_per_point / (1024**3)
    return {
        "estimated_neon_gb": estimated_gb,
        "estimated_bilingual_gb": estimated_gb * 2,
        "estimated_monthly_storage_cost": estimated_gb * storage_price_per_gb if storage_price_per_gb is not None else None,
    }
