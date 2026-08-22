"""Retrieval-corroboration policy for speculative partial-transcript work."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from app.rag.retrieval.bm25_store import tokenize_text
from app.rag.retrieval.hybrid_retriever import HybridRetrievedChunk


HINDI_STOPWORDS = frozenset({"और", "का", "की", "के", "को", "कितनी", "में", "पर", "से", "है", "हैं", "क्या"})


@dataclass(frozen=True)
class MaturityPolicy:
    """Validated, configurable thresholds for retrieval evidence corroboration."""

    semantic_dominant_min_count: int = 3
    hybrid_dominant_min_count: int = 3
    dominant_min_ratio: float = 0.60
    min_sanity_overlap_count: int = 1
    prefilter_min_meaningful_tokens: int = 2

    def __post_init__(self) -> None:
        if self.semantic_dominant_min_count < 1 or self.hybrid_dominant_min_count < 1:
            raise ValueError("dominant query-ID counts must be at least 1")
        if not 0 < self.dominant_min_ratio <= 1:
            raise ValueError("dominant_min_ratio must be in (0, 1]")
        if self.min_sanity_overlap_count < 1 or self.prefilter_min_meaningful_tokens < 1:
            raise ValueError("meaningful-token counts must be at least 1")


@dataclass(frozen=True)
class MaturityDecision:
    """Explainable signals and maturity outcome for a single partial transcript."""

    mature: bool
    reason: str
    maturity_path: str | None
    semantic_dominant_query_id: str | None
    semantic_dominant_query_count: int
    semantic_dominance_ratio: float | None
    hybrid_dominant_query_id: str | None
    hybrid_dominant_query_count: int
    hybrid_dominance_ratio: float | None
    common_query_ids: tuple[str, ...]
    common_provenances: tuple[tuple[str, str, str], ...]
    hybrid_top1_supported_by_both: bool
    meaningful_overlap_terms: tuple[str, ...]
    sanity_overlap_passed: bool


def meaningful_tokens(text: str) -> set[str]:
    """Return lexical terms that can be used as a lightweight partial prefilter."""
    return {token for token in tokenize_text(text) if token not in HINDI_STOPWORDS}


def has_enough_meaningful_tokens(text: str, policy: MaturityPolicy) -> bool:
    """Avoid an E5/BM25 call for trivially small partial transcripts."""
    return len(meaningful_tokens(text)) >= policy.prefilter_min_meaningful_tokens


def _dominant(results: list[HybridRetrievedChunk]) -> tuple[str | None, int, float | None]:
    if not results:
        return None, 0, None
    counts = Counter(str(result.metadata.get("query_id", "")) for result in results)
    first_rank = {str(result.metadata.get("query_id", "")): result.rank for result in results}
    query_id = min(counts, key=lambda item: (-counts[item], first_rank[item], item))
    return query_id, counts[query_id], counts[query_id] / len(results)


def assess_retrieval_maturity(
    partial_text: str,
    semantic_results: list[HybridRetrievedChunk],
    lexical_results: list[HybridRetrievedChunk],
    hybrid_results: list[HybridRetrievedChunk],
    policy: MaturityPolicy | None = None,
) -> MaturityDecision:
    """Apply the validated concentration/corroboration policy without score thresholds."""
    active_policy = policy or MaturityPolicy()
    semantic_ids = {str(result.metadata.get("query_id", "")) for result in semantic_results}
    lexical_ids = {str(result.metadata.get("query_id", "")) for result in lexical_results}
    semantic_provenances = {result.provenance for result in semantic_results}
    lexical_provenances = {result.provenance for result in lexical_results}
    semantic_id, semantic_count, semantic_ratio = _dominant(semantic_results)
    hybrid_id, hybrid_count, hybrid_ratio = _dominant(hybrid_results)
    semantic_concentrated = bool(
        semantic_id
        and semantic_count >= active_policy.semantic_dominant_min_count
        and semantic_ratio is not None
        and semantic_ratio >= active_policy.dominant_min_ratio
    )
    hybrid_concentrated = bool(
        hybrid_id
        and hybrid_count >= active_policy.hybrid_dominant_min_count
        and hybrid_ratio is not None
        and hybrid_ratio >= active_policy.dominant_min_ratio
    )

    query_terms = meaningful_tokens(partial_text)
    evidence_terms = meaningful_tokens(hybrid_results[0].text) if hybrid_results else set()
    overlap_terms = tuple(sorted(query_terms & evidence_terms))
    sanity_overlap_passed = len(overlap_terms) >= active_policy.min_sanity_overlap_count
    hybrid_top1_id = str(hybrid_results[0].metadata.get("query_id", "")) if hybrid_results else None
    hybrid_top1_supported_by_both = bool(
        hybrid_results
        and hybrid_results[0].provenance in semantic_provenances
        and hybrid_results[0].provenance in lexical_provenances
    )
    semantic_path = bool(semantic_concentrated and semantic_id in lexical_ids and hybrid_top1_id == semantic_id)
    hybrid_path = bool(hybrid_concentrated and hybrid_id in semantic_ids and hybrid_id in lexical_ids)

    if not hybrid_results:
        mature, reason, path = False, "No hybrid evidence was returned.", None
    elif not sanity_overlap_passed:
        mature, reason, path = False, "Top hybrid evidence has no meaningful partial-term sanity match.", None
    elif semantic_path:
        mature, reason, path = True, "Semantic concentration is corroborated by BM25 and selected by hybrid top-one.", "semantic_concentration"
    elif hybrid_path:
        mature, reason, path = True, "Hybrid concentration is corroborated by semantic and BM25 top-k evidence.", "hybrid_concentration"
    else:
        mature, reason, path = False, "No strongly concentrated query ID is corroborated across retrieval paths.", None

    return MaturityDecision(
        mature=mature,
        reason=reason,
        maturity_path=path,
        semantic_dominant_query_id=semantic_id,
        semantic_dominant_query_count=semantic_count,
        semantic_dominance_ratio=semantic_ratio,
        hybrid_dominant_query_id=hybrid_id,
        hybrid_dominant_query_count=hybrid_count,
        hybrid_dominance_ratio=hybrid_ratio,
        common_query_ids=tuple(sorted(semantic_ids & lexical_ids)),
        common_provenances=tuple(sorted(semantic_provenances & lexical_provenances)),
        hybrid_top1_supported_by_both=hybrid_top1_supported_by_both,
        meaningful_overlap_terms=overlap_terms,
        sanity_overlap_passed=sanity_overlap_passed,
    )
