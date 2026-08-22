"""Deterministic, benchmark-only evidence-quality checks for partial transcripts."""

from collections import Counter
from dataclasses import dataclass

from hybrid_partial_retrieval import RankedResult, tokenize_hindi


# Function words do not provide useful evidence that a partial identifies a passage.
HINDI_STOPWORDS = frozenset({"और", "का", "की", "के", "को", "कितनी", "में", "पर", "से", "है", "हैं", "क्या"})


@dataclass(frozen=True)
class MaturityPolicy:
    """Small, rank-based rules for allowing benchmark-only speculative retrieval."""

    semantic_dominant_min_count: int = 3
    hybrid_dominant_min_count: int = 3
    dominant_min_ratio: float = 0.60
    min_sanity_overlap_count: int = 1

    def __post_init__(self) -> None:
        if self.semantic_dominant_min_count < 1:
            raise ValueError("semantic_dominant_min_count must be at least 1")
        if self.hybrid_dominant_min_count < 1:
            raise ValueError("hybrid_dominant_min_count must be at least 1")
        if not 0 < self.dominant_min_ratio <= 1:
            raise ValueError("dominant_min_ratio must be greater than 0 and at most 1")
        if self.min_sanity_overlap_count < 1:
            raise ValueError("min_sanity_overlap_count must be at least 1")


@dataclass(frozen=True)
class MaturityDecision:
    """All evidence signals and the resulting transparent maturity decision."""

    mature: bool
    reason: str
    maturity_path: str | None
    semantic_top1_query_id: str | None
    lexical_top1_query_id: str | None
    semantic_bm25_top1_agree: bool
    common_query_ids_in_top_k: tuple[str, ...]
    common_provenances_in_top_k: tuple[tuple[str, str, str], ...]
    semantic_dominant_query_id: str | None
    semantic_dominant_query_count: int
    semantic_dominance_ratio: float | None
    semantic_concentration_passed: bool
    hybrid_dominant_query_id: str | None
    hybrid_dominant_query_count: int
    hybrid_dominance_ratio: float | None
    hybrid_concentration_passed: bool
    hybrid_top1_supported_by_both: bool
    top_result_margin: float | None
    meaningful_overlap_count: int
    meaningful_overlap_score: float
    meaningful_overlap_terms: tuple[str, ...]
    sanity_overlap_passed: bool
    corroboration_passed: bool


def meaningful_tokens(text: str) -> set[str]:
    """Return non-stopword tokens for a lightweight evidence sanity check."""
    return {token for token in tokenize_hindi(text) if token not in HINDI_STOPWORDS}


def _dominant_query_id(results: list[RankedResult]) -> tuple[str | None, int]:
    """Return the most frequent query ID, breaking equal counts by first rank."""
    if not results:
        return None, 0
    counts = Counter(result.document.query_id for result in results)
    first_rank = {result.document.query_id: result.rank for result in results}
    query_id = min(counts, key=lambda item: (-counts[item], first_rank[item], item))
    return query_id, counts[query_id]


def _dominance_ratio(count: int, results: list[RankedResult]) -> float | None:
    """Return a query-ID concentration ratio for an observed result list."""
    return count / len(results) if results else None


def _top_result_margin(results: list[RankedResult]) -> float | None:
    """Return first-versus-second RRF separation when both scores are present."""
    if len(results) < 2 or results[0].fused_score is None or results[1].fused_score is None:
        return None
    return results[0].fused_score - results[1].fused_score


def assess_retrieval_maturity(
    partial_text: str,
    semantic_results: list[RankedResult],
    lexical_results: list[RankedResult],
    hybrid_results: list[RankedResult],
    policy: MaturityPolicy | None = None,
) -> MaturityDecision:
    """Assess retrieval corroboration without using an absolute vector-score gate.

    A mature partial needs one meaningful-term sanity match plus either:

    * strong semantic query-ID concentration, presence of that ID in BM25, and
      a hybrid top-one result from that same ID; or
    * strong hybrid query-ID concentration whose dominant ID appears in both
      semantic and BM25 result lists.
    """
    active_policy = policy or MaturityPolicy()
    semantic_ids = {result.document.query_id for result in semantic_results}
    lexical_ids = {result.document.query_id for result in lexical_results}
    semantic_provenances = {result.provenance for result in semantic_results}
    lexical_provenances = {result.provenance for result in lexical_results}
    semantic_top1 = semantic_results[0].document.query_id if semantic_results else None
    lexical_top1 = lexical_results[0].document.query_id if lexical_results else None
    top1_agreement = bool(semantic_top1 and lexical_top1 and semantic_top1 == lexical_top1)
    common_query_ids = tuple(sorted(semantic_ids & lexical_ids))
    common_provenances = tuple(sorted(semantic_provenances & lexical_provenances))

    semantic_dominant_id, semantic_dominant_count = _dominant_query_id(semantic_results)
    semantic_ratio = _dominance_ratio(semantic_dominant_count, semantic_results)
    semantic_concentrated = bool(
        semantic_dominant_id
        and semantic_dominant_count >= active_policy.semantic_dominant_min_count
        and semantic_ratio is not None
        and semantic_ratio >= active_policy.dominant_min_ratio
    )
    hybrid_dominant_id, hybrid_dominant_count = _dominant_query_id(hybrid_results)
    hybrid_ratio = _dominance_ratio(hybrid_dominant_count, hybrid_results)
    hybrid_concentrated = bool(
        hybrid_dominant_id
        and hybrid_dominant_count >= active_policy.hybrid_dominant_min_count
        and hybrid_ratio is not None
        and hybrid_ratio >= active_policy.dominant_min_ratio
    )

    query_terms = meaningful_tokens(partial_text)
    evidence_terms = meaningful_tokens(hybrid_results[0].document.text) if hybrid_results else set()
    overlap_terms = tuple(sorted(query_terms & evidence_terms))
    overlap_count = len(overlap_terms)
    overlap_score = overlap_count / len(query_terms) if query_terms else 0.0
    sanity_overlap_passed = overlap_count >= active_policy.min_sanity_overlap_count

    hybrid_top1_id = hybrid_results[0].document.query_id if hybrid_results else None
    hybrid_top1_supported_by_both = bool(
        hybrid_results
        and hybrid_results[0].provenance in semantic_provenances
        and hybrid_results[0].provenance in lexical_provenances
    )
    semantic_path = bool(
        semantic_concentrated
        and semantic_dominant_id in lexical_ids
        and hybrid_top1_id == semantic_dominant_id
    )
    hybrid_path = bool(
        hybrid_concentrated
        and hybrid_dominant_id in semantic_ids
        and hybrid_dominant_id in lexical_ids
    )
    corroboration_passed = semantic_path or hybrid_path

    if not hybrid_results:
        reason, maturity_path = "No hybrid evidence was returned.", None
    elif not sanity_overlap_passed:
        reason, maturity_path = (
            f"Top hybrid evidence has no meaningful partial-term sanity match (count={overlap_count}).",
            None,
        )
    elif semantic_path:
        reason, maturity_path = (
            "Strong semantic query-ID concentration is corroborated by BM25 and selected by hybrid top-one.",
            "semantic_concentration",
        )
    elif hybrid_path:
        reason, maturity_path = (
            "Strong hybrid query-ID concentration is corroborated by both semantic and BM25 top-k evidence.",
            "hybrid_concentration",
        )
    else:
        reason, maturity_path = (
            "Retrieval evidence lacks a strongly concentrated query ID corroborated across semantic and BM25 paths.",
            None,
        )

    return MaturityDecision(
        mature=bool(hybrid_results and sanity_overlap_passed and corroboration_passed),
        reason=reason,
        maturity_path=maturity_path,
        semantic_top1_query_id=semantic_top1,
        lexical_top1_query_id=lexical_top1,
        semantic_bm25_top1_agree=top1_agreement,
        common_query_ids_in_top_k=common_query_ids,
        common_provenances_in_top_k=common_provenances,
        semantic_dominant_query_id=semantic_dominant_id,
        semantic_dominant_query_count=semantic_dominant_count,
        semantic_dominance_ratio=semantic_ratio,
        semantic_concentration_passed=semantic_concentrated,
        hybrid_dominant_query_id=hybrid_dominant_id,
        hybrid_dominant_query_count=hybrid_dominant_count,
        hybrid_dominance_ratio=hybrid_ratio,
        hybrid_concentration_passed=hybrid_concentrated,
        hybrid_top1_supported_by_both=hybrid_top1_supported_by_both,
        top_result_margin=_top_result_margin(hybrid_results),
        meaningful_overlap_count=overlap_count,
        meaningful_overlap_score=overlap_score,
        meaningful_overlap_terms=overlap_terms,
        sanity_overlap_passed=sanity_overlap_passed,
        corroboration_passed=corroboration_passed,
    )
