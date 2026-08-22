"""Benchmark evidence-quality maturity decisions for successive Hindi STT partials."""

import argparse
import sys
import time
from pathlib import Path


BACKEND_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIRECTORY = Path(__file__).resolve().parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))
if str(SCRIPTS_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIRECTORY))

from app.rag.language_config import get_qdrant_target_lang  # noqa: E402
from app.rag.runtime import RAGRuntime  # noqa: E402
from benchmark_hybrid_partial_retrieval import (  # noqa: E402
    DEFAULT_FINAL,
    DEFAULT_PARTIALS,
    PartialEvaluation,
    evaluate_partial,
    evidence_found,
    indexed_documents,
    print_ranked,
    query_id_found,
    semantic_retrieve,
)
from hybrid_partial_retrieval import BM25Index  # noqa: E402
from retrieval_maturity_detector import MaturityPolicy, assess_retrieval_maturity  # noqa: E402


def bool_text(value: bool) -> str:
    """Use lower-case booleans consistently in benchmark output."""
    return str(value).lower()


def earliest_label(values: list[tuple[str, bool]]) -> str:
    """Return the first ordered partial satisfying a condition."""
    return next((label for label, value in values if value), "NONE")


def print_signals(
    evaluation: PartialEvaluation,
    final_reference,
    policy: MaturityPolicy,
) -> tuple[bool, bool, bool, float]:
    """Evaluate and print one partial, returning maturity and reference outcomes."""
    decision_started_at = time.perf_counter()
    decision = assess_retrieval_maturity(
        evaluation.text,
        evaluation.semantic,
        evaluation.lexical,
        evaluation.hybrid,
        policy,
    )
    decision_latency_ms = (time.perf_counter() - decision_started_at) * 1_000
    final_query_id_recovered = query_id_found(final_reference, evaluation.hybrid)
    final_exact_evidence_recovered = evidence_found(final_reference, evaluation.hybrid)

    print(f"\n{'=' * 72}\nPARTIAL {evaluation.label}: {evaluation.text}")
    print_ranked("SEMANTIC TOP-K", evaluation.semantic)
    print_ranked("BM25 TOP-K", evaluation.lexical)
    print_ranked("HYBRID TOP-K", evaluation.hybrid)
    print("\nMATURITY SIGNALS")
    print(f"SEMANTIC_TOP1_QUERY_ID: {decision.semantic_top1_query_id}")
    print(f"BM25_TOP1_QUERY_ID: {decision.lexical_top1_query_id}")
    print(f"SEMANTIC_BM25_TOP1_AGREE: {bool_text(decision.semantic_bm25_top1_agree)}")
    print(f"COMMON_QUERY_IDS_IN_TOP_K: {list(decision.common_query_ids_in_top_k)}")
    print(f"COMMON_PROVENANCES_IN_TOP_K: {list(decision.common_provenances_in_top_k)}")
    print(f"SEMANTIC_DOMINANT_QUERY_ID: {decision.semantic_dominant_query_id}")
    print(f"SEMANTIC_DOMINANT_QUERY_COUNT: {decision.semantic_dominant_query_count}")
    print(f"SEMANTIC_DOMINANCE_RATIO: {decision.semantic_dominance_ratio}")
    print(f"SEMANTIC_CONCENTRATION_PASSED: {bool_text(decision.semantic_concentration_passed)}")
    print(f"HYBRID_DOMINANT_QUERY_ID: {decision.hybrid_dominant_query_id}")
    print(f"HYBRID_DOMINANT_QUERY_COUNT: {decision.hybrid_dominant_query_count}")
    print(f"HYBRID_DOMINANCE_RATIO: {decision.hybrid_dominance_ratio}")
    print(f"HYBRID_CONCENTRATION_PASSED: {bool_text(decision.hybrid_concentration_passed)}")
    print(f"HYBRID_TOP1_SUPPORTED_BY_BOTH: {bool_text(decision.hybrid_top1_supported_by_both)}")
    print(f"TOP_RESULT_MARGIN: {decision.top_result_margin}")
    print(f"MEANINGFUL_OVERLAP_COUNT: {decision.meaningful_overlap_count}")
    print(f"MEANINGFUL_OVERLAP_SCORE: {decision.meaningful_overlap_score:.2f}")
    print(f"MEANINGFUL_OVERLAP_TERMS: {list(decision.meaningful_overlap_terms)}")
    print(f"CORROBORATION_PASSED: {bool_text(decision.corroboration_passed)}")
    print(f"SANITY_OVERLAP_PASSED: {bool_text(decision.sanity_overlap_passed)}")
    print(f"MATURITY_PATH: {decision.maturity_path}")
    print(f"PARTIAL_MATURE: {bool_text(decision.mature)}")
    print(f"REASON: {decision.reason}")
    print(f"FINAL_QUERY_ID_RECOVERED: {bool_text(final_query_id_recovered)}")
    print(f"FINAL_EXACT_EVIDENCE_RECOVERED: {bool_text(final_exact_evidence_recovered)}")
    print("\nLATENCY")
    print(f"SEMANTIC_LATENCY_MS: {evaluation.semantic_latency_ms:.2f}")
    print(f"BM25_LATENCY_MS: {evaluation.lexical_latency_ms:.2f}")
    print(f"FUSION_LATENCY_MS: {evaluation.fusion_latency_ms:.2f}")
    print(f"MATURITY_DECISION_LATENCY_MS: {decision_latency_ms:.3f}")
    print(f"TOTAL_HYBRID_MATURITY_LATENCY_MS: {evaluation.total_hybrid_latency_ms + decision_latency_ms:.2f}")
    return decision.mature, final_query_id_recovered, final_exact_evidence_recovered, evaluation.total_hybrid_latency_ms + decision_latency_ms


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--language", default="hi")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--semantic-dominant-min-count", type=int, default=3)
    parser.add_argument("--hybrid-dominant-min-count", type=int, default=3)
    parser.add_argument("--dominant-min-ratio", type=float, default=0.60)
    parser.add_argument("--min-sanity-overlap-count", type=int, default=1)
    args = parser.parse_args()
    if args.top_k < 1 or args.rrf_k < 1:
        parser.error("--top-k and --rrf-k must be at least 1")
    try:
        policy = MaturityPolicy(
            semantic_dominant_min_count=args.semantic_dominant_min_count,
            hybrid_dominant_min_count=args.hybrid_dominant_min_count,
            dominant_min_ratio=args.dominant_min_ratio,
            min_sanity_overlap_count=args.min_sanity_overlap_count,
        )
    except ValueError as exc:
        parser.error(str(exc))

    runtime: RAGRuntime | None = None
    try:
        target_lang = get_qdrant_target_lang(args.language)
        runtime = RAGRuntime()
        documents = indexed_documents(runtime, target_lang)
        if not documents:
            parser.error("No current indexed chunks were found for the selected language")
        bm25 = BM25Index(documents)

        print(f"FINAL TRANSCRIPT: {DEFAULT_FINAL}")
        print(f"PARTIALS: {', '.join(DEFAULT_PARTIALS)}")
        print(f"CURRENT QDRANT CHUNKS FOR {target_lang}: {len(documents)}")
        print(
            "MATURITY POLICY: "
            f"semantic_dominant_count>={policy.semantic_dominant_min_count}, "
            f"hybrid_dominant_count>={policy.hybrid_dominant_min_count}, "
            f"dominant_ratio>={policy.dominant_min_ratio:.2f}, "
            f"sanity_overlap_count>={policy.min_sanity_overlap_count}"
        )
        print("WARMING E5 + QDRANT WITH RETRIEVAL ONLY...")
        runtime.retrieve(DEFAULT_FINAL, top_k=args.top_k, target_lang=target_lang)
        print("E5 + QDRANT WARM-UP COMPLETE")

        _, final_semantic = semantic_retrieve(runtime, DEFAULT_FINAL, target_lang, args.top_k)
        print_ranked("FINAL SEMANTIC TOP-K REFERENCE", final_semantic)
        final_reference = final_semantic[0] if final_semantic else None
        if final_reference:
            print(f"FINAL TOP1 QUERY_ID: {final_reference.document.query_id}")

        outcomes: list[tuple[str, bool, bool, bool, float]] = []
        for label, partial_text in DEFAULT_PARTIALS.items():
            evaluation = evaluate_partial(runtime, bm25, label, partial_text, target_lang, args.top_k, args.rrf_k)
            mature, query_id, exact_evidence, total_latency = print_signals(evaluation, final_reference, policy)
            outcomes.append((label, mature, query_id, exact_evidence, total_latency))

        print(f"\n{'=' * 72}\nMATURITY SUMMARY")
        print("PARTIAL | MATURE | FINAL QUERY_ID RECOVERED | FINAL EXACT EVIDENCE RECOVERED | TOTAL LATENCY_MS")
        for label, mature, query_id, exact_evidence, latency in outcomes:
            print(f"{label} | {bool_text(mature)} | {bool_text(query_id)} | {bool_text(exact_evidence)} | {latency:.2f}")
        print(f"EARLIEST PARTIAL MARKED MATURE: {earliest_label([(label, mature) for label, mature, _, _, _ in outcomes])}")
        print(f"EARLIEST PARTIAL THAT RECOVERS FINAL QUERY_ID: {earliest_label([(label, query_id) for label, _, query_id, _, _ in outcomes])}")
        print(f"EARLIEST PARTIAL THAT RECOVERS FINAL EXACT EVIDENCE: {earliest_label([(label, exact) for label, _, _, exact, _ in outcomes])}")
        false_early = [label for label, mature, _, exact, _ in outcomes if mature and not exact]
        missed = [label for label, mature, _, exact, _ in outcomes if exact and not mature]
        print(f"FALSE EARLY TRIGGERS: {false_early}")
        print(f"MISSED OPPORTUNITIES: {missed}")
    finally:
        if runtime is not None:
            runtime.close()


if __name__ == "__main__":
    main()
