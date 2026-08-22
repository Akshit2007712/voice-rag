"""Compare semantic, BM25, and RRF retrieval for successive Hindi STT partials.

This is an experiment-only script. It reads the existing development Qdrant
collection but does not modify any production retrieval component.
"""

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.rag.language_config import get_qdrant_target_lang  # noqa: E402
from app.rag.retrieval.retriever import RetrievedChunk  # noqa: E402
from app.rag.runtime import RAGRuntime  # noqa: E402
from hybrid_partial_retrieval import (  # noqa: E402
    BM25Index,
    IndexedDocument,
    RankedResult,
    provenance_overlap_ratio,
    reciprocal_rank_fusion,
)


DEFAULT_PARTIALS = {
    "A": "विला डेल सिया लैंक",
    "B": "विला डेल सिया लैंकेस्टर से",
    "C": "विला डेल सिया लैंकेस्टर से कितनी दूर",
    "D": "विला डेल सिया लैंकेस्टर से कितनी दूर है",
}
DEFAULT_FINAL = "फिलाडेल्फिया लैंकेस्टर से कितनी दूर है?"


@dataclass(frozen=True)
class PartialEvaluation:
    """One partial's comparable outputs and measured retrieval costs."""

    label: str
    text: str
    semantic: list[RankedResult]
    lexical: list[RankedResult]
    hybrid: list[RankedResult]
    semantic_latency_ms: float
    lexical_latency_ms: float
    fusion_latency_ms: float

    @property
    def total_hybrid_latency_ms(self) -> float:
        """Return semantic retrieval plus benchmark-only lexical and fusion work."""
        return self.semantic_latency_ms + self.lexical_latency_ms + self.fusion_latency_ms


def indexed_documents(runtime: RAGRuntime, target_lang: str) -> list[IndexedDocument]:
    """Build the lexical corpus only from chunks already in the current collection."""
    documents: list[IndexedDocument] = []
    offset = None
    while True:
        points, offset = runtime.vector_store.client.scroll(
            collection_name=runtime.vector_store.collection_name,
            limit=256,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for point in points:
            payload = dict(point.payload or {})
            if payload.get("target_lang") != target_lang:
                continue
            text = str(payload.get("text") or "").strip()
            if not text:
                continue
            documents.append(
                IndexedDocument(
                    text=text,
                    query_id=str(payload.get("query_id", "")),
                    passage_index=str(payload.get("passage_index", "")),
                    chunk_index=str(payload.get("chunk_index", "")),
                    target_lang=str(payload.get("target_lang", "")),
                )
            )
        if offset is None:
            return documents


def ranked_semantic(chunks: list[RetrievedChunk]) -> list[RankedResult]:
    """Project unchanged production Retriever output into experiment records."""
    return [
        RankedResult(
            rank=rank,
            document=IndexedDocument(
                text=chunk.text,
                query_id=str(chunk.metadata.get("query_id", "")),
                passage_index=str(chunk.metadata.get("passage_index", "")),
                chunk_index=str(chunk.metadata.get("chunk_index", "")),
                target_lang=str(chunk.metadata.get("target_lang", "")),
            ),
            source_score=chunk.score,
            semantic_score=chunk.score,
        )
        for rank, chunk in enumerate(chunks, start=1)
    ]


def semantic_retrieve(
    runtime: RAGRuntime, query: str, target_lang: str, top_k: int
) -> tuple[float, list[RankedResult]]:
    """Time existing E5 + Qdrant retrieval without changing the production path."""
    started_at = time.perf_counter()
    results = ranked_semantic(runtime.retrieve(query, top_k=top_k, target_lang=target_lang))
    return (time.perf_counter() - started_at) * 1_000, results


def evaluate_partial(
    runtime: RAGRuntime,
    bm25: BM25Index,
    label: str,
    partial_text: str,
    target_lang: str,
    top_k: int,
    rrf_k: int,
) -> PartialEvaluation:
    """Run all three benchmark retrieval modes for one partial transcript."""
    semantic_latency_ms, semantic = semantic_retrieve(runtime, partial_text, target_lang, top_k)

    lexical_started_at = time.perf_counter()
    lexical = bm25.search(partial_text, top_k)
    lexical_latency_ms = (time.perf_counter() - lexical_started_at) * 1_000

    fusion_started_at = time.perf_counter()
    hybrid = reciprocal_rank_fusion(semantic, lexical, rrf_k, top_k)
    fusion_latency_ms = (time.perf_counter() - fusion_started_at) * 1_000
    return PartialEvaluation(
        label=label,
        text=partial_text,
        semantic=semantic,
        lexical=lexical,
        hybrid=hybrid,
        semantic_latency_ms=semantic_latency_ms,
        lexical_latency_ms=lexical_latency_ms,
        fusion_latency_ms=fusion_latency_ms,
    )


def evidence_found(reference: RankedResult | None, candidates: list[RankedResult]) -> bool:
    """Return whether a candidate list contains the final top-one provenance."""
    return reference is not None and any(candidate.provenance == reference.provenance for candidate in candidates)


def query_id_found(reference: RankedResult | None, candidates: list[RankedResult]) -> bool:
    """Return whether a candidate list contains the final top-one query identifier."""
    return reference is not None and any(candidate.document.query_id == reference.document.query_id for candidate in candidates)


def print_ranked(title: str, results: list[RankedResult]) -> None:
    """Print provenance and scores while deliberately omitting chunk-text comparison."""
    print(f"\n{title}")
    if not results:
        print("n/a")
        return
    for result in results:
        print(
            f"rank={result.rank}, query_id={result.document.query_id}, "
            f"passage_index={result.document.passage_index}, chunk_index={result.document.chunk_index}, "
            f"semantic_score={result.semantic_score}, bm25_score={result.lexical_score}, "
            f"fused_score={result.fused_score}"
        )


def bool_text(value: bool) -> str:
    """Keep boolean benchmark output easy to scan and machine-copy."""
    return str(value).lower()


def first_recovery_label(
    evaluations: list[PartialEvaluation],
    reference: RankedResult | None,
    checker: Callable[[RankedResult | None, list[RankedResult]], bool],
) -> str:
    """Find the earliest A-to-D partial recovered by any retrieval mode."""
    for evaluation in evaluations:
        if any(checker(reference, candidates) for candidates in (evaluation.semantic, evaluation.lexical, evaluation.hybrid)):
            return evaluation.label
    return "NONE"


def first_recovery_by_method(
    evaluations: list[PartialEvaluation],
    reference: RankedResult | None,
    checker: Callable[[RankedResult | None, list[RankedResult]], bool],
    attribute: str,
) -> str:
    """Find the earliest partial recovered by one named retrieval mode."""
    for evaluation in evaluations:
        if checker(reference, getattr(evaluation, attribute)):
            return evaluation.label
    return "NONE"


def print_partial_metrics(evaluation: PartialEvaluation, final_semantic: list[RankedResult]) -> None:
    """Print one partial's complete top-k evidence comparison and timings."""
    final_top1 = final_semantic[0] if final_semantic else None
    semantic_query_id = query_id_found(final_top1, evaluation.semantic)
    lexical_query_id = query_id_found(final_top1, evaluation.lexical)
    hybrid_query_id = query_id_found(final_top1, evaluation.hybrid)
    semantic_evidence = evidence_found(final_top1, evaluation.semantic)
    lexical_evidence = evidence_found(final_top1, evaluation.lexical)
    hybrid_evidence = evidence_found(final_top1, evaluation.hybrid)
    hybrid_top1_same = bool(final_top1 and evaluation.hybrid and evaluation.hybrid[0].provenance == final_top1.provenance)

    print(f"\n{'=' * 72}\nPARTIAL {evaluation.label}: {evaluation.text}")
    print_ranked("SEMANTIC PARTIAL TOP-5", evaluation.semantic)
    print_ranked("LEXICAL PARTIAL TOP-5", evaluation.lexical)
    print_ranked("HYBRID PARTIAL TOP-5", evaluation.hybrid)
    print("\nRECOVERY METRICS")
    print(f"SEMANTIC_FINAL_TOP1_QUERY_ID_FOUND_TOP5: {bool_text(semantic_query_id)}")
    print(f"LEXICAL_FINAL_TOP1_QUERY_ID_FOUND_TOP5: {bool_text(lexical_query_id)}")
    print(f"HYBRID_FINAL_TOP1_QUERY_ID_FOUND_TOP5: {bool_text(hybrid_query_id)}")
    print(f"SEMANTIC_FINAL_TOP1_EXACT_EVIDENCE_FOUND_TOP5: {bool_text(semantic_evidence)}")
    print(f"LEXICAL_FINAL_TOP1_EXACT_EVIDENCE_FOUND_TOP5: {bool_text(lexical_evidence)}")
    print(f"HYBRID_FINAL_TOP1_EXACT_EVIDENCE_FOUND_TOP5: {bool_text(hybrid_evidence)}")
    print(f"HYBRID_TOP1_SAME_AS_FINAL: {bool_text(hybrid_top1_same)}")
    print(f"HYBRID_TOP5_OVERLAP_RATIO: {provenance_overlap_ratio(evaluation.hybrid, final_semantic, 5)}")
    print("\nLATENCY")
    print(f"SEMANTIC_LATENCY_MS: {evaluation.semantic_latency_ms:.2f}")
    print(f"BM25_LATENCY_MS: {evaluation.lexical_latency_ms:.2f}")
    print(f"FUSION_LATENCY_MS: {evaluation.fusion_latency_ms:.2f}")
    print(f"TOTAL_HYBRID_LATENCY_MS: {evaluation.total_hybrid_latency_ms:.2f}")


def print_summary(evaluations: list[PartialEvaluation], final_semantic: list[RankedResult]) -> None:
    """Print decision-support results without making an architectural decision."""
    final_top1 = final_semantic[0] if final_semantic else None
    print(f"\n{'=' * 72}\nMULTI-PARTIAL SUMMARY")
    print(
        "PARTIAL | SEMANTIC FINAL ID FOUND | BM25 FINAL ID FOUND | "
        "HYBRID FINAL ID FOUND | HYBRID EXACT EVIDENCE FOUND | HYBRID LATENCY_MS"
    )
    for evaluation in evaluations:
        print(
            f"{evaluation.label} | "
            f"{bool_text(query_id_found(final_top1, evaluation.semantic))} | "
            f"{bool_text(query_id_found(final_top1, evaluation.lexical))} | "
            f"{bool_text(query_id_found(final_top1, evaluation.hybrid))} | "
            f"{bool_text(evidence_found(final_top1, evaluation.hybrid))} | "
            f"{evaluation.total_hybrid_latency_ms:.2f}"
        )

    print(
        "EARLIEST PARTIAL WITH FINAL QUERY_ID RECOVERY: "
        f"{first_recovery_label(evaluations, final_top1, query_id_found)}"
    )
    print(
        "EARLIEST PARTIAL WITH FINAL EXACT EVIDENCE RECOVERY: "
        f"{first_recovery_label(evaluations, final_top1, evidence_found)}"
    )
    print(
        "EARLIEST QUERY_ID RECOVERY BY METHOD: "
        f"semantic={first_recovery_by_method(evaluations, final_top1, query_id_found, 'semantic')}, "
        f"bm25={first_recovery_by_method(evaluations, final_top1, query_id_found, 'lexical')}, "
        f"hybrid={first_recovery_by_method(evaluations, final_top1, query_id_found, 'hybrid')}"
    )
    print(
        "EARLIEST EXACT-EVIDENCE RECOVERY BY METHOD: "
        f"semantic={first_recovery_by_method(evaluations, final_top1, evidence_found, 'semantic')}, "
        f"bm25={first_recovery_by_method(evaluations, final_top1, evidence_found, 'lexical')}, "
        f"hybrid={first_recovery_by_method(evaluations, final_top1, evidence_found, 'hybrid')}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--partial-text",
        help="Optional one-off partial. Omit it to run the standard A-D partial series in one process.",
    )
    parser.add_argument("--final-text", default=DEFAULT_FINAL, help="Correct final transcript reference")
    parser.add_argument("--language", default="hi")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--rrf-k", type=int, default=60)
    args = parser.parse_args()
    if args.partial_text is not None and not args.partial_text.strip():
        parser.error("--partial-text must be non-empty when supplied")
    if not args.final_text.strip():
        parser.error("--final-text must be non-empty")
    if args.top_k < 1 or args.rrf_k < 1:
        parser.error("--top-k and --rrf-k must be at least 1")

    partials = {"CUSTOM": args.partial_text} if args.partial_text is not None else DEFAULT_PARTIALS
    runtime: RAGRuntime | None = None
    try:
        target_lang = get_qdrant_target_lang(args.language)
        runtime = RAGRuntime()
        documents = indexed_documents(runtime, target_lang)
        if not documents:
            parser.error("No current indexed chunks were found for the selected language")
        bm25 = BM25Index(documents)

        print(f"FINAL QUERY: {args.final_text}")
        print(f"PARTIALS TO EVALUATE: {', '.join(partials)}")
        print(f"CURRENT QDRANT CHUNKS INDEXED FOR {target_lang}: {len(documents)}")
        print(f"RRF K: {args.rrf_k}")
        print("WARMING E5 + QDRANT WITH RETRIEVAL ONLY...")
        runtime.retrieve(args.final_text, top_k=args.top_k, target_lang=target_lang)
        print("E5 + QDRANT WARM-UP COMPLETE")

        final_latency_ms, final_semantic = semantic_retrieve(runtime, args.final_text, target_lang, args.top_k)
        print_ranked("FINAL SEMANTIC TOP-5 REFERENCE", final_semantic)
        print(f"FINAL REFERENCE SEMANTIC LATENCY_MS: {final_latency_ms:.2f}")
        if final_semantic:
            print(f"FINAL TOP1 QUERY_ID: {final_semantic[0].document.query_id}")

        evaluations = [
            evaluate_partial(runtime, bm25, label, text, target_lang, args.top_k, args.rrf_k)
            for label, text in partials.items()
        ]
        for evaluation in evaluations:
            print_partial_metrics(evaluation, final_semantic)
        print_summary(evaluations, final_semantic)
    finally:
        if runtime is not None:
            runtime.close()


if __name__ == "__main__":
    main()
