"""Measure warmed sequential and real parallel bilingual hybrid-RAG latency."""

from __future__ import annotations

import argparse
import math
import statistics
import sys
import time
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.rag.bilingual_cloud_runtime import create_bilingual_cloud_runtime  # noqa: E402
from app.rag.ingestion.dataset_loader import iter_msmarco_xi_records  # noqa: E402
from app.rag.language_config import get_qdrant_target_lang  # noqa: E402
from app.rag.retrieval.bm25_store import BM25Store  # noqa: E402
from app.rag.retrieval.hybrid_retriever import HybridRetriever, HybridRetrievalResult  # noqa: E402
from app.rag.retrieval.maturity import assess_retrieval_maturity  # noqa: E402


def percentile(values: list[float], value: int) -> float:
    return values[0] if len(values) == 1 else statistics.quantiles(values, n=100, method="inclusive")[value - 1]


def query_sample(language: str, count: int) -> list[str]:
    field = "Eng_Query" if language == "en" else "query"
    result: list[str] = []
    for record in iter_msmarco_xi_records(language, "validation", 500):
        if query := str(record.get(field, "")).strip():
            result.append(query)
        if len(result) == count:
            return result
    return result


def measure_post_retrieval(runtime, query: str, result: HybridRetrievalResult) -> dict[str, float]:
    started = time.perf_counter()
    assess_retrieval_maturity(query, result.semantic, result.lexical, result.fused)
    maturity_ms = (time.perf_counter() - started) * 1_000
    started = time.perf_counter()
    runtime.answer_composer.compose(query, [item for chunk in result.fused if (item := chunk.as_retrieved_chunk())], 3)
    composer_ms = (time.perf_counter() - started) * 1_000
    return {
        "embedding": result.embedding_latency_ms or 0.0,
        "qdrant_operation": result.qdrant_operation_latency_ms or 0.0,
        "qdrant_retry_wait": result.qdrant_retry_wait_ms or 0.0,
        "qdrant_wall": result.qdrant_search_latency_ms or 0.0,
        "bm25": result.lexical_latency_ms,
        "bm25_queue_wait": result.executor_queue_wait_ms,
        "bm25_worker_compute": result.worker_compute_ms,
        "bm25_future_wait": result.future_wait_ms,
        "bm25_total_wall": result.bm25_total_wall_ms,
        "qdrant_branch_wall": result.qdrant_branch_wall_ms,
        "post_embed_parallel_wall": result.post_embed_parallel_wall_ms,
        "semantic_branch": result.semantic_latency_ms,
        "parallel_hybrid_retrieval": result.total_latency_ms,
        "rrf": result.fusion_latency_ms,
        "maturity": maturity_ms,
        "composer": composer_ms,
        "total": result.total_latency_ms + maturity_ms + composer_ms,
    }


def print_percentiles(name: str, values: list[float]) -> None:
    print(f"{name}_P50_MS={percentile(values, 50):.2f} P70_MS={percentile(values, 70):.2f} P95_MS={percentile(values, 95):.2f} P100_MS={max(values):.2f}")


def print_bm25_diagnostic(label: str, language: str, index: int, query: str, hybrid: HybridRetriever, target: str) -> None:
    """Profile the selected query without changing BM25 scoring or corpus state."""
    profile = hybrid.retrieve_sequential(query, 5, target).bm25_profile
    assert profile is not None
    print(f"BM25_{label}_DIAGNOSTIC language={language} query_index={index} query={query!r} query_terms={profile.query_terms} posting_list_sizes={profile.posting_list_sizes} total_posting_entries={profile.total_posting_entries} candidate_documents={profile.candidate_document_count} candidate_lookup_ms={profile.candidate_lookup_ms:.2f} score_ms={profile.score_ms:.2f} topk_ms={profile.topk_ms:.2f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries-per-language", type=int, default=50)
    args = parser.parse_args()
    if args.queries_per_language < 50:
        parser.error("--queries-per-language must be at least 50")
    runtime, _point_count = create_bilingual_cloud_runtime(BACKEND_ROOT)
    hybrids: dict[str, HybridRetriever] = {}
    try:
        samples = {language: query_sample(language, args.queries_per_language) for language in ("hi", "en")}
        if any(len(sample) != args.queries_per_language for sample in samples.values()):
            raise RuntimeError("Insufficient deterministic validation queries")
        targets = [get_qdrant_target_lang(language) for language in ("hi", "en")]
        stores = BM25Store.by_language_from_vector_store(runtime.vector_store, targets)
        hybrids = {language: HybridRetriever(runtime.retriever, stores[get_qdrant_target_lang(language)]) for language in ("hi", "en")}
        for language, hybrid in hybrids.items():
            target = get_qdrant_target_lang(language)
            for query in samples[language][:3]:
                hybrid.retrieve(query, 5, target)
        print("E5_WARMUP_COMPLETE=true")
        print("BENCHMARK_NETWORK=LOCAL_MACHINE_TO_REMOTE_QDRANT")
        print("DOCUMENT_EMBEDDINGS_CREATED=0 QDRANT_WRITES=0 CORPUS_CHANGED=0")
        for language, hybrid in hybrids.items():
            target = get_qdrant_target_lang(language)
            semantic_only = []
            bm25_only = []
            for query in samples[language]:
                _chunks, semantic_profile = runtime.retriever.retrieve_profiled(query, 5, target)
                semantic_only.append(semantic_profile)
                _matches, bm25_profile = hybrid.bm25_store.search_profiled(query, 5)
                bm25_only.append(bm25_profile)
            parallel = [measure_post_retrieval(runtime, query, hybrid.retrieve(query, 5, target)) for query in samples[language]]
            sequential = [measure_post_retrieval(runtime, query, hybrid.retrieve_sequential(query, 5, target)) for query in samples[language]]
            print(f"LANGUAGE={language} SAMPLES={len(parallel)}")
            print_percentiles("SEMANTIC_ONLY_E5", [item.embedding_ms for item in semantic_only])
            print_percentiles("SEMANTIC_ONLY_TOTAL", [item.total_ms for item in semantic_only])
            print_percentiles("BM25_ONLY_TOTAL", [item.total_bm25_ms for item in bm25_only])
            for key in ("embedding", "qdrant_operation", "qdrant_retry_wait", "qdrant_wall", "bm25", "bm25_queue_wait", "bm25_worker_compute", "bm25_future_wait", "bm25_total_wall", "qdrant_branch_wall", "post_embed_parallel_wall", "semantic_branch", "parallel_hybrid_retrieval", "rrf", "maturity", "composer", "total"):
                print_percentiles(key.upper(), [row[key] for row in parallel])
            print_percentiles("SEQUENTIAL_TOTAL", [row["total"] for row in sequential])
            print_percentiles("PARALLEL_TOTAL", [row["total"] for row in parallel])
            ordered = sorted(range(len(parallel)), key=lambda index: parallel[index]["bm25"])
            median_index = ordered[len(ordered) // 2]
            p95_index = ordered[math.ceil(len(ordered) * 0.95) - 1]
            outlier_index = ordered[-1]
            print_bm25_diagnostic("MEDIAN", language, median_index, samples[language][median_index], hybrid, target)
            print_bm25_diagnostic("P95", language, p95_index, samples[language][p95_index], hybrid, target)
            print_bm25_diagnostic("OUTLIER", language, outlier_index, samples[language][outlier_index], hybrid, target)
            for index, row in enumerate(parallel):
                if row["bm25_total_wall"] > 300:
                    print(f"BM25_WALL_OUTLIER language={language} query_index={index} queue_wait_ms={row['bm25_queue_wait']:.2f} worker_compute_ms={row['bm25_worker_compute']:.2f} future_wait_ms={row['bm25_future_wait']:.2f} total_wall_ms={row['bm25_total_wall']:.2f}")
                if row["embedding"] > 300:
                    token_count, tokenizer_ms = runtime.embedder.profile_query_tokenization(samples[language][index])
                    print(f"E5_WALL_OUTLIER language={language} query_index={index} token_count={token_count} diagnostic_tokenizer_ms={tokenizer_ms:.2f} embedding_wall_ms={row['embedding']:.2f} cuda_compute_ms=embedded_in_model_encode cuda_sync_ms=embedded_in_model_encode")
        print(f"QDRANT_RETRY_COUNT={runtime.vector_store.qdrant_retry_count}")
        print(f"QDRANT_FAILED_REQUEST_COUNT={runtime.vector_store.qdrant_failed_request_count}")
    finally:
        for hybrid in hybrids.values():
            hybrid.close()
        runtime.close()


if __name__ == "__main__":
    main()
