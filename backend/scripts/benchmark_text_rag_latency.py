"""Measure warm steady-state deterministic text-RAG latency in one process."""

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Any

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.rag.ingestion.dataset_loader import iter_msmarco_xi_records  # noqa: E402
from app.rag.language_config import get_qdrant_target_lang  # noqa: E402
from app.rag.runtime import RAGRuntime  # noqa: E402


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    return sorted(values)[max(0, math.ceil(len(values) * percentile) - 1)]


def _print_metrics(title: str, values: list[float]) -> None:
    print(title)
    if not values:
        print("mean: -\nP50: -\nP70: -\nP100: -")
        return
    print(f"mean: {statistics.mean(values):.2f} ms")
    print(f"P50: {_percentile(values, 0.50):.2f} ms")
    print(f"P70: {_percentile(values, 0.70):.2f} ms")
    print(f"P100: {max(values):.2f} ms")


def _indexed_query_ids(runtime: RAGRuntime) -> list[str]:
    ids, seen, offset = [], set(), None
    while True:
        points, offset = runtime.vector_store.client.scroll(
            collection_name=runtime.vector_store.collection_name, limit=256,
            offset=offset, with_payload=True, with_vectors=False,
        )
        for point in points:
            query_id = (point.payload or {}).get("query_id")
            if query_id is not None and str(query_id) not in seen:
                seen.add(str(query_id))
                ids.append(str(query_id))
        if offset is None:
            return ids


def _queries_from_current_index(runtime: RAGRuntime, language: str, limit: int) -> list[str]:
    """Resolve local text only for IDs that already exist in this Qdrant collection."""
    ids = _indexed_query_ids(runtime)
    wanted, resolved = set(ids), {}
    for record in iter_msmarco_xi_records(language=language, batch_size=100):
        key, query = str(record.get("query_id")), record.get("query")
        if key in wanted and isinstance(query, str) and query.strip():
            resolved.setdefault(key, " ".join(query.split()))
            if len(resolved) == len(wanted):
                break
    return [resolved[key] for key in ids if key in resolved][:limit]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-sentences", type=int, default=3)
    parser.add_argument("--language", default="hi")
    parser.add_argument("--query", help="Run one measured query after warm-up instead of percentile benchmarking")
    args = parser.parse_args()
    if min(args.queries, args.top_k, args.max_sentences) < 1 or args.warmup < 0:
        parser.error("queries, top-k, and max-sentences must be at least 1; warmup must be non-negative")
    if args.query is not None and not args.query.strip():
        parser.error("--query must be non-empty when provided")

    runtime: RAGRuntime | None = None
    try:
        runtime = RAGRuntime()
        queries = _queries_from_current_index(runtime, args.language, args.queries)
        if not queries:
            parser.error("No real queries could be resolved from the current Qdrant collection")
        target_lang = get_qdrant_target_lang(args.language)
        print("MODEL LOAD COUNT: 1")
        print(f"DEVICE: {runtime.embedder.device}")
        print(f"QUERIES AVAILABLE/MEASURED: {len(queries)}")
        warmup_timings = []
        for index in range(args.warmup):
            started_at = time.perf_counter()
            runtime.measure_answer(queries[index % len(queries)], args.top_k, target_lang, args.max_sentences)
            warmup_timings.append((time.perf_counter() - started_at) * 1_000)
        print(f"WARMUP COUNT: {args.warmup}")
        if warmup_timings:
            print("WARMUP TIMINGS_MS: " + ", ".join(f"{value:.2f}" for value in warmup_timings))

        if args.query is not None:
            result = runtime.measure_answer(args.query, args.top_k, target_lang, args.max_sentences)
            print(f"QUERY: {args.query}")
            print(f"RETRIEVED CHUNKS: {len(result.chunks)}")
            print(f"FINAL ANSWER: {result.answer.text}")
            print(f"E5 EMBEDDING LATENCY_MS: {result.embedding_ms:.2f}")
            print(f"QDRANT SEARCH LATENCY_MS: {result.qdrant_search_ms:.2f}")
            print(f"RESULT CONVERSION LATENCY_MS: {result.result_conversion_ms:.2f}")
            print(f"ANSWER COMPOSER LATENCY_MS: {result.composer_ms:.2f}")
            print(f"TOTAL TEXT-RAG LATENCY_MS: {result.total_ms:.2f}")
            print(f"EVIDENCE COUNT: {len(result.answer.evidence)}")
            for evidence in result.answer.evidence:
                print("EVIDENCE PROVENANCE: " + json.dumps({
                    "query_id": evidence.query_id,
                    "passage_index": evidence.passage_index,
                    "chunk_index": evidence.chunk_index,
                    "retrieval_score": evidence.retrieval_score,
                    "selected_sentence": evidence.source_sentence,
                }, ensure_ascii=False, default=str))
            return

        measurements: list[dict[str, Any]] = []
        for query in queries:
            result = runtime.measure_answer(query, args.top_k, target_lang, args.max_sentences)
            measurements.append({
                "embedding": result.embedding_ms, "qdrant": result.qdrant_search_ms,
                "conversion": result.result_conversion_ms, "composer": result.composer_ms,
                "total": result.total_ms, "no_answer": result.answer.is_no_answer,
            })
    finally:
        if runtime is not None:
            runtime.close()

    _print_metrics("\nE5 QUERY EMBEDDING", [item["embedding"] for item in measurements])
    _print_metrics("\nQDRANT SEARCH", [item["qdrant"] for item in measurements])
    _print_metrics("\nRESULT CONVERSION", [item["conversion"] for item in measurements])
    _print_metrics("\nANSWER COMPOSER", [item["composer"] for item in measurements])
    _print_metrics("\nTOTAL TEXT RAG", [item["total"] for item in measurements])
    print(f"\nSUCCESSFUL ANSWERS: {sum(not item['no_answer'] for item in measurements)}")
    print(f"NO-ANSWER COUNT: {sum(item['no_answer'] for item in measurements)}")


if __name__ == "__main__":
    main()
