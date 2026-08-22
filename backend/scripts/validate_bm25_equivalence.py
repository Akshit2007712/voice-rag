"""Compare legacy full-corpus BM25 scoring with the production inverted index.

This is a read-only validation utility.  It loads the compact bilingual corpus
from Qdrant Cloud once, then performs all scoring in memory.  It creates no
embeddings and makes no Qdrant writes.
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from pathlib import Path

from dotenv import load_dotenv


BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.rag.bilingual_cloud_runtime import BILINGUAL_COLLECTION  # noqa: E402
from app.rag.indexing.vector_store import QdrantSettings, VectorStore  # noqa: E402
from app.rag.ingestion.dataset_loader import iter_msmarco_xi_records  # noqa: E402
from app.rag.language_config import get_qdrant_target_lang  # noqa: E402
from app.rag.retrieval.bm25_store import BM25Match, BM25Store  # noqa: E402


def percentile(values: list[float], percentile_value: int) -> float:
    """Return an inclusive percentile for a non-empty latency sample."""
    if len(values) == 1:
        return values[0]
    return statistics.quantiles(values, n=100, method="inclusive")[percentile_value - 1]


def compact_query_sample(language: str, count: int) -> list[str]:
    """Read the first deterministic non-empty validation queries for one language."""
    field = "Eng_Query" if language == "en" else "query"
    queries: list[str] = []
    for record in iter_msmarco_xi_records(language, "validation", batch_size=500):
        query = str(record.get(field, "")).strip()
        if query:
            queries.append(query)
        if len(queries) == count:
            return queries
    return queries


def match_identity(match: BM25Match) -> tuple[str, str, str]:
    """Return the exact provenance used by the BM25 ranking tie-break."""
    return match.document.provenance


def render_matches(matches: list[BM25Match]) -> list[dict[str, object]]:
    """Render comparison-safe result details without altering the corpus."""
    return [
        {"rank": match.rank, "provenance": match_identity(match), "score": match.score}
        for match in matches
    ]


def validate_language(store: BM25Store, language: str, queries: list[str], top_k: int) -> dict[str, object]:
    """Run exact output and separate legacy/optimized latency comparisons."""
    differences: list[dict[str, object]] = []
    ranking_difference_count = 0
    score_difference_count = 0
    top1_matches = 0
    exact_top_k_matches = 0
    overlap_ratios: list[float] = []
    max_score_difference = 0.0

    # Compare each pair before timing so any semantic discrepancy is explicit.
    for query_index, query in enumerate(queries):
        old_results = store.search_full_corpus_reference(query, top_k=top_k)
        new_results = store.search(query, top_k=top_k)
        old_ids = [match_identity(match) for match in old_results]
        new_ids = [match_identity(match) for match in new_results]
        if old_ids[:1] == new_ids[:1]:
            top1_matches += 1
        if old_ids == new_ids:
            exact_top_k_matches += 1
        overlap_ratios.append(len(set(old_ids) & set(new_ids)) / max(len(old_ids), 1))

        old_scores = {match_identity(match): match.score for match in old_results}
        new_scores = {match_identity(match): match.score for match in new_results}
        score_difference = max(
            (abs(old_scores.get(identity, 0.0) - new_scores.get(identity, 0.0)) for identity in set(old_scores) | set(new_scores)),
            default=0.0,
        )
        max_score_difference = max(max_score_difference, score_difference)
        if old_ids != new_ids:
            ranking_difference_count += 1
        if score_difference != 0.0:
            score_difference_count += 1
        if old_ids != new_ids or score_difference != 0.0:
            differences.append(
                {
                    "query_index": query_index,
                    "query": query,
                    "old_top_k": render_matches(old_results),
                    "new_top_k": render_matches(new_results),
                    "max_score_difference": score_difference,
                }
            )

    # Warm each execution strategy without including warm-up in the measurements.
    for query in queries[:3]:
        store.search_full_corpus_reference(query, top_k=top_k)
        store.search(query, top_k=top_k)
    old_latencies: list[float] = []
    new_latencies: list[float] = []
    for query in queries:
        started_at = time.perf_counter()
        store.search_full_corpus_reference(query, top_k=top_k)
        old_latencies.append((time.perf_counter() - started_at) * 1_000)
    for query in queries:
        started_at = time.perf_counter()
        store.search(query, top_k=top_k)
        new_latencies.append((time.perf_counter() - started_at) * 1_000)

    return {
        "language": language,
        "query_count": len(queries),
        "top1_match_rate": top1_matches / len(queries),
        "exact_top_k_match_rate": exact_top_k_matches / len(queries),
        "mean_top_k_overlap": statistics.fmean(overlap_ratios),
        "max_score_difference": max_score_difference,
        "ranking_difference_count": ranking_difference_count,
        "score_difference_count": score_difference_count,
        "differences": differences,
        "old_latency_ms": {f"p{value}": percentile(old_latencies, value) for value in (50, 70, 95)} | {"p100": max(old_latencies)},
        "new_latency_ms": {f"p{value}": percentile(new_latencies, value) for value in (50, 70, 95)} | {"p100": max(new_latencies)},
        "speedup_p50": percentile(old_latencies, 50) / percentile(new_latencies, 50),
    }


def print_report(report: dict[str, object]) -> None:
    """Print the requested per-language validation report."""
    language = str(report["language"])
    print(f"LANGUAGE={language}")
    print(f"QUERY_COUNT={report['query_count']}")
    print(f"TOP1_MATCH_RATE={float(report['top1_match_rate']):.6f}")
    print(f"EXACT_TOP_K_MATCH_RATE={float(report['exact_top_k_match_rate']):.6f}")
    print(f"MEAN_TOP_K_OVERLAP={float(report['mean_top_k_overlap']):.6f}")
    print(f"MAX_SCORE_DIFFERENCE={float(report['max_score_difference']):.17g}")
    print(f"QUERIES_WITH_ANY_RANKING_DIFFERENCE={report['ranking_difference_count']}")
    print(f"QUERIES_WITH_ANY_SCORE_DIFFERENCE={report['score_difference_count']}")
    for name in ("old_latency_ms", "new_latency_ms"):
        latency = report[name]
        assert isinstance(latency, dict)
        print(
            f"{name.upper()} P50={latency['p50']:.3f}ms P70={latency['p70']:.3f}ms "
            f"P95={latency['p95']:.3f}ms P100={latency['p100']:.3f}ms"
        )
    print(f"P50_SPEEDUP={float(report['speedup_p50']):.2f}x")
    for difference in report["differences"]:
        assert isinstance(difference, dict)
        print("BM25_DIFFERENCE_START")
        print(f"QUERY_INDEX={difference['query_index']}")
        print(f"QUERY={difference['query']}")
        print(f"OLD_TOP_K={difference['old_top_k']}")
        print(f"NEW_TOP_K={difference['new_top_k']}")
        print(f"MAX_SCORE_DIFFERENCE_FOR_QUERY={difference['max_score_difference']:.17g}")
        print("BM25_DIFFERENCE_END")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries-per-language", type=int, default=50)
    parser.add_argument("--top-k", type=int, default=5)
    arguments = parser.parse_args()
    if arguments.queries_per_language != 50:
        parser.error("This freeze validation intentionally requires exactly 50 queries per language")
    if arguments.top_k < 1:
        parser.error("--top-k must be at least 1")

    load_dotenv(BACKEND_ROOT / ".env")
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")
    if not qdrant_url or not qdrant_api_key:
        raise RuntimeError("QDRANT_URL and QDRANT_API_KEY are required for compact-corpus validation")
    print("QDRANT_MODE=remote")
    print(f"QDRANT_COLLECTION={BILINGUAL_COLLECTION}")
    print("QDRANT_API_KEY_PRESENT=true")
    print("DOCUMENT_EMBEDDINGS_CREATED=0")
    print("QDRANT_WRITES=0")

    vector_store = VectorStore(
        QdrantSettings(mode="remote", url=qdrant_url, api_key=qdrant_api_key, collection_name=BILINGUAL_COLLECTION)
    )
    try:
        if not vector_store.collection_exists():
            raise RuntimeError(f"Required collection does not exist: {BILINGUAL_COLLECTION}")
        print(f"QDRANT_COLLECTION_POINT_COUNT={vector_store.point_count()}")
        target_languages = [get_qdrant_target_lang(language) for language in ("hi", "en")]
        stores = BM25Store.by_language_from_vector_store(vector_store, target_languages)
        reports = []
        for language in ("hi", "en"):
            queries = compact_query_sample(language, arguments.queries_per_language)
            if len(queries) != arguments.queries_per_language:
                raise RuntimeError(f"Only {len(queries)} deterministic {language} queries were available")
            reports.append(validate_language(stores[get_qdrant_target_lang(language)], language, queries, arguments.top_k))
        for report in reports:
            print_report(report)
        if any(report["differences"] for report in reports):
            print("BM25_FREEZE_STATUS=BLOCKED_RANKING_DIFFERENCE")
            return 2
        print("BM25_FREEZE_STATUS=EQUIVALENT")
        return 0
    finally:
        vector_store.close()


if __name__ == "__main__":
    raise SystemExit(main())
