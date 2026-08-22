"""Read-only 50+50 sequential-versus-parallel hybrid equivalence validation."""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.rag.bilingual_cloud_runtime import create_bilingual_cloud_runtime  # noqa: E402
from app.rag.ingestion.dataset_loader import iter_msmarco_xi_records  # noqa: E402
from app.rag.language_config import get_qdrant_target_lang  # noqa: E402
from app.rag.retrieval.bm25_store import BM25Store  # noqa: E402
from app.rag.retrieval.hybrid_retriever import HybridRetriever  # noqa: E402
from app.rag.retrieval.maturity import assess_retrieval_maturity  # noqa: E402


def queries(language: str, count: int) -> list[str]:
    field = "Eng_Query" if language == "en" else "query"
    result: list[str] = []
    for record in iter_msmarco_xi_records(language, "validation", 500):
        if query := str(record.get(field, "")).strip():
            result.append(query)
        if len(result) == count:
            return result
    return result


def ranking_signature(results) -> tuple[tuple[tuple[str, str, str], float | None, float | None, float], ...]:
    return tuple((item.provenance, item.semantic_score, item.lexical_score, item.fused_score) for item in results)


def answer_signature(runtime, query: str, retrieval) -> tuple[object, ...]:
    answer = runtime.answer_composer.compose(query, [item for chunk in retrieval.fused if (item := chunk.as_retrieved_chunk())], 3)
    return answer.text, tuple((item.query_id, item.passage_index, item.chunk_index, item.retrieval_score, item.source_sentence) for item in answer.evidence), answer.confidence, answer.is_no_answer


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries-per-language", type=int, default=50)
    args = parser.parse_args()
    if args.queries_per_language != 50:
        parser.error("This validation requires exactly 50 queries per language")
    runtime, _ = create_bilingual_cloud_runtime(BACKEND_ROOT)
    hybrids: dict[str, HybridRetriever] = {}
    try:
        targets = [get_qdrant_target_lang(language) for language in ("hi", "en")]
        stores = BM25Store.by_language_from_vector_store(runtime.vector_store, targets)
        hybrids = {language: HybridRetriever(runtime.retriever, stores[get_qdrant_target_lang(language)]) for language in ("hi", "en")}
        blocked = False
        for language, hybrid in hybrids.items():
            sample = queries(language, args.queries_per_language)
            if len(sample) != args.queries_per_language:
                raise RuntimeError(f"Insufficient {language} validation queries")
            counts = {"semantic": 0, "bm25": 0, "rrf": 0, "final": 0}
            target = get_qdrant_target_lang(language)
            for index, query in enumerate(sample):
                sequential = hybrid.retrieve_sequential(query, 5, target)
                parallel = hybrid.retrieve(query, 5, target)
                semantic_equal = ranking_signature(sequential.semantic) == ranking_signature(parallel.semantic)
                bm25_equal = ranking_signature(sequential.lexical) == ranking_signature(parallel.lexical)
                rrf_equal = ranking_signature(sequential.fused) == ranking_signature(parallel.fused)
                maturity_equal = asdict(assess_retrieval_maturity(query, sequential.semantic, sequential.lexical, sequential.fused)) == asdict(assess_retrieval_maturity(query, parallel.semantic, parallel.lexical, parallel.fused))
                final_equal = maturity_equal and answer_signature(runtime, query, sequential) == answer_signature(runtime, query, parallel)
                for key, matched in (("semantic", semantic_equal), ("bm25", bm25_equal), ("rrf", rrf_equal), ("final", final_equal)):
                    counts[key] += int(matched)
                if not all((semantic_equal, bm25_equal, rrf_equal, final_equal)):
                    blocked = True
                    print(f"DIFFERENCE language={language} query_index={index} query={query!r} semantic={semantic_equal} bm25={bm25_equal} rrf={rrf_equal} final={final_equal}")
            print(f"LANGUAGE={language} SEMANTIC_EQUIVALENCE_RATE={counts['semantic'] / 50:.6f} BM25_EQUIVALENCE_RATE={counts['bm25'] / 50:.6f} RRF_EQUIVALENCE_RATE={counts['rrf'] / 50:.6f} FINAL_RESULT_EQUIVALENCE_RATE={counts['final'] / 50:.6f}")
        print(f"PARALLEL_FREEZE_STATUS={'BLOCKED_DIFFERENCE' if blocked else 'EQUIVALENT'}")
        return 2 if blocked else 0
    finally:
        for hybrid in hybrids.values():
            hybrid.close()
        runtime.close()


if __name__ == "__main__":
    raise SystemExit(main())
