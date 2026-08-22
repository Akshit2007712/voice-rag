"""Inspect one bilingual hybrid-RAG answer without changing production data.

This development-only tool reads the frozen Qdrant Cloud collection, builds the
selected language's in-memory BM25 index, and prints each retrieval stage.  It
does not create document embeddings, write Qdrant points, or alter the corpus.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable
from pathlib import Path


BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.rag.bilingual_cloud_runtime import create_bilingual_cloud_runtime  # noqa: E402
from app.rag.language_config import get_qdrant_target_lang  # noqa: E402
from app.rag.retrieval.bm25_store import BM25Store  # noqa: E402
from app.rag.retrieval.hybrid_retriever import HybridRetrievedChunk, HybridRetriever  # noqa: E402
from app.rag.retrieval.retriever import RetrievedChunk  # noqa: E402


def _provenance(metadata: dict[str, object]) -> tuple[str, str, str]:
    """Return the stable identity used by production hybrid fusion."""
    return tuple(str(metadata.get(field, "")) for field in ("query_id", "passage_index", "chunk_index"))


def _render_hybrid_results(results: Iterable[HybridRetrievedChunk]) -> list[dict[str, object]]:
    """Serialize source/fused results with all ranking-score meanings explicit."""
    return [
        {
            "rank": result.rank,
            "query_id": result.metadata.get("query_id"),
            "passage_index": result.metadata.get("passage_index"),
            "chunk_index": result.metadata.get("chunk_index"),
            "chunk_strategy": result.metadata.get("chunk_strategy"),
            "semantic_score": result.semantic_score,
            "bm25_score": result.bm25_score,
            "fused_score": result.fused_score,
            "text": result.text,
        }
        for result in results
    ]


def deduplicate_evidence_by_provenance(
    chunks: Iterable[HybridRetrievedChunk],
) -> tuple[list[RetrievedChunk], list[tuple[str, str, str]]]:
    """Convert composer evidence while retaining the first fused rank per identity.

    Production fusion already uses this identity as its dictionary key.  The
    explicit defensive pass lets this diagnostic demonstrate whether an input
    duplicate exists without changing ranking or the production request path.
    BM25-only chunks remain excluded because AnswerComposer relies on semantic
    confidence and must not receive a fabricated cosine score.
    """
    evidence: list[RetrievedChunk] = []
    duplicate_provenance: list[tuple[str, str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for chunk in chunks:
        identity = _provenance(chunk.metadata)
        if identity in seen:
            duplicate_provenance.append(identity)
            continue
        seen.add(identity)
        converted = chunk.as_retrieved_chunk()
        if converted is not None:
            evidence.append(converted)
    return evidence, duplicate_provenance


def _render_composer_evidence(answer) -> list[dict[str, object]]:
    """Show source sentences so repeated provenance is not misleading."""
    return [
        {
            "query_id": item.query_id,
            "passage_index": item.passage_index,
            "chunk_index": item.chunk_index,
            "retrieval_score": item.retrieval_score,
            "source_sentence": item.source_sentence,
        }
        for item in answer.evidence
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", required=True, help="Hindi or English query to inspect.")
    parser.add_argument("--language", required=True, choices=("hi", "en"))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-sentences", type=int, default=3)
    args = parser.parse_args()
    if args.top_k < 1 or args.max_sentences < 1:
        parser.error("--top-k and --max-sentences must be at least 1")
    query = " ".join(args.query.split())
    if not query:
        parser.error("--query must contain text")

    target_lang = get_qdrant_target_lang(args.language)
    runtime, _ = create_bilingual_cloud_runtime(BACKEND_ROOT)
    hybrid_retriever: HybridRetriever | None = None
    try:
        # This startup-only read builds one isolated in-memory lexical corpus.
        # It is intentionally not a production request path and performs no writes.
        bm25_store = BM25Store.by_language_from_vector_store(runtime.vector_store, [target_lang])[target_lang]
        hybrid_retriever = HybridRetriever(runtime.retriever, bm25_store)
        retrieval = hybrid_retriever.retrieve(query, top_k=args.top_k, target_lang=target_lang)
        evidence, provenance_duplicates = deduplicate_evidence_by_provenance(retrieval.fused)
        answer = runtime.answer_composer.compose(query, evidence, max_sentences=args.max_sentences)

        print(f"QUERY: {query}")
        print(f"LANGUAGE: {args.language}")
        print(f"TARGET_LANG: {target_lang}")
        print("SEMANTIC_TOP_K:")
        print(json.dumps(_render_hybrid_results(retrieval.semantic), ensure_ascii=False, indent=2))
        print("BM25_TOP_K:")
        print(json.dumps(_render_hybrid_results(retrieval.lexical), ensure_ascii=False, indent=2))
        print("RRF_FUSED_TOP_K:")
        print(json.dumps(_render_hybrid_results(retrieval.fused), ensure_ascii=False, indent=2))
        print(f"FUSED_PROVENANCE_DUPLICATE_COUNT: {len(provenance_duplicates)}")
        print("FUSED_PROVENANCE_DUPLICATES:")
        print(json.dumps(provenance_duplicates, ensure_ascii=False, indent=2))
        print("DEDUPLICATED_COMPOSER_EVIDENCE:")
        print(json.dumps([
            {
                "query_id": item.metadata.get("query_id"),
                "passage_index": item.metadata.get("passage_index"),
                "chunk_index": item.metadata.get("chunk_index"),
                "retrieval_score": item.score,
                "text": item.text,
            }
            for item in evidence
        ], ensure_ascii=False, indent=2))
        print("FINAL_COMPOSED_ANSWER:")
        print(answer.text)
        print("FINAL_ANSWER_EVIDENCE:")
        print(json.dumps(_render_composer_evidence(answer), ensure_ascii=False, indent=2))
        print(f"ANSWER_IS_NO_ANSWER: {str(answer.is_no_answer).lower()}")
        print("DOCUMENT_EMBEDDINGS_CREATED=0")
        print("QDRANT_WRITES=0")
        print("CORPUS_CHANGED=0")
    finally:
        if hybrid_retriever is not None:
            hybrid_retriever.close()
        runtime.close()


if __name__ == "__main__":
    main()
