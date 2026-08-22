"""Exercise deterministic extractive composition against local Qdrant."""

import argparse
import json
import sys
import time
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.rag.generation.answer_composer import AnswerComposer  # noqa: E402
from app.rag.indexing.embedder import E5Embedder  # noqa: E402
from app.rag.indexing.vector_store import VectorStore  # noqa: E402
from app.rag.language_config import get_qdrant_target_lang  # noqa: E402
from app.rag.retrieval.retriever import Retriever  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-sentences", type=int, default=3)
    parser.add_argument("--max-answer-chars", type=int, default=600)
    parser.add_argument("--language", default="hi")
    args = parser.parse_args()
    store: VectorStore | None = None
    try:
        embedder = E5Embedder()
        store = VectorStore()
        started_at = time.perf_counter()
        chunks = Retriever(embedder, store).retrieve(args.query, args.top_k, get_qdrant_target_lang(args.language))
        retrieval_latency_ms = (time.perf_counter() - started_at) * 1_000
        answer = AnswerComposer().compose(args.query, chunks, args.max_sentences, args.max_answer_chars)
    except Exception as error:
        parser.error(f"Answer composer test failed: {type(error).__name__}: {error}")
    finally:
        if store is not None:
            store.close()

    print(f"QUERY: {args.query}")
    print(f"RETRIEVED CHUNKS: {len(chunks)}")
    print(f"COMPOSED ANSWER: {answer.text}")
    print(f"ANSWER LATENCY_MS: {answer.latency_ms:.2f}")
    print(f"QUERY EMBEDDING + QDRANT RETRIEVAL LATENCY_MS: {retrieval_latency_ms:.2f}")
    print(f"EVIDENCE COUNT: {len(answer.evidence)}")
    for evidence in answer.evidence:
        print("EVIDENCE PROVENANCE: " + json.dumps({
            "query_id": evidence.query_id, "passage_index": evidence.passage_index,
            "chunk_index": evidence.chunk_index, "retrieval_score": evidence.retrieval_score,
            "selected_sentence": evidence.source_sentence,
        }, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
