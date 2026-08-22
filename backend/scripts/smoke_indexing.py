"""Smoke-test bounded E5 indexing and persistent local Qdrant retrieval."""

import argparse
import sys
from pathlib import Path

import torch

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.rag.indexing.embedder import E5Embedder  # noqa: E402
from app.rag.indexing.indexer import index_raw_records  # noqa: E402
from app.rag.indexing.vector_store import QdrantSettings, VectorStore  # noqa: E402
from app.rag.ingestion.dataset_loader import iter_msmarco_xi_records  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=int, default=100)
    parser.add_argument("--loader-batch-size", type=int, default=100)
    parser.add_argument("--embedding-batch-size", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--reset", action="store_true")
    args = parser.parse_args()
    if min(args.records, args.loader_batch_size, args.embedding_batch_size, args.max_tokens, args.top_k) < 1:
        parser.error("all numeric options must be at least 1")

    try:
        embedder = E5Embedder(batch_size=args.embedding_batch_size)
        print(f"CUDA AVAILABLE: {torch.cuda.is_available()}")
        print(f"E5 DEVICE: {embedder.device}")
        if embedder.device == "cuda":
            print(f"GPU NAME: {torch.cuda.get_device_name(0)}")
        print(f"E5 EMBEDDING DIMENSION: {embedder.dimension}")
        settings = QdrantSettings.from_environment()
        store = VectorStore(settings)
        store.ensure_collection(embedder.dimension, reset=args.reset)

        query_holder: dict[str, str] = {}
        def records():
            for index, record in enumerate(iter_msmarco_xi_records(batch_size=args.loader_batch_size)):
                if index >= args.records:
                    return
                query = record.get("query")
                if "query" not in query_holder and isinstance(query, str) and query.strip():
                    query_holder["query"] = query
                yield record

        stats = index_raw_records(records(), embedder, store, args.max_tokens)
        if "query" not in query_holder:
            raise ValueError("No usable Hindi query was found in the sampled records")
        count_before_reopen = store.point_count()
        store.close()

        reopened_store = VectorStore(settings)
        count_after_reopen = reopened_store.point_count()
        query = query_holder["query"]
        results = reopened_store.search(embedder.embed_query(query), limit=args.top_k)

        print(f"RAW RECORDS PROCESSED: {stats.raw_records}")
        print(f"RETRIEVAL DOCUMENTS: {stats.retrieval_documents}")
        print(f"FINAL CHUNKS: {stats.chunks}")
        print(f"VECTORS EMBEDDED: {stats.chunks}")
        print(f"QDRANT POINTS UPSERTED: {stats.points_upserted}")
        print(f"COLLECTION: {reopened_store.collection_name}")
        print(f"QDRANT PATH: {settings.path}")
        print(f"EMBEDDING TIME S: {stats.embedding_time_s:.3f}")
        print(f"QDRANT UPSERT TIME S: {stats.upsert_time_s:.3f}")
        print(f"POINT COUNT BEFORE REOPEN: {count_before_reopen}")
        print(f"POINT COUNT AFTER REOPEN: {count_after_reopen}")
        print(f"HINDI QUERY: {query}")
        for rank, result in enumerate(results, start=1):
            payload = result.payload or {}
            print(f"RESULT {rank}: score={result.score:.4f} text={str(payload.get('text', ''))[:160]}")
            print(f"  query_id={payload.get('query_id')} passage_index={payload.get('passage_index')} chunk_index={payload.get('chunk_index')} is_selected={payload.get('is_selected')} chunk_strategy={payload.get('chunk_strategy')}")
        relevant_in_top_k = any((result.payload or {}).get("is_selected") == 1 for result in results)
        print(f"GROUND_TRUTH_RELEVANT_IN_TOP_K: {relevant_in_top_k}")
        reopened_store.close()
    except Exception as error:
        parser.error(f"Indexing smoke test failed: {type(error).__name__}: {error}")


if __name__ == "__main__":
    main()
