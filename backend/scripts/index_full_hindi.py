"""Build and verify a dedicated full MSMARCO-XI Hindi validation Qdrant index."""

from __future__ import annotations

import argparse
import random
import sys
import time
from dataclasses import replace
from itertools import islice
from pathlib import Path

import pyarrow.parquet as pq
import torch

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.rag.indexing.embedder import E5Embedder  # noqa: E402
from app.rag.indexing.indexer import IndexingStats, index_raw_records  # noqa: E402
from app.rag.indexing.vector_store import QdrantSettings, VectorStore, strategy_aware_point_id  # noqa: E402
from app.rag.ingestion.dataset_loader import (  # noqa: E402
    DEFAULT_LANGUAGE,
    DEFAULT_SPLIT,
    get_msmarco_xi_local_file_path,
    iter_msmarco_xi_records,
)
from app.rag.language_config import get_qdrant_target_lang  # noqa: E402
from app.rag.retrieval.bm25_store import BM25Store  # noqa: E402


FULL_HINDI_COLLECTION = "msmarco_xi_hindi_full"
KNOWN_QUERY = "फिलाडेल्फिया लैंकेस्टर से कितनी दूर है?"
KNOWN_QUERY_ID = "232017"


def _add_stats(total: IndexingStats, batch: IndexingStats) -> None:
    """Accumulate bounded batch indexing statistics."""
    total.raw_records += batch.raw_records
    total.retrieval_documents += batch.retrieval_documents
    total.chunks += batch.chunks
    total.points_upserted += batch.points_upserted
    total.embedding_time_s += batch.embedding_time_s
    total.upsert_time_s += batch.upsert_time_s
    total.malformed_records += batch.malformed_records


def _dataset_row_count(path: Path) -> int | None:
    """Read Parquet metadata only; never materialize the dataset for counting."""
    parquet_file = pq.ParquetFile(path)
    try:
        metadata = parquet_file.metadata
        return metadata.num_rows if metadata is not None else None
    finally:
        parquet_file.close()


def _current_point_count(store: VectorStore) -> int:
    """Return zero for a collection that has not been created yet."""
    return store.point_count() if store.client.collection_exists(store.collection_name) else 0


def _print_progress(stats: IndexingStats, started_at: float) -> None:
    elapsed = time.perf_counter() - started_at
    rows_per_second = stats.raw_records / elapsed if elapsed else 0.0
    chunks_per_second = stats.chunks / elapsed if elapsed else 0.0
    print(
        "PROGRESS "
        f"ROWS_PROCESSED={stats.raw_records} "
        f"PASSAGES_PROCESSED={stats.retrieval_documents} "
        f"CHUNKS_CREATED={stats.chunks} "
        f"EMBEDDINGS_CREATED={stats.chunks} "
        f"QDRANT_POINTS_UPSERTED={stats.points_upserted} "
        f"ELAPSED_TIME_S={elapsed:.1f} "
        f"ROWS_PER_SECOND={rows_per_second:.2f} "
        f"CHUNKS_PER_SECOND={chunks_per_second:.2f}",
        flush=True,
    )


def _reservoir_sample(sample: list[tuple[str, str]], record: dict, seen: int, rng: random.Random) -> None:
    """Keep a tiny deterministic query sample without retaining raw dataset records."""
    query = record.get("query")
    query_id = record.get("query_id")
    if not isinstance(query, str) or not query.strip() or query_id is None:
        return
    item = (str(query_id), " ".join(query.split()))
    if len(sample) < 3:
        sample.append(item)
    else:
        replacement_index = rng.randrange(seen)
        if replacement_index < len(sample):
            sample[replacement_index] = item


def _print_payload_samples(store: VectorStore) -> None:
    """Show a few Qdrant payloads to validate retained provenance metadata."""
    points, _ = store.client.scroll(
        collection_name=store.collection_name,
        limit=3,
        with_payload=True,
        with_vectors=False,
    )
    for index, point in enumerate(points, start=1):
        payload = dict(point.payload or {})
        print(
            f"PAYLOAD_SAMPLE_{index}: "
            f"query_id={payload.get('query_id')} "
            f"passage_index={payload.get('passage_index')} "
            f"chunk_index={payload.get('chunk_index')} "
            f"target_lang={payload.get('target_lang')} "
            f"chunk_strategy={payload.get('chunk_strategy')} "
            f"is_selected={payload.get('is_selected')} "
            f"query_type={payload.get('query_type')}",
            flush=True,
        )


def _print_retrieval_smoke_tests(
    embedder: E5Embedder,
    store: VectorStore,
    target_lang: str,
    sampled_queries: list[tuple[str, str]],
    top_k: int,
) -> None:
    """Run known and sampled Hindi retrieval checks against the completed collection."""
    queries = [(KNOWN_QUERY_ID, KNOWN_QUERY), *sampled_queries]
    seen: set[tuple[str, str]] = set()
    for expected_query_id, query in queries:
        key = (expected_query_id, query)
        if key in seen:
            continue
        seen.add(key)
        results = store.search(embedder.embed_query(query), limit=top_k, target_lang=target_lang)
        result_ids = [str((result.payload or {}).get("query_id", "")) for result in results]
        print(
            f"RETRIEVAL_SMOKE expected_query_id={expected_query_id} "
            f"top1_query_id={result_ids[0] if result_ids else None} "
            f"expected_in_top_k={expected_query_id in result_ids} "
            f"query={query}",
            flush=True,
        )


def _build_bm25_once(store: VectorStore, target_lang: str) -> None:
    """Verify the existing startup-time BM25 construction against the full corpus."""
    started_at = time.perf_counter()
    bm25 = BM25Store.from_vector_store(store, target_lang)
    elapsed = time.perf_counter() - started_at
    text_bytes = sum(len(document.text.encode("utf-8")) for document in bm25.documents)
    print(f"BM25_CORPUS_SIZE={len(bm25.documents)}", flush=True)
    print(f"BM25_BUILD_TIME_S={elapsed:.3f}", flush=True)
    print(f"BM25_TEXT_BYTES_APPROX={text_bytes}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--language", default=DEFAULT_LANGUAGE)
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--embedding-batch-size", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--collection", default=FULL_HINDI_COLLECTION)
    parser.add_argument("--reset-collection", action="store_true")
    parser.add_argument("--skip-bm25", action="store_true")
    args = parser.parse_args()
    if min(args.batch_size, args.embedding_batch_size, args.max_tokens, args.top_k) < 1:
        parser.error("all numeric options must be at least 1")
    if not args.collection.strip():
        parser.error("--collection must be non-empty")
    if args.collection == "msmarco_xi_hindi":
        parser.error("The full-index script refuses to modify the development collection msmarco_xi_hindi")

    local_path = get_msmarco_xi_local_file_path(args.language, args.split)
    dataset_rows = _dataset_row_count(local_path)
    target_lang = get_qdrant_target_lang(args.language)
    settings = replace(QdrantSettings.from_environment(), collection_name=args.collection)
    store = VectorStore(settings, point_id_builder=strategy_aware_point_id)
    try:
        starting_point_count = _current_point_count(store)
        print(f"DATASET_PATH={local_path}", flush=True)
        print(f"DATASET_ROWS={dataset_rows}", flush=True)
        print(f"CURRENT_COLLECTION={store.collection_name}", flush=True)
        print(f"CURRENT_POINT_COUNT={starting_point_count}", flush=True)
        print(f"TARGET_LANGUAGE={target_lang}", flush=True)
        print(f"RESET_COLLECTION={args.reset_collection}", flush=True)
        print("POINT_ID_SCHEME=query_id + passage_index + chunk_index + chunk_strategy", flush=True)

        embedder = E5Embedder(batch_size=args.embedding_batch_size)
        print(f"E5_DEVICE={embedder.device}", flush=True)
        print(f"CUDA_AVAILABLE={torch.cuda.is_available()}", flush=True)
        print(f"E5_DIMENSION={embedder.dimension}", flush=True)
        store.ensure_collection(embedder.dimension, reset=args.reset_collection)

        started_at = time.perf_counter()
        stats = IndexingStats()
        sample: list[tuple[str, str]] = []
        rng = random.Random(42)
        records_seen = 0
        record_iterator = iter_msmarco_xi_records(args.language, args.split, args.batch_size)
        while batch := list(islice(record_iterator, args.batch_size)):
            for record in batch:
                records_seen += 1
                _reservoir_sample(sample, record, records_seen, rng)
            _add_stats(stats, index_raw_records(batch, embedder, store, args.max_tokens))
            _print_progress(stats, started_at)

        elapsed = time.perf_counter() - started_at
        final_point_count = store.point_count()
        print(f"TOTAL_INDEXING_TIME_S={elapsed:.3f}", flush=True)
        print(f"ROWS_PROCESSED={stats.raw_records}", flush=True)
        print(f"PASSAGES_PROCESSED={stats.retrieval_documents}", flush=True)
        print(f"CHUNKS_CREATED={stats.chunks}", flush=True)
        print(f"QDRANT_POINTS_UPSERTED={stats.points_upserted}", flush=True)
        print(f"QDRANT_FINAL_POINT_COUNT={final_point_count}", flush=True)
        print(f"MALFORMED_RECORDS_SKIPPED={stats.malformed_records}", flush=True)
        print(f"ROWS_PER_SECOND={stats.raw_records / elapsed if elapsed else 0.0:.2f}", flush=True)
        print(f"CHUNKS_PER_SECOND={stats.chunks / elapsed if elapsed else 0.0:.2f}", flush=True)
        print(f"EMBEDDING_TIME_S={stats.embedding_time_s:.3f}", flush=True)
        print(f"QDRANT_UPSERT_TIME_S={stats.upsert_time_s:.3f}", flush=True)
        print(f"QDRANT_POINT_DELTA_VS_START={final_point_count - starting_point_count}", flush=True)
        print(f"FRESH_COLLECTION_POINT_COUNT_MATCHES_CHUNKS={starting_point_count == 0 and final_point_count == stats.chunks}", flush=True)
        _print_payload_samples(store)
        _print_retrieval_smoke_tests(embedder, store, target_lang, sample, args.top_k)
        if not args.skip_bm25:
            _build_bm25_once(store, target_lang)
    finally:
        store.close()


if __name__ == "__main__":
    main()
