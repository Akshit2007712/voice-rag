"""Bounded raw-record → chunk → embed → Qdrant indexing workflow."""

import time
from dataclasses import dataclass
from typing import Iterable

from app.rag.indexing.embedder import E5Embedder
from app.rag.indexing.vector_store import VectorStore
from app.rag.ingestion.chunker import Chunk, chunk_retrieval_document
from app.rag.ingestion.preprocessor import preprocess_msmarco_xi_record


@dataclass
class IndexingStats:
    raw_records: int = 0
    retrieval_documents: int = 0
    chunks: int = 0
    points_upserted: int = 0
    embedding_time_s: float = 0.0
    upsert_time_s: float = 0.0
    malformed_records: int = 0


def index_raw_records(records: Iterable[dict], embedder: E5Embedder, store: VectorStore, max_tokens: int) -> IndexingStats:
    """Index records incrementally, retaining only one embedding batch of chunks."""
    stats = IndexingStats()
    buffer: list[Chunk] = []

    def flush() -> None:
        nonlocal buffer
        if not buffer:
            return
        started_at = time.perf_counter()
        vectors = embedder.embed_passages([chunk.text for chunk in buffer])
        stats.embedding_time_s += time.perf_counter() - started_at
        started_at = time.perf_counter()
        stats.points_upserted += store.upsert_chunks(buffer, vectors)
        stats.upsert_time_s += time.perf_counter() - started_at
        buffer = []

    for record in records:
        stats.raw_records += 1
        try:
            documents = preprocess_msmarco_xi_record(record)
        except (TypeError, ValueError):
            stats.malformed_records += 1
            continue
        stats.retrieval_documents += len(documents)
        for document in documents:
            chunks = chunk_retrieval_document(document, max_tokens=max_tokens)
            stats.chunks += len(chunks)
            buffer.extend(chunks)
            while len(buffer) >= embedder.batch_size:
                current_batch, buffer = buffer[: embedder.batch_size], buffer[embedder.batch_size :]
                started_at = time.perf_counter()
                vectors = embedder.embed_passages([chunk.text for chunk in current_batch])
                stats.embedding_time_s += time.perf_counter() - started_at
                started_at = time.perf_counter()
                stats.points_upserted += store.upsert_chunks(current_batch, vectors)
                stats.upsert_time_s += time.perf_counter() - started_at
    flush()
    return stats
