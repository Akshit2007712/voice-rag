"""Build the final bilingual compact Qdrant Cloud collection behind a size gate."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.rag.analysis.bilingual_compact import (  # noqa: E402
    QDRANT_VECTOR_DIMENSION,
    bilingual_payload,
    build_bilingual_point_id,
    english_policy_a_documents,
)
from app.rag.indexing.embedder import E5Embedder  # noqa: E402
from app.rag.ingestion.chunker import Chunk, get_e5_tokenizer, iter_document_chunks  # noqa: E402
from app.rag.ingestion.dataset_loader import iter_msmarco_xi_records  # noqa: E402
from scripts.analyze_compact_english_corpus import analyze  # noqa: E402
from scripts.migrate_local_qdrant_to_server import (  # noqa: E402
    DEFAULT_SOURCE_PATH,
    _storage_db_path,
    iter_source_points,
)


FINAL_COLLECTION = "msmarco_xi_bilingual_compact"
FULL_HINDI_COLLECTION = "msmarco_xi_hindi_full"
HINDI_SOURCE_DB = _storage_db_path(DEFAULT_SOURCE_PATH, FULL_HINDI_COLLECTION)


def _selected(value: object) -> bool:
    return value is True or value == 1


def _ensure_target(client: QdrantClient, collection: str, reset: bool) -> None:
    if collection == FULL_HINDI_COLLECTION:
        raise ValueError("The bilingual indexer refuses to modify msmarco_xi_hindi_full")
    exists = client.collection_exists(collection)
    if exists and reset:
        client.delete_collection(collection)
        exists = False
    if not exists:
        client.create_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=QDRANT_VECTOR_DIMENSION, distance=Distance.COSINE),
        )
        return
    vector_config = client.get_collection(collection).config.params.vectors
    if vector_config.size != QDRANT_VECTOR_DIMENSION or vector_config.distance != Distance.COSINE:
        raise ValueError("Existing bilingual collection must use 768-dimensional cosine vectors")


def _upsert(client: QdrantClient, collection: str, points: list[PointStruct]) -> None:
    if points:
        client.upsert(collection_name=collection, points=points, wait=True)


def _copy_hindi_policy_a(client: QdrantClient, collection: str, batch_size: int) -> int:
    """Copy existing selected Hindi vectors directly; never instantiate E5 here."""
    copied = 0
    pending: list[PointStruct] = []
    for _, source_point in iter_source_points(HINDI_SOURCE_DB, read_batch_size=batch_size):
        payload = source_point.payload
        if payload.get("target_lang") != "hin_Deva" or not _selected(payload.get("is_selected")):
            continue
        text = str(payload.get("text", "")).strip()
        if not text:
            continue
        target_payload = bilingual_payload("hi", text, payload)
        pending.append(
            PointStruct(
                id=build_bilingual_point_id("hi", target_payload),
                vector=source_point.vector,
                payload=target_payload,
            )
        )
        if len(pending) >= batch_size:
            _upsert(client, collection, pending)
            copied += len(pending)
            pending.clear()
    _upsert(client, collection, pending)
    return copied + len(pending)


def _embed_english_policy_a(client: QdrantClient, collection: str, batch_size: int) -> int:
    """Embed only English Policy-A chunks after the preflight has passed."""
    embedder = E5Embedder(batch_size=batch_size)
    if embedder.dimension != QDRANT_VECTOR_DIMENSION:
        raise ValueError(f"E5 dimension {embedder.dimension} does not match required 768")
    tokenizer = get_e5_tokenizer()
    embedded = 0
    pending: list[Chunk] = []

    def flush() -> None:
        nonlocal embedded
        if not pending:
            return
        vectors = embedder.embed_passages([chunk.text for chunk in pending])
        points = []
        for chunk, vector in zip(pending, vectors, strict=True):
            payload = bilingual_payload("en", chunk.text, chunk.metadata)
            points.append(PointStruct(id=build_bilingual_point_id("en", payload), vector=vector.tolist(), payload=payload))
        _upsert(client, collection, points)
        embedded += len(points)
        pending.clear()

    for record in iter_msmarco_xi_records("en", "validation", batch_size=500):
        for chunk in iter_document_chunks(english_policy_a_documents(record), max_tokens=256, tokenizer=tokenizer):
            pending.append(chunk)
            if len(pending) >= batch_size:
                flush()
    flush()
    return embedded


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--collection", default=FINAL_COLLECTION)
    parser.add_argument("--reset-target", action="store_true")
    parser.add_argument("--execute", action="store_true", help="Permit Qdrant Cloud writes after size preflight.")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("batch-size must be at least 1")

    print("HINDI_DOCUMENT_EMBEDDINGS_CREATED=0")
    print("NEON_WRITES=0")
    print("FULL_HINDI_COLLECTION_WRITES=0")
    report = analyze(batch_size=500)
    print(f"ENGLISH_COMPACT_CHUNKS={report['english_dataset_statistics']['compact_chunks']}")
    print(f"ESTIMATED_FINAL_BILINGUAL_POINTS={report['estimated_final_bilingual_points']}")
    print(f"CONSERVATIVE_ESTIMATED_QDRANT_GIB={report['conservative_estimated_bilingual_qdrant_gib']:.3f}")
    print(f"QDRANT_4_GIB_TARGET_SAFE={str(report['estimated_final_collection_safe']).lower()}")
    if not report["estimated_final_collection_safe"]:
        raise RuntimeError("Bilingual size estimate exceeds the 4 GiB Qdrant target; no embeddings or writes occurred")
    if not args.execute:
        print("EXECUTE_REQUIRED_FOR_QDRANT_WRITES=true")
        return

    load_dotenv(BACKEND_ROOT / ".env")
    url, api_key = os.getenv("QDRANT_URL"), os.getenv("QDRANT_API_KEY")
    if not url or not api_key:
        raise RuntimeError("QDRANT_URL and QDRANT_API_KEY are required for Qdrant Cloud indexing")
    client = QdrantClient(url=url, api_key=api_key)
    try:
        _ensure_target(client, args.collection, args.reset_target)
        hindi_reused = _copy_hindi_policy_a(client, args.collection, args.batch_size)
        english_embedded = _embed_english_policy_a(client, args.collection, args.batch_size)
        total = int(client.count(collection_name=args.collection, exact=True).count)
    finally:
        client.close()
    print(f"HINDI_VECTORS_REUSED={hindi_reused}")
    print(f"ENGLISH_DOCUMENT_EMBEDDINGS_CREATED={english_embedded}")
    print(f"FINAL_BILINGUAL_POINT_COUNT={total}")


if __name__ == "__main__":
    main()
