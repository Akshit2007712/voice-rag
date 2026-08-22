"""Ground-truth benchmark for compact Hindi policies using existing Qdrant vectors.

Only disposable policy collections are written. MSMARCO-XI selected-passage
provenance is the correctness oracle; the full embedded Qdrant index is never
opened as a search baseline.
"""

from __future__ import annotations

import argparse
import base64
import json
import pickle
import sqlite3
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.rag.analysis.compact_hindi_policy import POLICY_A, POLICY_D, POLICY_E, selected_documents_for_policy  # noqa: E402
from app.rag.analysis.compact_policy_recommendation import (  # noqa: E402
    recommend_compact_policy,
    recommendation_interpretation,
)
from app.rag.indexing.embedder import E5Embedder  # noqa: E402
from app.rag.indexing.vector_store import build_strategy_aware_point_identity, strategy_aware_point_id  # noqa: E402
from app.rag.ingestion.chunker import get_e5_tokenizer, iter_document_chunks  # noqa: E402
from app.rag.ingestion.dataset_loader import iter_msmarco_xi_records  # noqa: E402
from scripts.migrate_local_qdrant_to_server import (  # noqa: E402
    DEFAULT_SOURCE_PATH, _deserialize_source_point, _storage_db_path,
)


POLICIES = (POLICY_A, POLICY_D, POLICY_E)
COLLECTIONS = {
    POLICY_A: "msmarco_xi_hindi_policy_a_benchmark",
    POLICY_D: "msmarco_xi_hindi_policy_d_benchmark",
    POLICY_E: "msmarco_xi_hindi_policy_e_benchmark",
}
CHUNK_COUNTS = {POLICY_A: 58_427, POLICY_D: 103_502, POLICY_E: 158_537}
BENCHMARK_PATH = BACKEND_ROOT / "data" / "qdrant_policy_benchmarks"
VALIDATION_PATH = BACKEND_ROOT / "data" / "validation" / "compact_hindi_policy_validation.json"
SOURCE_DB = _storage_db_path(DEFAULT_SOURCE_PATH, "msmarco_xi_hindi_full")
VERIFICATION_DB = BACKEND_ROOT / "tmp" / "full_index_verify.sqlite"
KNOWN_OVERWRITE_QUERY_ID = "177416"


@dataclass(frozen=True)
class ChunkDescriptor:
    """One generated compact-policy chunk and its exact full-index identity."""

    point_id: str
    identity: str
    policy: str
    query_id: str
    row_number: int
    passage_index: int
    chunk_index: int
    chunk_strategy: str
    text_hash: str
    preview: str


def _known_overwritten_source_chunks() -> set[tuple[str, int, str]]:
    """Return only source variants proven unrecoverable by the full-index audit.

    The full index's deterministic ID does not contain a dataset-row identity.
    For query 177416, the later source row overwrote three earlier, different
    passages with the same point IDs.  The verifier records both variants, so
    this read-only lookup excludes only the lost earlier variants.
    """
    if not VERIFICATION_DB.is_file():
        raise FileNotFoundError(
            "The full-index verification database is required to classify the "
            f"known overwrite safely: {VERIFICATION_DB}"
        )
    source = sqlite3.connect(f"file:{VERIFICATION_DB.as_posix()}?mode=ro", uri=True)
    try:
        rows = source.execute(
            """
            SELECT point_id, original_row_number, original_text_hash
            FROM collisions
            WHERE original_query_id = ?
              AND original_text_hash != duplicate_text_hash
            """,
            (KNOWN_OVERWRITE_QUERY_ID,),
        ).fetchall()
    finally:
        source.close()
    return {(str(point_id), int(row_number), str(text_hash)) for point_id, row_number, text_hash in rows}


def _encoded_local_id(point_id: str) -> str:
    return base64.b64encode(pickle.dumps(point_id, protocol=4)).decode("ascii")


def _percentile(values: list[float], q: float) -> float:
    return float(np.percentile(values, q)) if values else 0.0


def _load_validation(limit: int) -> dict[str, dict[str, object]]:
    raw = json.loads(VALIDATION_PATH.read_text(encoding="utf-8"))
    merged: dict[str, dict[str, object]] = {}
    for entries in raw.values():
        for entry in entries:
            key = str(entry["query_id"])
            merged.setdefault(key, entry)
            if len(merged) >= limit:
                return merged
    return merged


def _records_for_validation(query_ids: set[str]) -> dict[str, dict[str, object]]:
    found: dict[str, dict[str, object]] = {}
    for record in iter_msmarco_xi_records("hi", "validation", batch_size=500):
        key = str(record.get("query_id"))
        if key in query_ids:
            found[key] = record
            if len(found) == len(query_ids):
                return found
    return found


def _source_points(connection: sqlite3.Connection, ids: list[str]) -> list[Any]:
    if not ids:
        return []
    encoded = [_encoded_local_id(point_id) for point_id in ids]
    placeholders = ",".join("?" for _ in encoded)
    rows = connection.execute(f"SELECT point FROM points WHERE id IN ({placeholders})", encoded).fetchall()
    points = [_deserialize_source_point(row[0]) for row in rows]
    if len(points) != len(ids):
        raise RuntimeError("A compact-policy chunk ID was absent from the full local source store")
    return points


def _descriptor(policy: str, row_number: int, chunk: Any) -> ChunkDescriptor:
    metadata = chunk.metadata
    identity = build_strategy_aware_point_identity(
        metadata.get("query_id"), metadata.get("passage_index"), metadata.get("chunk_index"), metadata.get("chunk_strategy")
    )
    return ChunkDescriptor(
        point_id=strategy_aware_point_id(chunk), identity=identity, policy=policy,
        query_id=str(metadata.get("query_id")), row_number=row_number,
        passage_index=int(metadata["passage_index"]), chunk_index=int(metadata["chunk_index"]),
        chunk_strategy=str(metadata["chunk_strategy"]),
        text_hash=__import__("hashlib").sha256(chunk.text.encode("utf-8")).hexdigest(), preview=chunk.text[:160],
    )


def _classify_missing(
    descriptor: ChunkDescriptor,
    known_overwrites: set[tuple[str, int, str]],
) -> str:
    if (descriptor.point_id, descriptor.row_number, descriptor.text_hash) in known_overwrites:
        return "KNOWN_FULL_INDEX_OVERWRITE"
    return "UNEXPLAINED"


def _print_missing(descriptor: ChunkDescriptor, classification: str) -> None:
    print(json.dumps({
        "MISSING_POINT_ID": descriptor.point_id, "POLICY": descriptor.policy, "query_id": descriptor.query_id,
        "source_row_number": descriptor.row_number, "passage_index": descriptor.passage_index,
        "chunk_index": descriptor.chunk_index, "chunk_strategy": descriptor.chunk_strategy,
        "text_hash": descriptor.text_hash, "text_preview": descriptor.preview,
        "id_input": descriptor.identity, "classification": classification,
    }, ensure_ascii=False), flush=True)


def _preflight_policy(
    policy: str,
    batch_size: int,
) -> tuple[dict[str, int], set[tuple[str, int, str]]]:
    """Check every policy chunk ID against the immutable source store before writes."""
    tokenizer = get_e5_tokenizer()
    known_overwrites = _known_overwritten_source_chunks()
    source = sqlite3.connect(f"file:{SOURCE_DB.as_posix()}?mode=ro", uri=True)
    counts = Counter()
    excluded: set[tuple[str, int, str]] = set()
    first_missing_printed = False
    pending: list[ChunkDescriptor] = []
    def inspect_pending() -> None:
        nonlocal first_missing_printed
        if not pending:
            return
        keys = [_encoded_local_id(item.point_id) for item in pending]
        placeholders = ",".join("?" for _ in keys)
        found = {row[0] for row in source.execute(f"SELECT id FROM points WHERE id IN ({placeholders})", keys)}
        for item, key in zip(pending, keys, strict=True):
            counts["expected_policy_chunks"] += 1
            classification = _classify_missing(item, known_overwrites)
            if classification == "KNOWN_FULL_INDEX_OVERWRITE":
                # The UUID exists, but its vector/payload represent the later
                # duplicate row. It is not a faithful source for this chunk.
                counts["missing_from_full_source"] += 1
                counts["missing_known_full_index_overwrite"] += 1
                excluded.add((item.point_id, item.row_number, item.text_hash))
                if not first_missing_printed:
                    _print_missing(item, classification)
                    first_missing_printed = True
                continue
            if key in found:
                counts["found_in_full_source"] += 1
                continue
            counts["missing_from_full_source"] += 1
            counts["missing_unexplained"] += 1
            if not first_missing_printed:
                _print_missing(item, "UNEXPLAINED")
                first_missing_printed = True
        pending.clear()
    try:
        for row_number, record in enumerate(iter_msmarco_xi_records("hi", "validation", batch_size=500), start=1):
            for chunk in iter_document_chunks(selected_documents_for_policy(record, policy), max_tokens=256, tokenizer=tokenizer):
                pending.append(_descriptor(policy, row_number, chunk))
                if len(pending) == batch_size:
                    inspect_pending()
        inspect_pending()
    finally:
        source.close()
    for key in (
        "expected_policy_chunks",
        "found_in_full_source",
        "missing_from_full_source",
        "missing_known_full_index_overwrite",
        "missing_unexplained",
    ):
        counts.setdefault(key, 0)
    print(f"PREFLIGHT policy={policy} EXPECTED_POLICY_CHUNKS={counts['expected_policy_chunks']} FOUND_IN_FULL_SOURCE={counts['found_in_full_source']} MISSING_FROM_FULL_SOURCE={counts['missing_from_full_source']} MISSING_KNOWN_COLLISION={counts['missing_known_full_index_overwrite']} MISSING_UNEXPLAINED={counts['missing_unexplained']}", flush=True)
    return dict(counts), excluded


def _build_collection(
    client: QdrantClient,
    policy: str,
    batch_size: int,
    reset: bool,
    excluded_variants: set[tuple[str, int, str]],
) -> int:
    name = COLLECTIONS[policy]
    exists = client.collection_exists(name)
    if exists and reset:
        client.delete_collection(name)
        exists = False
    if not exists:
        client.create_collection(name, vectors_config=VectorParams(size=768, distance=Distance.COSINE))
    tokenizer = get_e5_tokenizer()
    source = sqlite3.connect(f"file:{SOURCE_DB.as_posix()}?mode=ro", uri=True)
    copied = 0
    try:
        pending_ids: list[str] = []
        for row_number, record in enumerate(
            iter_msmarco_xi_records("hi", "validation", batch_size=500), start=1
        ):
            chunks = iter_document_chunks(selected_documents_for_policy(record, policy), max_tokens=256, tokenizer=tokenizer)
            for chunk in chunks:
                descriptor = _descriptor(policy, row_number, chunk)
                if (descriptor.point_id, descriptor.row_number, descriptor.text_hash) not in excluded_variants:
                    pending_ids.append(descriptor.point_id)
            while len(pending_ids) >= batch_size:
                ids, pending_ids = pending_ids[:batch_size], pending_ids[batch_size:]
                # A harmless identical duplicate can occur for query 177416.
                # Qdrant only needs one copy of the stable point ID per upsert.
                unique_ids = list(dict.fromkeys(ids))
                points = _source_points(source, unique_ids)
                client.upsert(name, [PointStruct(id=point.point_id, vector=point.vector, payload=point.payload) for point in points], wait=True)
                copied += len(points)
        if pending_ids:
            points = _source_points(source, list(dict.fromkeys(pending_ids)))
            client.upsert(name, [PointStruct(id=point.point_id, vector=point.vector, payload=point.payload) for point in points], wait=True)
            copied += len(points)
    finally:
        source.close()
    return copied


def _expected_provenance(record: dict[str, object]) -> set[tuple[str, str]]:
    tokenizer = get_e5_tokenizer()
    selected = [document for document in selected_documents_for_policy(record, POLICY_A)]
    return {
        (str(chunk.metadata["query_id"]), str(chunk.metadata["passage_index"]))
        for chunk in iter_document_chunks(selected, max_tokens=256, tokenizer=tokenizer)
    }


def _evaluate(client: QdrantClient, collection: str, records: dict[str, dict[str, object]], validation: dict[str, dict[str, object]], top_k: int, repeats: int) -> dict[str, object]:
    embedder = E5Embedder()
    metrics = Counter()
    latencies: list[float] = []
    selected_count = zero_count = 0
    for query_id, record in records.items():
        query = str(record.get("query", "")).strip()
        if not query:
            continue
        expected = _expected_provenance(record)
        has_selected = bool(expected)
        selected_count += int(has_selected)
        zero_count += int(not has_selected)
        vector = embedder.embed_query(query).tolist()  # Query embeddings are intentionally allowed.
        for repeat in range(repeats):
            started = time.perf_counter()
            response = client.query_points(collection, query=vector, limit=top_k, with_payload=True)
            latency = (time.perf_counter() - started) * 1000
            if repeat:
                latencies.append(latency)
            points = response.points
            provenances = [(str((point.payload or {}).get("query_id")), str((point.payload or {}).get("passage_index"))) for point in points]
            query_ids = [item[0] for item in provenances]
            if has_selected:
                metrics["top1_expected_evidence"] += int(bool(provenances) and provenances[0] in expected)
                metrics["expected_evidence_top_k"] += int(any(item in expected for item in provenances))
                metrics["top1_expected_query"] += int(bool(query_ids) and query_ids[0] == query_id)
                metrics["expected_query_top_k"] += int(query_id in query_ids)
                metrics["correct_evidence_family"] += int(any(item[0] == query_id for item in provenances))
            else:
                metrics["zero_represented"] += int(bool(points))
                metrics["zero_top1_same_query"] += int(bool(query_ids) and query_ids[0] == query_id)
                metrics["zero_query_top_k"] += int(query_id in query_ids)
    selected_denominator = max(1, selected_count * repeats)
    zero_denominator = max(1, zero_count * repeats)
    return {
        "selected_query_count": selected_count,
        "zero_selected_query_count": zero_count,
        "top1_expected_evidence_rate": metrics["top1_expected_evidence"] / selected_denominator,
        "expected_evidence_in_top_k_rate": metrics["expected_evidence_top_k"] / selected_denominator,
        "top1_expected_query_id_rate": metrics["top1_expected_query"] / selected_denominator,
        "expected_query_id_in_top_k_rate": metrics["expected_query_top_k"] / selected_denominator,
        "correct_evidence_family_rate": metrics["correct_evidence_family"] / selected_denominator,
        "zero_selected": {
            "query_representation_rate": metrics["zero_represented"] / zero_denominator,
            "top1_same_query_id_rate": metrics["zero_top1_same_query"] / zero_denominator,
            "same_query_id_in_top_k_rate": metrics["zero_query_top_k"] / zero_denominator,
        },
        "latency_ms": {"p50": _percentile(latencies, 50), "p70": _percentile(latencies, 70), "p95": _percentile(latencies, 95), "p100": max(latencies) if latencies else 0.0},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--queries", type=int, default=150)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--reset-benchmarks", action="store_true")
    parser.add_argument("--cleanup", action="store_true")
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Audit source mappability without opening or writing a temporary Qdrant collection.",
    )
    parser.add_argument("--report-path", type=Path, default=Path("tmp/compact_hindi_policy_benchmark.json"))
    args = parser.parse_args()
    if min(args.batch_size, args.queries, args.top_k, args.repeats) < 1:
        parser.error("batch-size, queries, top-k, and repeats must be positive")
    print("DOCUMENT_EMBEDDINGS_CREATED=0")
    print("FULL_COLLECTION_WRITES=0")
    print("NEON_WRITES=0")
    print("QDRANT_CLOUD_WRITES=0")
    print("ENGLISH_DATA_LOADED=0")
    if args.cleanup:
        client = QdrantClient(path=str(BENCHMARK_PATH))
        try:
            for name in COLLECTIONS.values():
                if client.collection_exists(name):
                    client.delete_collection(name)
            print("TEMPORARY_BENCHMARK_COLLECTIONS_CLEANED=true")
        finally:
            client.close()
        return
    preflight: dict[str, dict[str, int]] = {}
    exclusions: dict[str, set[tuple[str, int, str]]] = {}
    for policy in POLICIES:
        preflight[policy], exclusions[policy] = _preflight_policy(policy, args.batch_size)
    if any(details["missing_unexplained"] for details in preflight.values()):
        raise RuntimeError(
            "Unexplained policy chunk IDs are missing from the full source; "
            "no benchmark collection was created"
        )
    if args.preflight_only:
        print("POLICY_MAPPABILITY_PREFLIGHT_OK=true")
        return
    client = QdrantClient(path=str(BENCHMARK_PATH))
    try:
        for policy, name in COLLECTIONS.items():
            print(f"TEMP_COLLECTION_COUNT policy={policy} collection={name} count={client.count(name, exact=True).count if client.collection_exists(name) else 0}")
        validation = _load_validation(args.queries)
        records = _records_for_validation(set(validation))
        if len(records) < 100:
            raise RuntimeError("Validation artifact did not resolve at least 100 source queries")
        results = {}
        for policy in POLICIES:
            copied = _build_collection(client, policy, args.batch_size, args.reset_benchmarks, exclusions[policy])
            metrics = _evaluate(client, COLLECTIONS[policy], records, validation, args.top_k, args.repeats)
            metrics["temporary_points_copied"] = copied
            metrics["chunk_count"] = CHUNK_COUNTS[policy]
            metrics["quality_per_additional_10k_chunks"] = None
            results[policy] = metrics
        for policy, previous in ((POLICY_D, POLICY_A), (POLICY_E, POLICY_D)):
            chunk_delta = (CHUNK_COUNTS[policy] - CHUNK_COUNTS[previous]) / 10_000
            results[policy]["quality_per_additional_10k_chunks"] = (
                (results[policy]["expected_evidence_in_top_k_rate"] - results[previous]["expected_evidence_in_top_k_rate"])
                / chunk_delta
            )
        recommendation = recommend_compact_policy(results)
        report = {
            "validation_query_count": len(records),
            "preflight": preflight,
            "policies": results,
            "recommendation": recommendation,
            "recommendation_interpretation": recommendation_interpretation(results, recommendation),
            "safety": {"document_embeddings_created": 0, "full_collection_writes": 0, "neon_writes": 0, "qdrant_cloud_writes": 0, "english_data_loaded": 0},
        }
        args.report_path.parent.mkdir(parents=True, exist_ok=True)
        args.report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False))
    finally:
        client.close()


if __name__ == "__main__":
    main()
