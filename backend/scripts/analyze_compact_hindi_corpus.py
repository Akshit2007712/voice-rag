"""Read-only compact-corpus analysis for Hindi MSMARCO-XI deployment planning."""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import pyarrow.parquet as pq

# Allow direct execution via `python scripts/analyze_compact_hindi_corpus.py`
# from the backend project root, matching the existing scripts convention.
BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.rag.analysis.compact_hindi_policy import (
    POLICY_A,
    POLICY_D,
    POLICY_E,
    POLICIES,
    is_selected,
    selected_documents_for_policy,
    storage_estimate,
)
from app.rag.ingestion.chunker import get_e5_tokenizer, iter_document_chunks
from app.rag.ingestion.dataset_loader import get_msmarco_xi_local_file_path
from app.rag.ingestion.preprocessor import preprocess_msmarco_xi_record


FULL_PASSAGE_BASELINE = 977_545
FULL_CHUNK_BASELINE = 1_000_361
NEON_BYTES_PER_POINT = 9_193.472
VALIDATION_POLICIES = (POLICY_A, POLICY_D, POLICY_E)


@dataclass
class PolicyStats:
    rows_covered: int = 0
    passages_included: int = 0
    selected_passages_included: int = 0
    non_selected_passages_included: int = 0
    chunks_generated: int = 0
    max_chunks_for_single_row: int = 0
    strategy_counts: Counter[str] = field(default_factory=Counter)
    query_ids: set[str] = field(default_factory=set)


def _validation_candidate(record: dict[str, object], policy: str, selected_indexes: list[int]) -> tuple[int, dict[str, object]] | None:
    query_id = record.get("query_id")
    query = record.get("query")
    if query_id is None or not isinstance(query, str) or not query.strip():
        return None
    payload = {
        "policy_name": policy,
        "query_id": str(query_id),
        "query": query.strip(),
        "expected_selected_passage_indexes": selected_indexes,
        "had_selected_evidence": bool(selected_indexes),
    }
    priority = int.from_bytes(hashlib.blake2b(f"{policy}:{query_id}".encode(), digest_size=8).digest(), "big")
    return priority, payload


def _consider_validation(heap: list[tuple[int, str, dict[str, object]]], candidate: tuple[int, dict[str, object]] | None) -> None:
    if candidate is None:
        return
    priority, payload = candidate
    entry = (-priority, payload["query_id"], repr(payload), payload)
    if len(heap) < 100:
        heapq.heappush(heap, entry)
    elif entry > heap[0]:
        heapq.heapreplace(heap, entry)


def analyze(
    parquet_path: Path,
    *,
    batch_size: int,
    storage_price_per_gb: float | None,
) -> tuple[dict[str, object], dict[str, list[dict[str, object]]]]:
    """Stream local Parquet records and reuse the established E5-token chunker."""
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    parquet_file = pq.ParquetFile(parquet_path)
    schema = str(parquet_file.schema_arrow)
    tokenizer = get_e5_tokenizer()
    source = Counter()
    selected_per_row = Counter()
    query_ids: set[str] = set()
    policy_stats = {policy: PolicyStats() for policy in POLICIES}
    validation_heaps = {policy: [] for policy in VALIDATION_POLICIES}
    started = time.perf_counter()
    try:
        for batch in parquet_file.iter_batches(batch_size=batch_size):
            for record in batch.to_pylist():
                source["total_rows"] += 1
                if record.get("query_id") is not None:
                    query_ids.add(str(record["query_id"]))
                passages = record.get("passages")
                translated = passages.get("Translated_passages", []) if isinstance(passages, dict) else []
                labels = passages.get("is_selected", []) if isinstance(passages, dict) else []
                if not isinstance(translated, list) or not isinstance(labels, list) or len(translated) != len(labels):
                    raise ValueError("Malformed MSMARCO-XI passages/is_selected parallel lists")
                source["total_translated_passages"] += len(translated)
                source["non_empty_translated_passages"] += sum(isinstance(value, str) and bool(value.split()) for value in translated)
                documents = preprocess_msmarco_xi_record(record)
                selected_indexes = [int(document.metadata["passage_index"]) for document in documents if is_selected(document.metadata["is_selected"])]
                source["selected_passages"] += len(selected_indexes)
                source["non_selected_passages"] += len(documents) - len(selected_indexes)
                selected_per_row[len(selected_indexes)] += 1
                source["rows_with_at_least_one_selected_passage"] += int(bool(selected_indexes))
                source["rows_with_zero_selected_passages"] += int(not selected_indexes)
                for policy, stats in policy_stats.items():
                    chosen = selected_documents_for_policy(record, policy)
                    if chosen:
                        stats.rows_covered += 1
                        stats.query_ids.add(str(record.get("query_id")))
                    stats.passages_included += len(chosen)
                    stats.selected_passages_included += sum(is_selected(document.metadata["is_selected"]) for document in chosen)
                    stats.non_selected_passages_included += sum(not is_selected(document.metadata["is_selected"]) for document in chosen)
                    row_chunks = list(iter_document_chunks(chosen, max_tokens=256, tokenizer=tokenizer))
                    stats.chunks_generated += len(row_chunks)
                    stats.max_chunks_for_single_row = max(stats.max_chunks_for_single_row, len(row_chunks))
                    stats.strategy_counts.update(str(chunk.metadata["chunk_strategy"]) for chunk in row_chunks)
                    if policy in validation_heaps and chosen:
                        _consider_validation(validation_heaps[policy], _validation_candidate(record, policy, selected_indexes))
            elapsed = time.perf_counter() - started
            print(
                f"ROWS_SCANNED={source['total_rows']} PASSAGES_SCANNED={source['total_translated_passages']} ELAPSED_TIME_S={elapsed:.1f}",
                flush=True,
            )
    finally:
        parquet_file.close()
    source["total_query_ids"] = len(query_ids)
    source["duplicate_query_ids"] = source["total_rows"] - len(query_ids)
    policies: dict[str, object] = {}
    for policy, stats in policy_stats.items():
        policies[policy] = {
            "rows_covered": stats.rows_covered,
            "rows_dropped": source["total_rows"] - stats.rows_covered,
            "passages_included": stats.passages_included,
            "selected_passages_included": stats.selected_passages_included,
            "non_selected_passages_included": stats.non_selected_passages_included,
            "chunks_generated": stats.chunks_generated,
            "chunk_strategy_counts": dict(stats.strategy_counts),
            "average_passages_per_row": stats.passages_included / source["total_rows"],
            "average_chunks_per_row": stats.chunks_generated / source["total_rows"],
            "max_chunks_for_single_row": stats.max_chunks_for_single_row,
            "query_ids_retained": len(stats.query_ids),
            "query_ids_dropped": len(query_ids.difference(stats.query_ids)),
            "all_selected_evidence_retained": stats.selected_passages_included == source["selected_passages"],
            "passage_reduction_percent": 100 * (1 - stats.passages_included / FULL_PASSAGE_BASELINE),
            "chunk_reduction_percent": 100 * (1 - stats.chunks_generated / FULL_CHUNK_BASELINE),
            "zero_selected_rows_represented_percent": (
                100 * (stats.rows_covered - source["rows_with_at_least_one_selected_passage"]) / source["rows_with_zero_selected_passages"]
                if source["rows_with_zero_selected_passages"] else 0.0
            ),
            "storage_extrapolation": storage_estimate(stats.chunks_generated, NEON_BYTES_PER_POINT, storage_price_per_gb),
        }
    validations = {
        policy: [entry[3] for entry in sorted(heap, key=lambda item: (item[1], item[0]))]
        for policy, heap in validation_heaps.items()
    }
    report = {
        "schema": schema,
        "source_statistics": {**source, "selected_passages_per_row_distribution": dict(selected_per_row)},
        "full_hindi_baseline": {"passages": FULL_PASSAGE_BASELINE, "chunks": FULL_CHUNK_BASELINE},
        "measured_neon_bytes_per_point": NEON_BYTES_PER_POINT,
        "policies": policies,
        "recommendation": "SELECTED_PLUS_1_WITH_FALLBACK_RECOMMENDED",
        "safety": {"document_embeddings_created": 0, "qdrant_writes": 0, "neon_writes": 0},
    }
    return report, validations


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--storage-price-per-gb", type=float)
    parser.add_argument("--report-path", type=Path, default=Path("tmp/compact_hindi_corpus_analysis.json"))
    parser.add_argument("--validation-path", type=Path, default=Path("data/validation/compact_hindi_policy_validation.json"))
    args = parser.parse_args()
    if args.storage_price_per_gb is not None and args.storage_price_per_gb < 0:
        parser.error("--storage-price-per-gb must not be negative")
    parquet_path = get_msmarco_xi_local_file_path("hi", "validation")
    report, validations = analyze(parquet_path, batch_size=args.batch_size, storage_price_per_gb=args.storage_price_per_gb)
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.validation_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    args.validation_path.write_text(json.dumps(validations, ensure_ascii=False, indent=2), encoding="utf-8")
    print("DOCUMENT_EMBEDDINGS_CREATED=0")
    print("QDRANT_WRITES=0")
    print("NEON_WRITES=0")
    print(f"REPORT_PATH={args.report_path}")
    print(f"VALIDATION_PATH={args.validation_path}")
    print(f"RECOMMENDATION={report['recommendation']}")
    for policy, details in report["policies"].items():
        print(f"{policy} chunks={details['chunks_generated']} estimated_neon_gb={details['storage_extrapolation']['estimated_neon_gb']:.2f}")


if __name__ == "__main__":
    main()
