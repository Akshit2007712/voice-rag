"""Inspect and size-gate the original-English MSMARCO-XI Policy-A compact corpus."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.rag.analysis.bilingual_compact import (  # noqa: E402
    EnglishDatasetStatistics,
    count_selected_labels,
    english_policy_a_documents,
    estimate_is_safe,
    estimated_bilingual_storage_gib,
)
from app.rag.ingestion.chunker import get_e5_tokenizer, iter_document_chunks  # noqa: E402
from app.rag.ingestion.dataset_loader import iter_msmarco_xi_records  # noqa: E402


def analyze(batch_size: int) -> dict[str, object]:
    """Stream the local English view and count frozen Policy-A chunks exactly."""
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    stats = EnglishDatasetStatistics()
    tokenizer = get_e5_tokenizer()
    for record in iter_msmarco_xi_records("en", "validation", batch_size=batch_size):
        stats.rows += 1
        passages = record.get("passages")
        english = passages.get("English_passages", []) if isinstance(passages, dict) else []
        labels = passages.get("is_selected", []) if isinstance(passages, dict) else []
        if not isinstance(english, list) or not isinstance(labels, list):
            raise ValueError("English passages and is_selected must be lists")
        stats.passages += len(english)
        stats.non_empty_passages += sum(isinstance(value, str) and bool(value.strip()) for value in english)
        selected_count = count_selected_labels(labels)
        stats.selected_passages += selected_count
        stats.zero_selected_rows += int(selected_count == 0)
        stats.compact_chunks += sum(
            1
            for _ in iter_document_chunks(
                english_policy_a_documents(record), max_tokens=256, tokenizer=tokenizer
            )
        )
    estimated_gib = estimated_bilingual_storage_gib(stats.compact_chunks)
    return {
        "language": "en",
        "source": "MSMARCO-XI original English fields in validation/hinval.parquet",
        "policy": "POLICY_A_SELECTED_ONLY",
        "max_tokens": 256,
        "english_dataset_statistics": asdict(stats),
        "frozen_hindi_compact_chunks": 58_427,
        "estimated_final_bilingual_points": 58_427 + stats.compact_chunks,
        "conservative_estimated_bilingual_qdrant_gib": estimated_gib,
        "qdrant_target_gib": 4.0,
        "estimated_final_collection_safe": estimate_is_safe(stats.compact_chunks),
        "document_embeddings_created": 0,
        "qdrant_writes": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument(
        "--report-path",
        type=Path,
        default=BACKEND_ROOT / "benchmarks" / "compact_english_policy_a_analysis.json",
    )
    args = parser.parse_args()
    report = analyze(args.batch_size)
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
