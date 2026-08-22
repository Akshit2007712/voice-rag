"""Analyze translated-passage word counts in local MSMARCO-XI Parquet data."""

import argparse
import math
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

# Allow `python scripts/analyze_passage_lengths.py` from the backend directory.
BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.rag.ingestion.dataset_loader import get_msmarco_xi_local_file_path  # noqa: E402


BUCKETS = (
    (0, 50, "0-50"),
    (51, 100, "51-100"),
    (101, 200, "101-200"),
    (201, 300, "201-300"),
    (301, 500, "301-500"),
    (501, 750, "501-750"),
    (751, 1000, "751-1000"),
)


def _percentile_nearest_rank(sorted_lengths: list[int], percentile: float) -> int | None:
    """Return an exact nearest-rank percentile from already-sorted word counts."""
    if not sorted_lengths:
        return None
    rank = max(1, math.ceil(percentile * len(sorted_lengths)))
    return sorted_lengths[rank - 1]


def _bucket_label(word_count: int) -> str:
    """Return the display bucket for a non-empty passage word count."""
    for lower, upper, label in BUCKETS:
        if lower <= word_count <= upper:
            return label
    return "1000+"


def analyze_passage_lengths(local_path: Path, batch_size: int) -> dict[str, Any]:
    """Compute exact word-count statistics while reading only Parquet's passages column."""
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")

    histogram = {label: 0 for _, _, label in BUCKETS}
    histogram["1000+"] = 0
    lengths: list[int] = []
    total_records = 0
    total_translated_passages = 0
    empty_passages = 0
    malformed_records = 0
    non_string_passages = 0
    next_progress_count = 10_000

    print(f"[analysis] Opening local Parquet: {local_path}", flush=True)
    parquet_file = pq.ParquetFile(local_path)
    try:
        print(
            f"[analysis] Reading passages in batches of {batch_size} records...",
            flush=True,
        )
        for batch in parquet_file.iter_batches(batch_size=batch_size, columns=["passages"]):
            for record in batch.to_pylist():
                total_records += 1
                passages = record.get("passages")
                if not isinstance(passages, Mapping):
                    malformed_records += 1
                    continue

                translated_passages = passages.get("Translated_passages")
                if not isinstance(translated_passages, list):
                    malformed_records += 1
                    continue

                total_translated_passages += len(translated_passages)
                for passage in translated_passages:
                    if not isinstance(passage, str):
                        non_string_passages += 1
                        continue

                    word_count = len(passage.split())
                    if word_count == 0:
                        empty_passages += 1
                        continue

                    lengths.append(word_count)
                    histogram[_bucket_label(word_count)] += 1

            if total_records >= next_progress_count:
                print(f"[analysis] Processed {total_records:,} records...", flush=True)
                next_progress_count += 10_000
    finally:
        parquet_file.close()

    lengths.sort()
    return {
        "total_records_processed": total_records,
        "total_translated_passages": total_translated_passages,
        "non_empty_translated_passages": len(lengths),
        "empty_passages": empty_passages,
        "non_string_passages": non_string_passages,
        "malformed_records": malformed_records,
        "minimum_word_count": lengths[0] if lengths else None,
        "maximum_word_count": lengths[-1] if lengths else None,
        "mean_word_count": (sum(lengths) / len(lengths)) if lengths else None,
        "p50_word_count": _percentile_nearest_rank(lengths, 0.50),
        "p75_word_count": _percentile_nearest_rank(lengths, 0.75),
        "p90_word_count": _percentile_nearest_rank(lengths, 0.90),
        "p95_word_count": _percentile_nearest_rank(lengths, 0.95),
        "p99_word_count": _percentile_nearest_rank(lengths, 0.99),
        "histogram": histogram,
    }


def main() -> None:
    """Run the local Hindi validation-passage length analysis."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=500)
    args = parser.parse_args()

    try:
        local_path = get_msmarco_xi_local_file_path(language="hi", split="validation")
        report = analyze_passage_lengths(local_path, args.batch_size)
    except ValueError as exc:
        parser.error(str(exc))

    print("\nPASSAGE WORD-COUNT ANALYSIS")
    for key, value in report.items():
        if key == "histogram":
            continue
        print(f"{key.upper()}: {value}")

    print("\nHISTOGRAM (NON-EMPTY TRANSLATED PASSAGES)")
    for label, count in report["histogram"].items():
        print(f"{label}: {count}")


if __name__ == "__main__":
    main()
