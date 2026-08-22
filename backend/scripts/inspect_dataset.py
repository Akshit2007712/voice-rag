"""Inspect a small MSMARCO-XI Parquet sample directly with PyArrow."""

import argparse
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Sequence

import pyarrow.parquet as pq

# Allow `python scripts/inspect_dataset.py` from the backend directory.
BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.rag.ingestion.dataset_loader import (  # noqa: E402
    DATASET_NAME,
    DEFAULT_LANGUAGE,
    DEFAULT_SPLIT,
    get_msmarco_xi_file_path,
)


def _keys(value: Any) -> list[str]:
    """Return sorted keys when the value is a mapping."""
    return sorted(value) if isinstance(value, Mapping) else []


INSPECTION_COLUMNS = (
    "query",
    "query_id",
    "query_type",
    "passages",
    "meta",
    "source_lang",
    "target_lang",
)


def _sample_records(
    parquet_file: pq.ParquetFile,
    sample_size: int,
    columns: Sequence[str],
) -> list[dict[str, Any]]:
    """Read selected columns from the first row group and return a small slice."""
    first_row_group = parquet_file.read_row_group(0, columns=list(columns))
    return first_row_group.slice(0, sample_size).to_pylist()


def build_report(
    records: list[dict[str, Any]],
    schema: Any,
    total_row_count: int | None,
    row_group_count: int,
    dataset_path: str,
    language: str,
    split: str,
) -> dict[str, Any]:
    """Build a lightweight schema and data-quality report from a sample."""
    first_record = records[0] if records else {}
    passages = first_record.get("passages", {})

    passage_counts = [
        len(record.get("passages", {}).get("Translated_passages", []))
        for record in records
        if isinstance(record.get("passages"), Mapping)
    ]

    return {
        "dataset": DATASET_NAME,
        "dataset_path": dataset_path,
        "configuration_language": language,
        "split": split,
        "total_row_count": total_row_count,
        "row_group_count": row_group_count,
        "sample_records_loaded": len(records),
        "schema": str(schema),
        "record_fields": list(schema.names),
        "sample_record_fields": _keys(first_record),
        "passage_fields": _keys(passages),
        "metadata_fields": _keys(first_record.get("meta")),
        "source_languages_in_sample": sorted(
            {
                str(record.get("source_lang"))
                for record in records
                if record.get("source_lang")
            }
        ),
        "target_languages_in_sample": sorted(
            {
                str(record.get("target_lang"))
                for record in records
                if record.get("target_lang")
            }
        ),
        "translated_passage_counts_in_sample": passage_counts,
        "sample_records": records,
    }


def main() -> None:
    """Inspect a remote Parquet file without consuming the production stream."""
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--language",
        default=DEFAULT_LANGUAGE,
        help="MSMARCO-XI language configuration",
    )

    parser.add_argument(
        "--split",
        default=DEFAULT_SPLIT,
        help="Dataset split to inspect",
    )

    parser.add_argument(
        "--sample-size",
        type=int,
        default=5,
        help="Number of records to inspect",
    )

    args = parser.parse_args()
    if args.sample_size < 1:
        parser.error("--sample-size must be at least 1")

    file_path = get_msmarco_xi_file_path(args.language, args.split)
    dataset_path = f"hf://datasets/{DATASET_NAME}/{file_path}"

    print(f"[inspect] Opening remote Parquet file: {dataset_path}", flush=True)
    parquet_file = pq.ParquetFile(dataset_path)
    try:
        print("[inspect] Reading Parquet metadata and schema...", flush=True)
        metadata = parquet_file.metadata
        available_columns = parquet_file.schema_arrow.names
        inspection_columns = [
            column for column in INSPECTION_COLUMNS if column in available_columns
        ]
        print(
            "[inspect] Reading first row group with columns: "
            f"{', '.join(inspection_columns)}",
            flush=True,
        )
        report = build_report(
            records=_sample_records(parquet_file, args.sample_size, inspection_columns),
            schema=parquet_file.schema_arrow,
            total_row_count=metadata.num_rows if metadata is not None else None,
            row_group_count=metadata.num_row_groups if metadata is not None else 0,
            dataset_path=dataset_path,
            language=args.language,
            split=args.split,
        )
    finally:
        parquet_file.close()

    print(json.dumps(
        report,
        ensure_ascii=False,
        indent=2,
        default=str,
    ))


if __name__ == "__main__":
    main()
