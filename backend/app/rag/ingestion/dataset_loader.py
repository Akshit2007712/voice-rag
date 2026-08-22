"""Lazy local-Parquet loader for MSMARCO-XI offline ingestion."""

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


DATASET_NAME = "ai4bharat/MSMARCO-XI"
DEFAULT_LANGUAGE = "hi"
DEFAULT_SPLIT = "validation"


LANGUAGE_FILES = {
    # MSMARCO-XI stores original English query/passage fields alongside every
    # translated record. English is therefore a local view of this validation
    # file, not a second independently downloaded corpus.
    "en": {
        "validation": "validation/hinval.parquet",
    },
    "hi": {
        "train": "train/hintrain.parquet",
        "validation": "validation/hinval.parquet",
    },
}

PROJECT_ROOT = Path(__file__).resolve().parents[3]
LOCAL_DATA_ROOT = PROJECT_ROOT / "data" / "raw"


def get_msmarco_xi_file_path(
    language: str = DEFAULT_LANGUAGE,
    split: str = DEFAULT_SPLIT,
) -> str:
    """Return the configured Parquet path after validating language and split."""
    normalized_language = language.lower()
    if normalized_language not in LANGUAGE_FILES:
        raise ValueError(f"Unsupported language: {language}")
    if split not in LANGUAGE_FILES[normalized_language]:
        raise ValueError(f"Unsupported split: {split}")

    return LANGUAGE_FILES[normalized_language][split]


def get_msmarco_xi_local_file_path(
    language: str = DEFAULT_LANGUAGE,
    split: str = DEFAULT_SPLIT,
) -> Path:
    """Resolve and validate a local MSMARCO-XI Parquet file path."""
    file_path = get_msmarco_xi_file_path(language=language, split=split)
    local_path = LOCAL_DATA_ROOT / split / Path(file_path).name
    if not local_path.is_file():
        raise FileNotFoundError(f"Local MSMARCO-XI Parquet file not found: {local_path}")

    return local_path


def iter_msmarco_xi_records(
    language: str = DEFAULT_LANGUAGE,
    split: str = DEFAULT_SPLIT,
    batch_size: int = 500,
) -> Iterator[dict[str, Any]]:
    """Yield raw MSMARCO-XI records incrementally from local Parquet batches."""
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")

    local_path = get_msmarco_xi_local_file_path(language=language, split=split)
    parquet_file = pq.ParquetFile(local_path)
    try:
        for batch in parquet_file.iter_batches(batch_size=batch_size):
            yield from batch.to_pylist()
    finally:
        parquet_file.close()
