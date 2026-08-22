"""Shared policy-A and storage-planning helpers for the bilingual compact corpus."""

from __future__ import annotations

import uuid
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from app.rag.ingestion.preprocessor import RetrievalDocument, preprocess_msmarco_xi_english_record


FROZEN_HINDI_POLICY_A_CHUNKS = 58_427
QDRANT_VECTOR_DIMENSION = 768
QDRANT_TARGET_GIB = 4.0
# 3 KiB raw vector plus a deliberately conservative 13 KiB allowance for
# payload, index, and operational overhead. This is a planning estimate, not
# deployed-server measurement.
CONSERVATIVE_QDRANT_BYTES_PER_POINT = 16 * 1024


@dataclass
class EnglishDatasetStatistics:
    """Bounded counters for the original-English view of MSMARCO-XI."""

    rows: int = 0
    passages: int = 0
    non_empty_passages: int = 0
    selected_passages: int = 0
    zero_selected_rows: int = 0
    compact_chunks: int = 0


def is_selected(value: object) -> bool:
    """Keep the frozen MSMARCO selected-label interpretation."""
    return value is True or value == 1


def english_policy_a_documents(record: Mapping[str, Any]) -> list[RetrievalDocument]:
    """Apply frozen Policy A to original English passages only."""
    return [
        document
        for document in preprocess_msmarco_xi_english_record(record)
        if is_selected(document.metadata["is_selected"])
    ]


def build_bilingual_point_identity(
    language: str,
    query_id: object,
    passage_index: object,
    chunk_index: object,
    chunk_strategy: object,
) -> str:
    """Build a language-aware deterministic identity for the new collection."""
    normalized_language = language.strip().lower()
    if normalized_language not in {"hi", "en"}:
        raise ValueError(f"Unsupported bilingual language: {language}")
    return (
        f"language={normalized_language}|query_id={query_id}|"
        f"passage_index={passage_index}|chunk_index={chunk_index}|"
        f"chunk_strategy={chunk_strategy}"
    )


def build_bilingual_point_id(language: str, metadata: Mapping[str, object]) -> str:
    """Create a UUIDv5 that cannot collide across Hindi and English."""
    identity = build_bilingual_point_identity(
        language,
        metadata.get("query_id"),
        metadata.get("passage_index"),
        metadata.get("chunk_index"),
        metadata.get("chunk_strategy"),
    )
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"msmarco-xi-bilingual-compact:{identity}"))


def bilingual_payload(language: str, text: str, metadata: Mapping[str, object]) -> dict[str, object]:
    """Return the complete language-routable Qdrant payload contract."""
    normalized_language = language.strip().lower()
    target_lang = "hin_Deva" if normalized_language == "hi" else "eng_Latn"
    required = ("query_id", "passage_index", "chunk_index", "chunk_strategy", "is_selected", "token_count")
    missing = [key for key in required if key not in metadata]
    if missing:
        raise ValueError(f"Chunk metadata is missing required bilingual payload fields: {missing}")
    return {
        "language": normalized_language,
        "target_lang": target_lang,
        "query_id": metadata["query_id"],
        "passage_index": metadata["passage_index"],
        "chunk_index": metadata["chunk_index"],
        "chunk_strategy": metadata["chunk_strategy"],
        "is_selected": metadata["is_selected"],
        "token_count": metadata["token_count"],
        "text": text,
    }


def estimated_bilingual_storage_gib(
    english_compact_chunks: int,
    *,
    bytes_per_point: int = CONSERVATIVE_QDRANT_BYTES_PER_POINT,
) -> float:
    """Estimate final Qdrant storage from frozen Hindi plus English Policy-A chunks."""
    if english_compact_chunks < 0 or bytes_per_point < QDRANT_VECTOR_DIMENSION * 4:
        raise ValueError("Invalid bilingual storage estimate input")
    return (FROZEN_HINDI_POLICY_A_CHUNKS + english_compact_chunks) * bytes_per_point / (1024**3)


def estimate_is_safe(english_compact_chunks: int, *, target_gib: float = QDRANT_TARGET_GIB) -> bool:
    """Return whether the conservative storage estimate fits the configured target."""
    if target_gib <= 0:
        raise ValueError("target_gib must be positive")
    return estimated_bilingual_storage_gib(english_compact_chunks) <= target_gib


def count_selected_labels(labels: Iterable[object]) -> int:
    """Count positive MSMARCO labels without altering their source values."""
    return sum(is_selected(label) for label in labels)
