"""Convert raw MSMARCO-XI records into retrieval documents."""

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class RetrievalDocument:
    """A passage ready for a future chunking stage."""

    text: str
    metadata: dict[str, Any]


def preprocess_msmarco_xi_record(record: Mapping[str, Any]) -> list[RetrievalDocument]:
    """Create one document for each non-empty translated passage in a record.

    A missing passage block produces no documents. Malformed passage data or
    mismatched parallel passage/label lists raises a ``ValueError`` to prevent
    relevance labels from being paired with the wrong text.
    """
    if not isinstance(record, Mapping):
        raise TypeError("MSMARCO-XI record must be a mapping")

    passages = record.get("passages")
    if passages is None:
        return []
    if not isinstance(passages, Mapping):
        raise ValueError("MSMARCO-XI passages must be a mapping")

    translated_passages = passages.get("Translated_passages")
    if translated_passages is None:
        return []
    if not isinstance(translated_passages, list):
        raise ValueError("passages.Translated_passages must be a list")

    is_selected = passages.get("is_selected")
    if not isinstance(is_selected, list):
        raise ValueError("passages.is_selected must be a list when passages exist")
    if len(translated_passages) != len(is_selected):
        raise ValueError(
            "passages.Translated_passages and passages.is_selected must have the same length"
        )

    documents: list[RetrievalDocument] = []
    for passage_index, (passage, relevance_label) in enumerate(
        zip(translated_passages, is_selected, strict=True)
    ):
        if not isinstance(passage, str):
            raise ValueError(
                f"passages.Translated_passages[{passage_index}] must be a string"
            )

        cleaned_passage = " ".join(passage.split())
        if not cleaned_passage:
            continue

        documents.append(
            RetrievalDocument(
                text=cleaned_passage,
                metadata={
                    "query_id": record.get("query_id"),
                    "query_type": record.get("query_type"),
                    "source_lang": record.get("source_lang"),
                    "target_lang": record.get("target_lang"),
                    "passage_index": passage_index,
                    "is_selected": relevance_label,
                },
            )
        )

    return documents


def preprocess_msmarco_xi_english_record(record: Mapping[str, Any]) -> list[RetrievalDocument]:
    """Create retrieval documents from MSMARCO-XI's original English fields.

    The source Parquet pairs ``English_passages`` with the same ``is_selected``
    labels used by the translated passage list. This preserves their positional
    relevance relationship while keeping English documents distinct from Hindi.
    """
    if not isinstance(record, Mapping):
        raise TypeError("MSMARCO-XI record must be a mapping")
    passages = record.get("passages")
    if passages is None:
        return []
    if not isinstance(passages, Mapping):
        raise ValueError("MSMARCO-XI passages must be a mapping")
    english_passages = passages.get("English_passages")
    if english_passages is None:
        return []
    if not isinstance(english_passages, list):
        raise ValueError("passages.English_passages must be a list")
    is_selected = passages.get("is_selected")
    if not isinstance(is_selected, list):
        raise ValueError("passages.is_selected must be a list when passages exist")
    if len(english_passages) != len(is_selected):
        raise ValueError(
            "passages.English_passages and passages.is_selected must have the same length"
        )

    documents: list[RetrievalDocument] = []
    for passage_index, (passage, relevance_label) in enumerate(
        zip(english_passages, is_selected, strict=True)
    ):
        if not isinstance(passage, str):
            raise ValueError(f"passages.English_passages[{passage_index}] must be a string")
        cleaned_passage = " ".join(passage.split())
        if not cleaned_passage:
            continue
        documents.append(
            RetrievalDocument(
                text=cleaned_passage,
                metadata={
                    "query_id": record.get("query_id"),
                    "query_type": record.get("query_type"),
                    "source_lang": "eng_Latn",
                    "target_lang": "eng_Latn",
                    "passage_index": passage_index,
                    "is_selected": relevance_label,
                },
            )
        )
    return documents
