"""Build bounded, provenance-aware context from ranked retrieval results."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from math import ceil
from typing import Any

from app.rag.retrieval.retriever import RetrievedChunk


TokenEstimator = Callable[[str], int]
_PROVENANCE_KEYS = ("query_id", "passage_index", "chunk_index")


@dataclass(frozen=True)
class ContextBundle:
    """LLM-ready context together with its selected evidence provenance."""

    text: str
    evidence_count: int
    estimated_token_count: int
    provenance: list[dict[str, Any]]


def estimate_context_tokens(text: str) -> int:
    """Estimate tokens cheaply from character count until an LLM tokenizer is chosen."""
    return ceil(len(text) / 4) if text else 0


class ContextBuilder:
    """Select complete ranked evidence blocks without exceeding a context budget."""

    def __init__(self, token_estimator: TokenEstimator = estimate_context_tokens) -> None:
        self.token_estimator = token_estimator

    def build(
        self,
        retrieved_chunks: Sequence[RetrievedChunk],
        max_context_tokens: int = 1_000,
    ) -> ContextBundle:
        """Deduplicate, budget, and format retrieved chunks in relevance order."""
        if isinstance(max_context_tokens, bool) or not isinstance(max_context_tokens, int):
            raise TypeError("max_context_tokens must be an integer")
        if max_context_tokens < 1:
            raise ValueError("max_context_tokens must be at least 1")

        selected_blocks: list[str] = []
        provenance: list[dict[str, Any]] = []
        seen_texts: set[str] = set()
        seen_provenance: set[tuple[Any, Any, Any]] = set()

        for chunk in retrieved_chunks:
            if not isinstance(chunk, RetrievedChunk):
                raise TypeError("retrieved_chunks must contain RetrievedChunk instances")
            if not isinstance(chunk.text, str) or not chunk.text.strip():
                continue

            text = chunk.text.strip()
            metadata = chunk.metadata if isinstance(chunk.metadata, dict) else {}
            chunk_provenance = {key: metadata.get(key) for key in _PROVENANCE_KEYS}
            provenance_key = tuple(chunk_provenance[key] for key in _PROVENANCE_KEYS)

            if text in seen_texts:
                continue
            if any(value is not None for value in provenance_key) and provenance_key in seen_provenance:
                continue

            block = self._format_evidence_block(
                evidence_number=len(selected_blocks) + 1,
                text=text,
                provenance=chunk_provenance,
            )
            candidate_text = "\n\n".join([*selected_blocks, block])
            estimated_tokens = self.token_estimator(candidate_text)
            if estimated_tokens > max_context_tokens:
                break

            selected_blocks.append(block)
            provenance.append(chunk_provenance)
            seen_texts.add(text)
            if any(value is not None for value in provenance_key):
                seen_provenance.add(provenance_key)

        text = "\n\n".join(selected_blocks)
        return ContextBundle(
            text=text,
            evidence_count=len(selected_blocks),
            estimated_token_count=self.token_estimator(text),
            provenance=provenance,
        )

    @staticmethod
    def _format_evidence_block(
        evidence_number: int,
        text: str,
        provenance: dict[str, Any],
    ) -> str:
        return (
            f"[Evidence {evidence_number}]\n"
            f"query_id: {provenance['query_id'] if provenance['query_id'] is not None else ''}\n"
            f"passage_index: {provenance['passage_index'] if provenance['passage_index'] is not None else ''}\n"
            f"chunk_index: {provenance['chunk_index'] if provenance['chunk_index'] is not None else ''}\n"
            f"text: {text}"
        )
