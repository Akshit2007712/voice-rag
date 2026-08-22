"""Shared one-shot bilingual hybrid-RAG orchestration for voice and typed input."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Literal

from app.rag.generation.answer_formatter import format_composed_answer
from app.rag.language_config import get_qdrant_target_lang


InputMode = Literal["voice", "text"]
MAX_TYPED_QUERY_CHARS = 4_000


@dataclass(frozen=True)
class RagLatency:
    """Actual one-shot RAG request timings; parallel branches are not additive."""

    embedding_ms: float
    qdrant_ms: float
    bm25_ms: float
    post_embedding_parallel_ms: float
    rrf_ms: float
    maturity_ms: float
    composer_ms: float
    rag_total_ms: float
    total_ms: float = 0.0  # alias for rag_total_ms; consumed by the frontend LatencyPanel


@dataclass(frozen=True)
class TextRagResult:
    """One-shot final-answer result after query text is available."""

    query: str
    answer: object
    input_mode: InputMode
    latency: RagLatency
    qdrant_retry_count: int = 0


def run_text_rag(query_text: str, language: str, input_mode: InputMode, app) -> TextRagResult:
    """Run the frozen hybrid path without creating request-scoped RAG resources.

    One-shot HTTP requests intentionally do not fabricate realtime partials or a
    trusted candidate. They use the identical E5/Qdrant/BM25/RRF final evidence.
    """
    if input_mode not in {"voice", "text"}:
        raise ValueError("input_mode must be voice or text")
    if not isinstance(query_text, str):
        raise ValueError("Query must be text.")
    normalized_query = " ".join(query_text.split())
    if not normalized_query:
        raise ValueError("Query cannot be empty.")
    if input_mode == "text" and len(normalized_query) > MAX_TYPED_QUERY_CHARS:
        raise ValueError(f"Typed query must be at most {MAX_TYPED_QUERY_CHARS} characters.")
    target_lang = get_qdrant_target_lang(language)
    runtime = getattr(app.state, "rag_runtime", None)
    hybrid_retriever = getattr(app.state, "hybrid_retrievers", {}).get(target_lang)
    if runtime is None or hybrid_retriever is None:
        raise RuntimeError("Text retrieval runtime is unavailable.")
    # This timer deliberately begins after normalization and validation, so it
    # excludes STT and measures the request-time frozen text-RAG path only.
    started_at = time.perf_counter()
    retrieval = hybrid_retriever.retrieve(normalized_query, top_k=5, target_lang=target_lang)
    evidence = [converted for chunk in retrieval.fused if (converted := chunk.as_retrieved_chunk()) is not None]
    composer_started_at = time.perf_counter()
    answer = runtime.answer_composer.compose(normalized_query, evidence, max_sentences=3)
    composer_ms = (time.perf_counter() - composer_started_at) * 1_000
    # Formatting is intentionally local and excluded from the existing composer
    # stage metric; rag_total_ms still measures the complete request lifecycle.
    answer = format_composed_answer(normalized_query, answer, language)
    rag_total = (time.perf_counter() - started_at) * 1_000
    latency = RagLatency(
        embedding_ms=float(getattr(retrieval, "embedding_latency_ms", 0.0) or 0.0),
        qdrant_ms=float(getattr(retrieval, "qdrant_branch_wall_ms", 0.0) or 0.0),
        bm25_ms=float(getattr(retrieval, "worker_compute_ms", 0.0) or 0.0),
        post_embedding_parallel_ms=float(getattr(retrieval, "post_embed_parallel_wall_ms", 0.0) or 0.0),
        rrf_ms=float(getattr(retrieval, "fusion_latency_ms", 0.0) or 0.0),
        # One-shot HTTP requests have no partial/final maturity lifecycle.
        maturity_ms=0.0,
        composer_ms=composer_ms,
        rag_total_ms=rag_total,
        total_ms=rag_total,
    )
    return TextRagResult(
        normalized_query,
        answer,
        input_mode,
        latency,
        max(0, int(getattr(retrieval, "qdrant_retry_count", 0))),
    )
