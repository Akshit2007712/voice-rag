"""Long-lived dependencies for the deterministic text-RAG request path."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from app.rag.generation.answer_composer import AnswerComposer, ComposedAnswer
from app.rag.generation.gemini_composer import GeminiAnswerComposer
from app.rag.indexing.embedder import E5Embedder
from app.rag.indexing.vector_store import VectorStore
from app.rag.retrieval.retriever import RetrievedChunk, Retriever



@dataclass(frozen=True)
class RetrievalBreakdown:
    """Steady-state timing and output of one text-RAG retrieval/composition pass."""

    chunks: list[RetrievedChunk]
    answer: ComposedAnswer
    embedding_ms: float
    qdrant_search_ms: float
    result_conversion_ms: float
    composer_ms: float
    total_ms: float


class RAGRuntime:
    """Own E5, Qdrant, retrieval, and composition for one process lifetime."""

    def __init__(
        self,
        embedder: E5Embedder | None = None,
        vector_store: VectorStore | None = None,
        answer_composer: AnswerComposer | GeminiAnswerComposer | None = None,
    ) -> None:
        if embedder is None:
            print("Loading E5 model...", flush=True)
            embedder = E5Embedder()
            print("E5 READY", flush=True)
            print(f"DEVICE: {embedder.device}", flush=True)
        self.embedder = embedder
        self.vector_store = vector_store or VectorStore()
        self.retriever = Retriever(self.embedder, self.vector_store)
        # Use GeminiAnswerComposer by default; it self-detects GEMINI_API_KEY
        # and gracefully falls back to the extractive composer when unset.
        self.answer_composer = answer_composer or GeminiAnswerComposer()


    def retrieve(self, query: str, top_k: int = 5, target_lang: str | None = None) -> list[RetrievedChunk]:
        """Use the persistent Retriever for ordinary application requests."""
        return self.retriever.retrieve(query, top_k=top_k, target_lang=target_lang)

    def answer(
        self,
        query: str,
        top_k: int = 5,
        target_lang: str | None = None,
        max_sentences: int = 3,
    ) -> ComposedAnswer:
        """Compose a grounded answer using persistent runtime dependencies."""
        return self.answer_composer.compose(query, self.retrieve(query, top_k, target_lang), max_sentences=max_sentences)

    def measure_answer(
        self,
        query: str,
        top_k: int = 5,
        target_lang: str | None = None,
        max_sentences: int = 3,
    ) -> RetrievalBreakdown:
        """Run the same stages with component timings for development benchmarking."""
        normalized_query = " ".join(query.split())
        if not normalized_query:
            raise ValueError("query must be non-empty")
        started_at = time.perf_counter()
        self._sync_cuda()
        embedding_started_at = time.perf_counter()
        query_vector = self.embedder.embed_query(normalized_query)
        self._sync_cuda()
        embedding_ms = (time.perf_counter() - embedding_started_at) * 1_000

        search_started_at = time.perf_counter()
        hits = self.vector_store.search(query_vector, limit=top_k, target_lang=target_lang)
        qdrant_search_ms = (time.perf_counter() - search_started_at) * 1_000

        conversion_started_at = time.perf_counter()
        chunks = []
        for hit in hits:
            payload = dict(hit.payload or {})
            chunks.append(RetrievedChunk(float(hit.score), str(payload.pop("text", "")), payload))
        result_conversion_ms = (time.perf_counter() - conversion_started_at) * 1_000

        answer = self.answer_composer.compose(normalized_query, chunks, max_sentences=max_sentences)
        return RetrievalBreakdown(
            chunks=chunks,
            answer=answer,
            embedding_ms=embedding_ms,
            qdrant_search_ms=qdrant_search_ms,
            result_conversion_ms=result_conversion_ms,
            composer_ms=answer.latency_ms,
            total_ms=(time.perf_counter() - started_at) * 1_000,
        )

    def close(self) -> None:
        """Close the owned Qdrant client when the process/runtime ends."""
        self.vector_store.close()

    def _sync_cuda(self) -> None:
        if getattr(self.embedder, "device", None) == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()
