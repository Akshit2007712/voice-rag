"""Query embedding and structured Qdrant retrieval, without answer generation."""

from dataclasses import dataclass
import time
from typing import Any

from app.rag.indexing.embedder import E5Embedder
from app.rag.indexing.vector_store import VectorStore


@dataclass(frozen=True)
class RetrievedChunk:
    """A Qdrant search hit converted into application-level retrieval data."""

    score: float
    text: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class RetrieverProfile:
    """Measured stages of one E5 and Qdrant semantic retrieval branch."""

    embedding_ms: float
    qdrant_search_ms: float
    result_conversion_ms: float
    total_ms: float
    qdrant_operation_ms: float
    qdrant_retry_wait_ms: float
    qdrant_wall_ms: float
    qdrant_retry_count: int


@dataclass(frozen=True)
class QueryEmbeddingProfile:
    """Caller-thread E5 timing before any post-embedding concurrent work."""

    embedding_ms: float


@dataclass(frozen=True)
class QdrantBranchProfile:
    """Qdrant/search-result branch timings using an already prepared query vector."""

    qdrant_search_ms: float
    result_conversion_ms: float
    total_ms: float
    qdrant_operation_ms: float
    qdrant_retry_wait_ms: float
    qdrant_wall_ms: float
    qdrant_retry_count: int


class Retriever:
    """Embed a user query with E5 and retrieve ranked chunks from VectorStore."""

    def __init__(self, embedder: E5Embedder, vector_store: VectorStore) -> None:
        self.embedder = embedder
        self.vector_store = vector_store

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        target_lang: str | None = None,
    ) -> list[RetrievedChunk]:
        """Return semantic results and retain the existing development timing logs."""
        results, profile = self.retrieve_profiled(query, top_k, target_lang)
        print(f"E5 EMBEDDING LATENCY_MS: {profile.embedding_ms:.2f}")
        print(f"QDRANT SEARCH LATENCY_MS: {profile.qdrant_search_ms:.2f}")
        print(f"RESULT CONVERSION LATENCY_MS: {profile.result_conversion_ms:.2f}")
        return results

    def retrieve_profiled(
        self,
        query: str,
        top_k: int = 5,
        target_lang: str | None = None,
    ) -> tuple[list[RetrievedChunk], RetrieverProfile]:
        """Return semantic results and timings without mutating runtime resources."""
        if not isinstance(query, str):
            raise TypeError("query must be a string")

        normalized_query = " ".join(query.split())

        if not normalized_query:
            raise ValueError("query must be non-empty")

        if top_k < 1:
            raise ValueError("top_k must be at least 1")

        if target_lang is not None and (
            not isinstance(target_lang, str) or not target_lang.strip()
        ):
            raise ValueError(
                "target_lang must be a non-empty string when provided"
            )

        branch_started_at = time.perf_counter()
        query_vector, embedding_profile = self.embed_query_profiled(normalized_query)
        results, qdrant_profile = self.retrieve_from_query_vector_profiled(query_vector, top_k, target_lang)

        return results, RetrieverProfile(
            embedding_ms=embedding_profile.embedding_ms,
            qdrant_search_ms=qdrant_profile.qdrant_search_ms,
            result_conversion_ms=qdrant_profile.result_conversion_ms,
            total_ms=(time.perf_counter() - branch_started_at) * 1_000,
            qdrant_operation_ms=qdrant_profile.qdrant_operation_ms,
            qdrant_retry_wait_ms=qdrant_profile.qdrant_retry_wait_ms,
            qdrant_wall_ms=qdrant_profile.qdrant_wall_ms,
            qdrant_retry_count=qdrant_profile.qdrant_retry_count,
        )

    def embed_query_profiled(self, normalized_query: str) -> tuple[Any, QueryEmbeddingProfile]:
        """Embed on the caller thread before starting lexical work."""
        started_at = time.perf_counter()
        vector = self.embedder.embed_query(normalized_query)
        return vector, QueryEmbeddingProfile((time.perf_counter() - started_at) * 1_000)

    def retrieve_from_query_vector_profiled(
        self, query_vector: Any, top_k: int, target_lang: str | None
    ) -> tuple[list[RetrievedChunk], QdrantBranchProfile]:
        """Search Qdrant using a completed vector; no E5 work occurs here."""
        started_at = time.perf_counter()
        search_started_at = started_at
        hits = self.vector_store.search(query_vector, limit=top_k, target_lang=target_lang)
        search_latency_ms = (time.perf_counter() - search_started_at) * 1_000
        network_metrics = getattr(self.vector_store, "last_qdrant_operation_metrics", None)
        conversion_started_at = time.perf_counter()
        results = []
        for hit in hits:
            payload = dict(hit.payload or {})
            results.append(RetrievedChunk(float(hit.score), str(payload.pop("text", "")), payload))
        conversion_latency_ms = (time.perf_counter() - conversion_started_at) * 1_000
        return results, QdrantBranchProfile(
            qdrant_search_ms=search_latency_ms,
            result_conversion_ms=conversion_latency_ms,
            total_ms=(time.perf_counter() - started_at) * 1_000,
            qdrant_operation_ms=(network_metrics.operation_ms if network_metrics is not None else search_latency_ms),
            qdrant_retry_wait_ms=(network_metrics.retry_wait_ms if network_metrics is not None else 0.0),
            qdrant_wall_ms=(network_metrics.wall_ms if network_metrics is not None else search_latency_ms),
            qdrant_retry_count=(network_metrics.retry_count if network_metrics is not None else 0),
        )
