"""Production hybrid retrieval: persistent E5/Qdrant plus in-memory BM25 and RRF."""

from __future__ import annotations

import time
from concurrent.futures import Executor, ThreadPoolExecutor
from dataclasses import dataclass

from app.rag.retrieval.bm25_store import BM25Profile, BM25Store
from app.rag.retrieval.retriever import RetrievedChunk, Retriever, RetrieverProfile


@dataclass(frozen=True)
class HybridRetrievedChunk:
    """One fused chunk while retaining independent semantic and lexical scores."""

    rank: int
    text: str
    metadata: dict[str, object]
    semantic_score: float | None
    lexical_score: float | None
    fused_score: float

    @property
    def bm25_score(self) -> float | None:
        """Expose the lexical score with its retrieval-method name."""
        return self.lexical_score

    @property
    def provenance(self) -> tuple[str, str, str]:
        return tuple(str(self.metadata.get(key, "")) for key in ("query_id", "passage_index", "chunk_index"))

    def as_retrieved_chunk(self) -> RetrievedChunk | None:
        """Adapt hybrid evidence to AnswerComposer's semantic-confidence contract.

        RRF determines rank only. AnswerComposer's score guardrail is calibrated to
        semantic cosine scores, so BM25-only evidence is deliberately excluded rather
        than receiving a fabricated confidence value.
        """
        if self.semantic_score is None:
            return None
        metadata = dict(self.metadata)
        metadata.update(
            {
                "semantic_score": self.semantic_score,
                "bm25_score": self.bm25_score,
                "fused_score": self.fused_score,
            }
        )
        return RetrievedChunk(score=self.semantic_score, text=self.text, metadata=metadata)


@dataclass(frozen=True)
class HybridRetrievalResult:
    """Fused and source rankings plus the measured steady-state stage timings."""

    semantic: list[HybridRetrievedChunk]
    lexical: list[HybridRetrievedChunk]
    fused: list[HybridRetrievedChunk]
    semantic_latency_ms: float
    lexical_latency_ms: float
    fusion_latency_ms: float
    wall_clock_latency_ms: float | None = None
    embedding_latency_ms: float | None = None
    qdrant_search_latency_ms: float | None = None
    result_conversion_latency_ms: float | None = None
    qdrant_operation_latency_ms: float | None = None
    qdrant_retry_wait_ms: float | None = None
    qdrant_retry_count: int = 0
    bm25_profile: BM25Profile | None = None
    execution_mode: str = "sequential"
    executor_submit_ms: float = 0.0
    executor_queue_wait_ms: float = 0.0
    worker_start_delay_ms: float = 0.0
    worker_compute_ms: float = 0.0
    future_wait_ms: float = 0.0
    bm25_total_wall_ms: float = 0.0
    qdrant_branch_wall_ms: float = 0.0
    post_embed_parallel_wall_ms: float = 0.0

    @property
    def total_latency_ms(self) -> float:
        """Return elapsed wall time when available, otherwise preserve legacy timing."""
        if self.wall_clock_latency_ms is not None:
            return self.wall_clock_latency_ms
        return self.semantic_latency_ms + self.lexical_latency_ms + self.fusion_latency_ms


class HybridRetriever:
    """Add benchmark-validated BM25/RRF retrieval without changing E5/Qdrant behavior."""

    def __init__(
        self,
        semantic_retriever: Retriever,
        bm25_store: BM25Store,
        rrf_k: int = 60,
        executor: Executor | None = None,
    ) -> None:
        if rrf_k < 1:
            raise ValueError("rrf_k must be at least 1")
        self.semantic_retriever = semantic_retriever
        self.bm25_store = bm25_store
        self.rrf_k = rrf_k
        # One persistent worker per language-specific retriever bounds CPU work and
        # lets BM25 overlap the synchronous E5/Qdrant semantic branch.
        self._executor = executor or ThreadPoolExecutor(max_workers=1, thread_name_prefix="bm25-retrieval")
        self._owns_executor = executor is None

    def retrieve(self, query: str, top_k: int = 5, target_lang: str | None = None) -> HybridRetrievalResult:
        """Embed first, then overlap Qdrant network wait with bounded BM25 work."""
        self._validate_top_k(top_k)
        normalized_query = self._normalize_query(query)
        started_at = time.perf_counter()
        if hasattr(self.semantic_retriever, "embed_query_profiled"):
            query_vector, embedding_profile = self.semantic_retriever.embed_query_profiled(normalized_query)
            post_embed_started_at = time.perf_counter()
            submit_started_at = time.perf_counter()
            lexical_future = self._executor.submit(self._lexical_worker, submit_started_at, normalized_query, top_k)
            executor_submit_ms = (time.perf_counter() - submit_started_at) * 1_000
            try:
                semantic_chunks, qdrant_profile = self.semantic_retriever.retrieve_from_query_vector_profiled(query_vector, top_k, target_lang)
            except BaseException:
                try:
                    lexical_future.result()
                except BaseException:
                    pass
                raise
            future_wait_started_at = time.perf_counter()
            lexical, lexical_latency_ms, bm25_profile, worker_started_at = lexical_future.result()
            future_wait_ms = (time.perf_counter() - future_wait_started_at) * 1_000
            semantic = [self._from_semantic(rank, chunk) for rank, chunk in enumerate(semantic_chunks, start=1)]
            semantic_profile = RetrieverProfile(
                embedding_ms=embedding_profile.embedding_ms,
                qdrant_search_ms=qdrant_profile.qdrant_search_ms,
                result_conversion_ms=qdrant_profile.result_conversion_ms,
                total_ms=embedding_profile.embedding_ms + qdrant_profile.total_ms,
                qdrant_operation_ms=qdrant_profile.qdrant_operation_ms,
                qdrant_retry_wait_ms=qdrant_profile.qdrant_retry_wait_ms,
                qdrant_wall_ms=qdrant_profile.qdrant_wall_ms,
                qdrant_retry_count=qdrant_profile.qdrant_retry_count,
            )
            return self._fuse_result(
                semantic, lexical, top_k, started_at, semantic_profile.total_ms, lexical_latency_ms,
                semantic_profile, bm25_profile, execution_mode="post_embedding_parallel",
                executor_submit_ms=executor_submit_ms,
                executor_queue_wait_ms=(worker_started_at - submit_started_at) * 1_000,
                worker_start_delay_ms=(worker_started_at - submit_started_at) * 1_000,
                worker_compute_ms=lexical_latency_ms,
                future_wait_ms=future_wait_ms,
                bm25_total_wall_ms=lexical_latency_ms,
                qdrant_branch_wall_ms=qdrant_profile.total_ms,
                post_embed_parallel_wall_ms=(time.perf_counter() - post_embed_started_at) * 1_000,
            )
        return self.retrieve_old_parallel(normalized_query, top_k, target_lang)

    def retrieve_old_parallel(self, query: str, top_k: int = 5, target_lang: str | None = None) -> HybridRetrievalResult:
        """Retain prior E5/BM25 overlap only for diagnostic comparisons."""
        self._validate_top_k(top_k)
        normalized_query = self._normalize_query(query)
        started_at = time.perf_counter()
        submit_started_at = time.perf_counter()
        lexical_future = self._executor.submit(self._lexical_worker, submit_started_at, normalized_query, top_k)
        executor_submit_ms = (time.perf_counter() - submit_started_at) * 1_000
        try:
            semantic, semantic_latency_ms, semantic_profile = self._semantic_retrieve(
                normalized_query, top_k, target_lang
            )
        except BaseException:
            # A running worker cannot be cancelled safely; consume it before
            # propagating the semantic failure rather than leaving an orphan task.
            try:
                lexical_future.result()
            except BaseException:
                pass
            raise
        future_wait_started_at = time.perf_counter()
        lexical, lexical_latency_ms, bm25_profile, worker_started_at = lexical_future.result()
        future_wait_ms = (time.perf_counter() - future_wait_started_at) * 1_000
        return self._fuse_result(
            semantic,
            lexical,
            top_k,
            started_at,
            semantic_latency_ms,
            lexical_latency_ms,
            semantic_profile,
            bm25_profile,
            execution_mode="parallel",
            executor_submit_ms=executor_submit_ms,
            executor_queue_wait_ms=(worker_started_at - submit_started_at) * 1_000,
            worker_start_delay_ms=(worker_started_at - submit_started_at) * 1_000,
            worker_compute_ms=lexical_latency_ms,
            future_wait_ms=future_wait_ms,
            bm25_total_wall_ms=(time.perf_counter() - submit_started_at) * 1_000,
        )

    def retrieve_sequential(self, query: str, top_k: int = 5, target_lang: str | None = None) -> HybridRetrievalResult:
        """Run the original branch order for equivalence validation and benchmarking."""
        self._validate_top_k(top_k)
        normalized_query = self._normalize_query(query)
        started_at = time.perf_counter()
        semantic, semantic_latency_ms, semantic_profile = self._semantic_retrieve(normalized_query, top_k, target_lang)
        lexical, lexical_latency_ms, bm25_profile = self._lexical_retrieve(normalized_query, top_k)
        return self._fuse_result(
            semantic,
            lexical,
            top_k,
            started_at,
            semantic_latency_ms,
            lexical_latency_ms,
            semantic_profile,
            bm25_profile,
            execution_mode="sequential",
            worker_compute_ms=lexical_latency_ms,
            bm25_total_wall_ms=lexical_latency_ms,
        )

    def close(self) -> None:
        """Release the retriever-owned bounded worker during application shutdown."""
        if self._owns_executor:
            self._executor.shutdown(wait=True, cancel_futures=False)

    @staticmethod
    def _validate_top_k(top_k: int) -> None:
        if top_k < 1:
            raise ValueError("top_k must be at least 1")

    @staticmethod
    def _normalize_query(query: str) -> str:
        if not isinstance(query, str):
            raise TypeError("query must be a string")
        normalized_query = " ".join(query.split())
        if not normalized_query:
            raise ValueError("query must be non-empty")
        return normalized_query

    def _semantic_retrieve(
        self,
        query: str,
        top_k: int,
        target_lang: str | None,
    ) -> tuple[list[HybridRetrievedChunk], float, RetrieverProfile | None]:
        semantic_started_at = time.perf_counter()
        if hasattr(self.semantic_retriever, "retrieve_profiled"):
            semantic_chunks, semantic_profile = self.semantic_retriever.retrieve_profiled(
                query, top_k=top_k, target_lang=target_lang
            )
        else:
            semantic_chunks = self.semantic_retriever.retrieve(query, top_k=top_k, target_lang=target_lang)
            semantic_profile = None
        semantic_latency_ms = (time.perf_counter() - semantic_started_at) * 1_000
        semantic = [self._from_semantic(rank, chunk) for rank, chunk in enumerate(semantic_chunks, start=1)]
        return semantic, semantic_latency_ms, semantic_profile

    def _lexical_retrieve(self, query: str, top_k: int) -> tuple[list[HybridRetrievedChunk], float, BM25Profile]:
        lexical_started_at = time.perf_counter()
        lexical_matches, bm25_profile = self.bm25_store.search_profiled(query, top_k=top_k)
        lexical_latency_ms = (time.perf_counter() - lexical_started_at) * 1_000
        lexical = [
            HybridRetrievedChunk(match.rank, match.document.text, dict(match.document.metadata), None, match.score, 0.0)
            for match in lexical_matches
        ]
        return lexical, lexical_latency_ms, bm25_profile

    def _lexical_worker(self, submitted_at: float, query: str, top_k: int) -> tuple[list[HybridRetrievedChunk], float, BM25Profile, float]:
        """Timestamp the persistent worker start before executing unchanged BM25."""
        worker_started_at = time.perf_counter()
        lexical, lexical_latency_ms, bm25_profile = self._lexical_retrieve(query, top_k)
        return lexical, lexical_latency_ms, bm25_profile, worker_started_at

    def _fuse_result(
        self,
        semantic: list[HybridRetrievedChunk],
        lexical: list[HybridRetrievedChunk],
        top_k: int,
        started_at: float,
        semantic_latency_ms: float,
        lexical_latency_ms: float,
        semantic_profile: RetrieverProfile | None,
        bm25_profile: BM25Profile,
        execution_mode: str,
        executor_submit_ms: float = 0.0,
        executor_queue_wait_ms: float = 0.0,
        worker_start_delay_ms: float = 0.0,
        worker_compute_ms: float = 0.0,
        future_wait_ms: float = 0.0,
        bm25_total_wall_ms: float = 0.0,
        qdrant_branch_wall_ms: float = 0.0,
        post_embed_parallel_wall_ms: float = 0.0,
    ) -> HybridRetrievalResult:
        fusion_started_at = time.perf_counter()
        fused = self._fuse(semantic, lexical, top_k)
        fusion_latency_ms = (time.perf_counter() - fusion_started_at) * 1_000
        return HybridRetrievalResult(
            semantic=semantic,
            lexical=lexical,
            fused=fused,
            semantic_latency_ms=semantic_latency_ms,
            lexical_latency_ms=lexical_latency_ms,
            fusion_latency_ms=fusion_latency_ms,
            wall_clock_latency_ms=(time.perf_counter() - started_at) * 1_000,
            embedding_latency_ms=semantic_profile.embedding_ms if semantic_profile is not None else None,
            qdrant_search_latency_ms=semantic_profile.qdrant_search_ms if semantic_profile is not None else None,
            result_conversion_latency_ms=semantic_profile.result_conversion_ms if semantic_profile is not None else None,
            qdrant_operation_latency_ms=semantic_profile.qdrant_operation_ms if semantic_profile is not None else None,
            qdrant_retry_wait_ms=semantic_profile.qdrant_retry_wait_ms if semantic_profile is not None else None,
            qdrant_retry_count=semantic_profile.qdrant_retry_count if semantic_profile is not None else 0,
            bm25_profile=bm25_profile,
            execution_mode=execution_mode,
            executor_submit_ms=executor_submit_ms,
            executor_queue_wait_ms=executor_queue_wait_ms,
            worker_start_delay_ms=worker_start_delay_ms,
            worker_compute_ms=worker_compute_ms,
            future_wait_ms=future_wait_ms,
            bm25_total_wall_ms=bm25_total_wall_ms,
            qdrant_branch_wall_ms=qdrant_branch_wall_ms,
            post_embed_parallel_wall_ms=post_embed_parallel_wall_ms,
        )

    @staticmethod
    def _from_semantic(rank: int, chunk: RetrievedChunk) -> HybridRetrievedChunk:
        return HybridRetrievedChunk(rank, chunk.text, dict(chunk.metadata), chunk.score, None, 0.0)

    def _fuse(
        self,
        semantic: list[HybridRetrievedChunk],
        lexical: list[HybridRetrievedChunk],
        top_k: int,
    ) -> list[HybridRetrievedChunk]:
        candidates: dict[tuple[str, str, str], dict[str, object]] = {}
        for result in semantic:
            entry = candidates.setdefault(
                result.provenance,
                {"chunk": result, "semantic": None, "lexical": None, "score": 0.0},
            )
            entry["semantic"] = result.semantic_score
            entry["score"] = float(entry["score"]) + 1 / (self.rrf_k + result.rank)
        for result in lexical:
            entry = candidates.setdefault(
                result.provenance,
                {"chunk": result, "semantic": None, "lexical": None, "score": 0.0},
            )
            entry["lexical"] = result.lexical_score
            entry["score"] = float(entry["score"]) + 1 / (self.rrf_k + result.rank)

        ordered = sorted(candidates.values(), key=lambda entry: (-float(entry["score"]), entry["chunk"].provenance))
        return [
            HybridRetrievedChunk(
                rank=rank,
                text=entry["chunk"].text,
                metadata=dict(entry["chunk"].metadata),
                semantic_score=entry["semantic"],
                lexical_score=entry["lexical"],
                fused_score=float(entry["score"]),
            )
            for rank, entry in enumerate(ordered[:top_k], start=1)
        ]
