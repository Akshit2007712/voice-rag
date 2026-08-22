"""Focused concurrency and semantic-equivalence checks for HybridRetriever."""

from __future__ import annotations

import threading
import time
import unittest

from app.rag.retrieval.bm25_store import BM25Match, BM25Profile, LexicalDocument
from app.rag.retrieval.hybrid_retriever import HybridRetriever
from app.rag.retrieval.retriever import RetrievedChunk


DOCUMENT = LexicalDocument("फिलाडेल्फिया लैंकेस्टर दूरी", {"query_id": "1", "passage_index": 0, "chunk_index": 0})


class BlockingSemanticRetriever:
    def __init__(self, bm25_started: threading.Event) -> None:
        self.bm25_started = bm25_started
        self.started = threading.Event()

    def retrieve(self, *_args, **_kwargs):
        self.started.set()
        if not self.bm25_started.wait(1):
            raise TimeoutError("BM25 did not overlap semantic retrieval")
        return [RetrievedChunk(0.9, DOCUMENT.text, dict(DOCUMENT.metadata))]


class BlockingBM25Store:
    def __init__(self, semantic_started: threading.Event) -> None:
        self.semantic_started = semantic_started
        self.started = threading.Event()

    def search_profiled(self, *_args, **_kwargs):
        self.started.set()
        if not self.semantic_started.wait(1):
            raise TimeoutError("Semantic branch did not overlap BM25")
        return [BM25Match(1, DOCUMENT, 2.0)], BM25Profile(0, 0, 0, 0, 0)


class RaisingBM25Store:
    def search_profiled(self, *_args, **_kwargs):
        raise RuntimeError("bm25 failure")


class ParallelHybridRetrieverTests(unittest.TestCase):
    def test_semantic_and_bm25_branches_overlap_and_rrf_waits_for_both(self) -> None:
        retriever = HybridRetriever(BlockingSemanticRetriever(threading.Event()), BlockingBM25Store(threading.Event()))
        # Link the two fakes to each other's real start events.
        retriever.semantic_retriever.bm25_started = retriever.bm25_store.started
        retriever.bm25_store.semantic_started = retriever.semantic_retriever.started
        try:
            result = retriever.retrieve("फिलाडेल्फिया दूरी", 5, "hin_Deva")
        finally:
            retriever.close()
        self.assertTrue(retriever.semantic_retriever.started.is_set())
        self.assertTrue(retriever.bm25_store.started.is_set())
        self.assertEqual(result.execution_mode, "parallel")
        self.assertEqual(len(result.fused), 1)

    def test_branch_exception_propagates_without_returning_partial_results(self) -> None:
        class Semantic:
            def retrieve(self, *_args, **_kwargs):
                return [RetrievedChunk(0.9, DOCUMENT.text, dict(DOCUMENT.metadata))]

        retriever = HybridRetriever(Semantic(), RaisingBM25Store())
        try:
            with self.assertRaisesRegex(RuntimeError, "bm25 failure"):
                retriever.retrieve("फिलाडेल्फिया दूरी", 5, "hin_Deva")
        finally:
            retriever.close()

    def test_sequential_and_parallel_results_have_identical_rankings(self) -> None:
        class Semantic:
            def retrieve(self, *_args, **_kwargs):
                return [RetrievedChunk(0.9, DOCUMENT.text, dict(DOCUMENT.metadata))]

        class BM25:
            def search_profiled(self, *_args, **_kwargs):
                return [BM25Match(1, DOCUMENT, 2.0)], BM25Profile(0, 0, 0, 0, 0)

        retriever = HybridRetriever(Semantic(), BM25())
        try:
            sequential = retriever.retrieve_sequential("फिलाडेल्फिया दूरी", 5, "hin_Deva")
            parallel = retriever.retrieve("फिलाडेल्फिया दूरी", 5, "hin_Deva")
        finally:
            retriever.close()
        self.assertEqual(
            [(item.provenance, item.semantic_score, item.lexical_score, item.fused_score) for item in sequential.fused],
            [(item.provenance, item.semantic_score, item.lexical_score, item.fused_score) for item in parallel.fused],
        )


if __name__ == "__main__":
    unittest.main()
