import unittest
from types import SimpleNamespace

import numpy as np

from app.rag.retrieval.retriever import Retriever


class FakeEmbedder:
    def __init__(self) -> None:
        self.queries: list[str] = []

    def embed_query(self, query: str) -> np.ndarray:
        self.queries.append(query)
        return np.array([1.0, 0.0], dtype=np.float32)


class FakeVectorStore:
    def __init__(self) -> None:
        self.calls = []
        self.hits = [
            SimpleNamespace(
                score=0.91,
                payload={
                    "text": "पहला परिणाम",
                    "query_id": 11,
                    "passage_index": 2,
                    "chunk_index": 0,
                    "is_selected": 1,
                    "chunk_strategy": "whole_passage",
                    "token_count": 12,
                },
            ),
            SimpleNamespace(score=0.82, payload={"text": "दूसरा परिणाम", "is_selected": 0}),
        ]

    def search(self, vector, limit, target_lang=None):
        self.calls.append((vector, limit, target_lang))
        return self.hits


class RetrieverTests(unittest.TestCase):
    def setUp(self) -> None:
        self.embedder = FakeEmbedder()
        self.store = FakeVectorStore()
        self.retriever = Retriever(self.embedder, self.store)

    def test_empty_query_and_invalid_top_k_raise_clear_errors(self) -> None:
        with self.assertRaisesRegex(ValueError, "query must be non-empty"):
            self.retriever.retrieve(" \t ")
        with self.assertRaisesRegex(ValueError, "top_k must be at least 1"):
            self.retriever.retrieve("प्रश्न", top_k=0)

    def test_uses_embedder_and_vector_store_in_ranking_order(self) -> None:
        results = self.retriever.retrieve("  कॉर्पोरेशन   क्या है? ", top_k=2, target_lang="hin_Deva")

        self.assertEqual(self.embedder.queries, ["कॉर्पोरेशन क्या है?"])
        self.assertEqual(self.store.calls[0][1:], (2, "hin_Deva"))
        self.assertEqual([result.score for result in results], [0.91, 0.82])
        self.assertEqual(results[0].text, "पहला परिणाम")
        self.assertEqual(results[0].metadata["query_id"], 11)
        self.assertNotIn("text", results[0].metadata)

    def test_is_selected_is_preserved_but_does_not_change_order(self) -> None:
        results = self.retriever.retrieve("प्रश्न")

        self.assertEqual([result.metadata["is_selected"] for result in results], [1, 0])
        self.assertEqual([result.score for result in results], [0.91, 0.82])


if __name__ == "__main__":
    unittest.main()
