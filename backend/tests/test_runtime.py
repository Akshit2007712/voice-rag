"""Lightweight lifecycle tests for persistent RAGRuntime dependencies."""

import unittest

import numpy as np

from app.rag.generation.answer_composer import AnswerComposer
from app.rag.runtime import RAGRuntime


class FakeEmbedder:
    device = "cpu"

    def __init__(self) -> None:
        self.calls = 0

    def embed_query(self, _: str) -> np.ndarray:
        self.calls += 1
        return np.array([1.0], dtype=np.float32)


class FakeStore:
    def __init__(self) -> None:
        self.calls = 0

    def search(self, *_args, **_kwargs):
        self.calls += 1
        return []

    def close(self) -> None:
        pass


class RuntimeTests(unittest.TestCase):
    def test_injected_dependencies_are_reused_across_queries(self) -> None:
        embedder, store, composer = FakeEmbedder(), FakeStore(), AnswerComposer()
        runtime = RAGRuntime(embedder=embedder, vector_store=store, answer_composer=composer)
        runtime.retrieve("पहला प्रश्न")
        runtime.retrieve("दूसरा प्रश्न")
        self.assertIs(runtime.embedder, embedder)
        self.assertIs(runtime.vector_store, store)
        self.assertIs(runtime.retriever.embedder, embedder)
        self.assertIs(runtime.answer_composer, composer)
        self.assertEqual(embedder.calls, 2)
        self.assertEqual(store.calls, 2)

