"""Shared one-shot text-RAG tests for typed and transcribed queries."""

import unittest
from types import SimpleNamespace

from app.rag.generation.answer_composer import ComposedAnswer
from app.services.text_rag import MAX_TYPED_QUERY_CHARS, run_text_rag


class FakeChunk:
    def as_retrieved_chunk(self):
        return SimpleNamespace(score=0.9, text="evidence", metadata={})


class FakeHybrid:
    def __init__(self) -> None:
        self.calls = []

    def retrieve(self, query, top_k, target_lang):
        self.calls.append((query, top_k, target_lang))
        return SimpleNamespace(fused=[FakeChunk()])


class TextRagTests(unittest.TestCase):
    def setUp(self) -> None:
        self.answer = ComposedAnswer("answer", [], 0.9, 0.1, False)
        self.composer = SimpleNamespace(compose=lambda *_args, **_kwargs: self.answer)
        self.hi = FakeHybrid()
        self.en = FakeHybrid()
        self.app = SimpleNamespace(state=SimpleNamespace(
            rag_runtime=SimpleNamespace(answer_composer=self.composer),
            hybrid_retrievers={"hin_Deva": self.hi, "eng_Latn": self.en},
        ))

    def test_voice_and_text_share_equivalent_one_shot_hybrid_output(self) -> None:
        voice = run_text_rag("  एक   प्रश्न ", "hi", "voice", self.app)
        typed = run_text_rag("एक प्रश्न", "hi", "text", self.app)
        self.assertEqual(voice.query, typed.query)
        self.assertEqual(voice.answer, typed.answer)
        self.assertEqual(self.hi.calls[0], self.hi.calls[1])
        self.assertEqual(voice.input_mode, "voice")
        self.assertEqual(typed.input_mode, "text")

    def test_english_uses_english_hybrid_store(self) -> None:
        run_text_rag("What is the distance?", "en", "text", self.app)
        self.assertEqual(self.en.calls[0][2], "eng_Latn")
        self.assertEqual(self.hi.calls, [])

    def test_empty_and_oversized_text_queries_fail(self) -> None:
        with self.assertRaisesRegex(ValueError, "cannot be empty"):
            run_text_rag("  ", "hi", "text", self.app)
        with self.assertRaisesRegex(ValueError, "at most"):
            run_text_rag("x" * (MAX_TYPED_QUERY_CHARS + 1), "hi", "text", self.app)


if __name__ == "__main__":
    unittest.main()
