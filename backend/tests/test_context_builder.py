"""Focused tests for bounded retrieval-context construction."""

import unittest

from app.rag.generation.context_builder import ContextBuilder
from app.rag.retrieval.retriever import RetrievedChunk


def chunk(text: str, **metadata: object) -> RetrievedChunk:
    return RetrievedChunk(score=0.9, text=text, metadata=dict(metadata))


class ContextBuilderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.builder = ContextBuilder(token_estimator=lambda text: len(text.split()))

    def test_preserves_retrieval_order(self) -> None:
        bundle = self.builder.build([
            chunk("first text", query_id=1, passage_index=0, chunk_index=0),
            chunk("second text", query_id=2, passage_index=1, chunk_index=0),
        ], max_context_tokens=100)
        self.assertLess(bundle.text.index("first text"), bundle.text.index("second text"))

    def test_removes_exact_duplicate_text(self) -> None:
        bundle = self.builder.build([
            chunk("same text", query_id=1, passage_index=0, chunk_index=0),
            chunk("same text", query_id=2, passage_index=0, chunk_index=0),
        ], max_context_tokens=100)
        self.assertEqual(bundle.evidence_count, 1)

    def test_removes_duplicate_provenance(self) -> None:
        bundle = self.builder.build([
            chunk("first version", query_id=1, passage_index=0, chunk_index=0),
            chunk("second version", query_id=1, passage_index=0, chunk_index=0),
        ], max_context_tokens=100)
        self.assertEqual(bundle.evidence_count, 1)
        self.assertIn("first version", bundle.text)

    def test_empty_input_and_empty_text_are_clean(self) -> None:
        self.assertEqual(self.builder.build([], 100).text, "")
        self.assertEqual(self.builder.build([chunk("   ")], 100).evidence_count, 0)

    def test_invalid_context_budget_raises(self) -> None:
        with self.assertRaises(ValueError):
            self.builder.build([], 0)
        with self.assertRaises(TypeError):
            self.builder.build([], 1.5)  # type: ignore[arg-type]

    def test_context_respects_budget_without_partial_chunk(self) -> None:
        first = "one two three"
        second = "four five six"
        one_block_tokens = len(self.builder._format_evidence_block(1, first, {
            "query_id": 1, "passage_index": 0, "chunk_index": 0,
        }).split())
        bundle = self.builder.build([
            chunk(first, query_id=1, passage_index=0, chunk_index=0),
            chunk(second, query_id=2, passage_index=0, chunk_index=0),
        ], max_context_tokens=one_block_tokens)
        self.assertEqual(bundle.evidence_count, 1)
        self.assertIn(first, bundle.text)
        self.assertNotIn(second, bundle.text)
        self.assertLessEqual(bundle.estimated_token_count, one_block_tokens)

    def test_context_exposes_only_lightweight_provenance(self) -> None:
        bundle = self.builder.build([chunk(
            "evidence text",
            query_id=7,
            passage_index=3,
            chunk_index=2,
            is_selected=1,
            chunk_strategy="sentence_overlap",
            token_count=123,
            source_lang="en",
            target_lang="hi",
            query_type="entity",
        )], 100)
        self.assertIn("query_id: 7", bundle.text)
        self.assertIn("passage_index: 3", bundle.text)
        self.assertIn("chunk_index: 2", bundle.text)
        self.assertIn("text: evidence text", bundle.text)
        for forbidden in ("is_selected", "chunk_strategy", "token_count", "source_lang", "target_lang", "query_type", "score"):
            self.assertNotIn(forbidden, bundle.text)

    def test_missing_optional_metadata_does_not_fail(self) -> None:
        bundle = self.builder.build([chunk("evidence")], 100)
        self.assertEqual(bundle.evidence_count, 1)
        self.assertIn("query_id: ", bundle.text)
        self.assertEqual(bundle.provenance, [{"query_id": None, "passage_index": None, "chunk_index": None}])


if __name__ == "__main__":
    unittest.main()
