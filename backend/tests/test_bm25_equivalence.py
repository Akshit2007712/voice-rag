"""Focused checks for the BM25 inverted-index execution path."""

from __future__ import annotations

import unittest

from app.rag.retrieval.bm25_store import BM25Store, LexicalDocument


def document(text: str, query_id: str, passage_index: int) -> LexicalDocument:
    return LexicalDocument(
        text=text,
        metadata={"query_id": query_id, "passage_index": passage_index, "chunk_index": 0},
    )


class BM25EquivalenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.store = BM25Store(
            [
                document("फिलाडेल्फिया लैंकेस्टर दूरी यात्रा", "hi-1", 0),
                document("लैंकेस्टर फिलाडेल्फिया मार्ग दूरी", "hi-2", 0),
                document("दिल्ली भारत की राजधानी", "hi-3", 0),
                document("Philadelphia Lancaster travel distance", "en-1", 0),
                document("Lancaster Philadelphia route distance", "en-2", 0),
                document("Delhi is the capital of India", "en-3", 0),
            ]
        )

    def test_inverted_index_matches_full_corpus_reference_for_deterministic_queries(self) -> None:
        queries = [
            "फिलाडेल्फिया लैंकेस्टर",
            "लैंकेस्टर दूरी",
            "दिल्ली राजधानी",
            "Philadelphia Lancaster",
            "Lancaster distance",
            "capital India",
        ] * 9
        for query in queries[:50]:
            with self.subTest(query=query):
                old = self.store.search_full_corpus_reference(query, top_k=5)
                new = self.store.search(query, top_k=5)
                self.assertEqual(
                    [match.document.provenance for match in old],
                    [match.document.provenance for match in new],
                )
                self.assertEqual([match.score for match in old], [match.score for match in new])

    def test_postings_are_built_once_and_not_rebuilt_per_query(self) -> None:
        postings_identity = id(self.store._postings)
        posting_sizes = {term: len(entries) for term, entries in self.store._postings.items()}
        self.store.search("फिलाडेल्फिया दूरी")
        self.store.search("Philadelphia distance")
        self.assertEqual(id(self.store._postings), postings_identity)
        self.assertEqual({term: len(entries) for term, entries in self.store._postings.items()}, posting_sizes)

    def test_profile_reports_all_production_stages(self) -> None:
        _matches, profile = self.store.search_profiled("फिलाडेल्फिया लैंकेस्टर", top_k=5)
        self.assertGreaterEqual(profile.tokenize_ms, 0.0)
        self.assertGreaterEqual(profile.candidate_lookup_ms, 0.0)
        self.assertGreaterEqual(profile.score_ms, 0.0)
        self.assertGreaterEqual(profile.topk_ms, 0.0)
        self.assertGreaterEqual(profile.total_bm25_ms, 0.0)

    def test_no_matching_term_has_identical_empty_results(self) -> None:
        self.assertEqual(self.store.search_full_corpus_reference("अनुपस्थित"), [])
        self.assertEqual(self.store.search("अनुपस्थित"), [])


if __name__ == "__main__":
    unittest.main()
