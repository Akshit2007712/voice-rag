"""Unit tests for the benchmark-only Hindi BM25 and RRF helpers."""

import sys
import unittest
from pathlib import Path


SCRIPTS_DIRECTORY = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIRECTORY))

from hybrid_partial_retrieval import BM25Index, IndexedDocument, RankedResult, reciprocal_rank_fusion, tokenize_hindi  # noqa: E402


def document(query_id: int, text: str, passage_index: int = 0, chunk_index: int = 0) -> IndexedDocument:
    return IndexedDocument(text, str(query_id), str(passage_index), str(chunk_index), "hin_Deva")


class HybridPartialRetrievalTests(unittest.TestCase):
    def test_tokenizer_preserves_devanagari_marks_and_removes_punctuation(self) -> None:
        self.assertEqual(tokenize_hindi("लैंकेस्टर,  कितनी दूर?"), ["लैंकेस्टर", "कितनी", "दूर"])

    def test_bm25_returns_matching_current_document(self) -> None:
        expected = document(1, "फिलाडेल्फिया लैंकेस्टर से कितनी दूर है")
        index = BM25Index([expected, document(2, "दिल्ली से आगरा की यात्रा")])
        results = index.search("लैंकेस्टर दूर", top_k=5)
        self.assertEqual(results[0].document.provenance, expected.provenance)

    def test_rrf_preserves_provenance_and_rewards_cross_retriever_agreement(self) -> None:
        shared, semantic_only, lexical_only = document(1, "a"), document(2, "b"), document(3, "c")
        semantic = [RankedResult(1, semantic_only, 0.9), RankedResult(2, shared, 0.8)]
        lexical = [RankedResult(1, lexical_only, 4.0), RankedResult(2, shared, 3.0)]
        fused = reciprocal_rank_fusion(semantic, lexical, rrf_k=60, top_k=5)
        self.assertEqual(fused[0].document.provenance, shared.provenance)
        self.assertIsNotNone(fused[0].fused_score)
