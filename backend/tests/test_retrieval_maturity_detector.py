"""Focused tests for the benchmark-only partial retrieval maturity policy."""

import sys
import unittest
from pathlib import Path


SCRIPTS_DIRECTORY = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIRECTORY))

from hybrid_partial_retrieval import IndexedDocument, RankedResult  # noqa: E402
from retrieval_maturity_detector import MaturityPolicy, assess_retrieval_maturity  # noqa: E402


def result(
    rank: int,
    query_id: str,
    text: str,
    *,
    passage_index: str = "0",
    chunk_index: str | None = None,
    score: float = 0.9,
) -> RankedResult:
    """Build a benchmark result without an embedder, vector DB, or external API."""
    document = IndexedDocument(text, query_id, passage_index, chunk_index or str(rank - 1), "hin_Deva")
    return RankedResult(rank, document, source_score=score, semantic_score=score, fused_score=score)


class RetrievalMaturityDetectorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.policy = MaturityPolicy()
        self.evidence = "लैंकेस्टर से फिलाडेल्फिया बहुत दूर नहीं है"

    def test_a_like_distributed_unrelated_results_are_not_mature(self) -> None:
        semantic = [result(1, "10", "दिल्ली भारत की राजधानी है", score=0.95)]
        lexical = [result(1, "20", "मुंबई समुद्र के किनारे है", score=4.0)]
        hybrid = [result(1, "10", "दिल्ली भारत की राजधानी है"), result(2, "20", "मुंबई समुद्र के किनारे है")]
        decision = assess_retrieval_maturity("विला डेल सिया लैंक", semantic, lexical, hybrid, self.policy)
        self.assertFalse(decision.mature)
        self.assertFalse(decision.corroboration_passed)

    def test_semantic_concentration_and_lexical_presence_are_mature(self) -> None:
        semantic = [result(rank, "232017", self.evidence, chunk_index=str(rank)) for rank in range(1, 6)]
        lexical = [result(1, "9", "दूसरा दस्तावेज"), result(2, "232017", self.evidence, chunk_index="1")]
        hybrid = [result(1, "232017", self.evidence, chunk_index="1")]
        decision = assess_retrieval_maturity("विला डेल सिया लैंकेस्टर से कितनी दूर", semantic, lexical, hybrid, self.policy)
        self.assertTrue(decision.mature)
        self.assertEqual(decision.maturity_path, "semantic_concentration")
        self.assertTrue(decision.semantic_concentration_passed)
        self.assertIn("232017", decision.common_query_ids_in_top_k)

    def test_hybrid_concentration_and_cross_retriever_corroboration_are_mature(self) -> None:
        semantic = [result(1, "9", "दूसरा दस्तावेज"), result(2, "232017", self.evidence, chunk_index="1")]
        lexical = [result(1, "8", "अलग दस्तावेज"), result(2, "232017", self.evidence, chunk_index="1")]
        hybrid = [
            result(1, "232017", self.evidence, chunk_index="1"),
            result(2, "232017", self.evidence, chunk_index="2"),
            result(3, "232017", self.evidence, chunk_index="3"),
            result(4, "9", "दूसरा दस्तावेज"),
            result(5, "8", "अलग दस्तावेज"),
        ]
        decision = assess_retrieval_maturity("विला डेल सिया लैंकेस्टर से कितनी दूर", semantic, lexical, hybrid, self.policy)
        self.assertTrue(decision.mature)
        self.assertEqual(decision.maturity_path, "hybrid_concentration")
        self.assertTrue(decision.hybrid_concentration_passed)

    def test_high_cosine_score_alone_never_makes_a_partial_mature(self) -> None:
        semantic = [result(1, "9", "दिल्ली भारत की राजधानी है", score=0.99)]
        decision = assess_retrieval_maturity("विला डेल सिया लैंक", semantic, [], semantic, self.policy)
        self.assertFalse(decision.mature)

    def test_bm25_only_recovery_without_semantic_corroboration_is_not_mature(self) -> None:
        semantic = [result(1, "9", "दूसरा दस्तावेज")]
        lexical = [result(1, "232017", self.evidence, score=4.0)]
        hybrid = [result(1, "232017", self.evidence)]
        decision = assess_retrieval_maturity("विला डेल सिया लैंकेस्टर से", semantic, lexical, hybrid, self.policy)
        self.assertFalse(decision.mature)
        self.assertFalse(decision.corroboration_passed)

    def test_one_sanity_term_is_enough_with_strong_corroboration(self) -> None:
        semantic = [result(rank, "232017", self.evidence, chunk_index=str(rank)) for rank in range(1, 4)]
        lexical = [result(1, "232017", self.evidence, chunk_index="1")]
        hybrid = [result(1, "232017", self.evidence, chunk_index="1")]
        decision = assess_retrieval_maturity("विला डेल सिया लैंकेस्टर से", semantic, lexical, hybrid, self.policy)
        self.assertTrue(decision.mature)
        self.assertEqual(decision.meaningful_overlap_terms, ("लैंकेस्टर",))

    def test_no_sanity_terms_and_weak_corroboration_are_not_mature(self) -> None:
        semantic = [result(1, "232017", self.evidence)]
        lexical = [result(1, "232017", self.evidence)]
        hybrid = [result(1, "232017", self.evidence)]
        decision = assess_retrieval_maturity("विला डेल सिया लैंक", semantic, lexical, hybrid, self.policy)
        self.assertFalse(decision.mature)
        self.assertFalse(decision.sanity_overlap_passed)

    def test_c_like_case_is_mature(self) -> None:
        semantic = [result(rank, "232017", self.evidence, chunk_index=str(rank)) for rank in range(1, 6)]
        lexical = [result(1, "7", "अलग दस्तावेज"), result(2, "232017", self.evidence, chunk_index="1")]
        hybrid = [result(1, "232017", self.evidence, chunk_index="1")]
        decision = assess_retrieval_maturity("विला डेल सिया लैंकेस्टर से कितनी दूर", semantic, lexical, hybrid, self.policy)
        self.assertTrue(decision.mature)
        self.assertEqual(decision.maturity_path, "semantic_concentration")

    def test_d_like_case_is_mature(self) -> None:
        semantic = [result(1, "7", "अलग दस्तावेज"), result(2, "232017", self.evidence, chunk_index="1")]
        lexical = [result(1, "8", "दूसरा दस्तावेज"), result(2, "232017", self.evidence, chunk_index="1")]
        hybrid = [
            result(1, "232017", self.evidence, chunk_index="1"),
            result(2, "232017", self.evidence, chunk_index="2"),
            result(3, "232017", self.evidence, chunk_index="3"),
            result(4, "7", "अलग दस्तावेज"),
            result(5, "8", "दूसरा दस्तावेज"),
        ]
        decision = assess_retrieval_maturity("विला डेल सिया लैंकेस्टर से कितनी दूर है", semantic, lexical, hybrid, self.policy)
        self.assertTrue(decision.mature)
        self.assertEqual(decision.maturity_path, "hybrid_concentration")

    def test_b_like_case_remains_not_mature(self) -> None:
        semantic = [result(1, "9", "दूसरा दस्तावेज"), result(2, "8", "अलग दस्तावेज")]
        lexical = [result(1, "232017", self.evidence)]
        hybrid = [result(1, "9", "दूसरा दस्तावेज"), result(2, "232017", self.evidence)]
        decision = assess_retrieval_maturity("विला डेल सिया लैंकेस्टर से", semantic, lexical, hybrid, self.policy)
        self.assertFalse(decision.mature)

    def test_cross_evidence_agreement_is_reported_without_mutating_provenance(self) -> None:
        semantic = [result(1, "232017", self.evidence, chunk_index="1")]
        lexical = [result(1, "232017", self.evidence, chunk_index="1")]
        hybrid = [result(1, "232017", self.evidence, chunk_index="1")]
        decision = assess_retrieval_maturity("लैंकेस्टर", semantic, lexical, hybrid, self.policy)
        self.assertTrue(decision.hybrid_top1_supported_by_both)
        self.assertEqual(decision.common_provenances_in_top_k, (("232017", "0", "1"),))
        self.assertEqual(hybrid[0].provenance, ("232017", "0", "1"))

    def test_detector_has_no_external_model_or_api_dependency(self) -> None:
        decision = assess_retrieval_maturity("कुछ", [], [], [], self.policy)
        self.assertFalse(decision.mature)
        self.assertEqual(decision.reason, "No hybrid evidence was returned.")
