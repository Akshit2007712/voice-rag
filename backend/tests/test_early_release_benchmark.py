"""Focused tests for opt-in dual-condition early-release instrumentation."""

import unittest

from app.rag.generation.answer_composer import AnswerComposer
from app.rag.retrieval.hybrid_retriever import HybridRetrievedChunk, HybridRetrievalResult
from app.rag.retrieval.maturity import MaturityDecision
from app.services.early_release_benchmark import EarlyReleaseBenchmark
from app.services.voice_rag_session import VoiceRAGSession


EVIDENCE_TEXT = "फिलाडेल्फिया से लैंकेस्टर की दूरी लगभग 110 किलोमीटर है।"
PARTIAL = "फिलाडेल्फिया लैंकेस्टर दूरी"


def retrieval(query_id: str = "232017", *, provenance: tuple[str, str] = ("8", "0")) -> HybridRetrievalResult:
    """Create semantically trusted hybrid evidence with a separate RRF score."""
    chunk = HybridRetrievedChunk(
        rank=1,
        text=EVIDENCE_TEXT,
        metadata={"query_id": query_id, "passage_index": provenance[0], "chunk_index": provenance[1]},
        semantic_score=0.84,
        lexical_score=2.5,
        fused_score=0.032018,
    )
    return HybridRetrievalResult([chunk], [chunk], [chunk], 30.0, 2.0, 0.2)


class FakeHybridRetriever:
    def __init__(self, results: list[HybridRetrievalResult]) -> None:
        self.results = results
        self.calls: list[str] = []

    def retrieve(self, query: str, **_kwargs) -> HybridRetrievalResult:
        self.calls.append(query)
        return self.results.pop(0)


def benchmark(hybrid: FakeHybridRetriever | None = None) -> EarlyReleaseBenchmark:
    return EarlyReleaseBenchmark(hybrid or FakeHybridRetriever([]), AnswerComposer(), "hin_Deva", 5, started_at=0.0)


def decision(
    query_id: str,
    *,
    mature: bool = False,
    count: int = 3,
    ratio: float = 0.6,
) -> MaturityDecision:
    """Create a focused maturity result without changing policy implementation."""
    return MaturityDecision(
        mature=mature,
        reason="Mature evidence." if mature else "Evidence is still being refined.",
        maturity_path="semantic_concentration" if mature else None,
        semantic_dominant_query_id=query_id,
        semantic_dominant_query_count=count,
        semantic_dominance_ratio=ratio,
        hybrid_dominant_query_id=query_id,
        hybrid_dominant_query_count=count,
        hybrid_dominance_ratio=ratio,
        common_query_ids=(query_id,),
        common_provenances=((query_id, "8", "0"),),
        hybrid_top1_supported_by_both=True,
        meaningful_overlap_terms=("लैंकेस्टर",),
        sanity_overlap_passed=True,
    )


class EarlyReleaseBenchmarkTests(unittest.TestCase):
    def test_maturity_first_then_speech_end_evaluates_at_speech_end(self) -> None:
        subject = benchmark()
        mature = retrieval()
        subject.observe_partial(PARTIAL)
        subject.record_mature(PARTIAL, mature, maturity_at=10.0)

        subject.mark_speech_end(speech_end_at=20.0)
        report = subject.finalize(mature, final_transcript_at=30.0, current_speech_end_to_answer_ms=304.0)

        self.assertTrue(report.early_release_safe)
        self.assertEqual(report.which_happened_first, "maturity")
        self.assertTrue(report.dual_condition_reached)
        self.assertEqual(report.dual_condition_at_ms, 20_000.0)
        self.assertEqual(report.speech_end_to_dual_condition_ms, 0.0)

    def test_speech_end_first_then_maturity_evaluates_at_maturity(self) -> None:
        subject = benchmark()
        mature = retrieval()
        subject.observe_partial(PARTIAL)

        subject.mark_speech_end(speech_end_at=10.0)
        subject.record_mature(PARTIAL, mature, maturity_at=20.0)
        report = subject.finalize(mature, final_transcript_at=30.0, current_speech_end_to_answer_ms=304.0)

        self.assertTrue(report.early_release_safe)
        self.assertEqual(report.which_happened_first, "speech_end")
        self.assertEqual(report.dual_condition_at_ms, 20_000.0)
        self.assertEqual(report.speech_end_to_dual_condition_ms, 10_000.0)

    def test_maturity_only_never_becomes_an_early_release_candidate(self) -> None:
        subject = benchmark()
        mature = retrieval()
        subject.observe_partial(PARTIAL)
        subject.record_mature(PARTIAL, mature, maturity_at=10.0)

        report = subject.finalize(mature, final_transcript_at=30.0, current_speech_end_to_answer_ms=None)

        self.assertTrue(report.maturity_reached)
        self.assertFalse(report.speech_end_seen)
        self.assertFalse(report.dual_condition_reached)
        self.assertFalse(report.early_release_safe)

    def test_speech_end_only_never_becomes_an_early_release_candidate(self) -> None:
        subject = benchmark()
        subject.observe_partial(PARTIAL)
        subject.mark_speech_end(speech_end_at=10.0)

        report = subject.finalize(retrieval(), final_transcript_at=30.0, current_speech_end_to_answer_ms=304.0)

        self.assertTrue(report.speech_end_seen)
        self.assertFalse(report.maturity_reached)
        self.assertFalse(report.dual_condition_reached)
        self.assertFalse(report.early_release_safe)

    def test_dual_condition_evaluates_once_and_later_partials_do_not_retrigger(self) -> None:
        hybrid = FakeHybridRetriever([])
        subject = benchmark(hybrid)
        mature = retrieval()
        subject.observe_partial(PARTIAL)
        subject.record_mature(PARTIAL, mature, maturity_at=10.0)
        subject.mark_speech_end(speech_end_at=20.0)

        subject.observe_partial("बाद में बदल गया प्रश्न")
        subject.record_mature("बाद में बदल गया प्रश्न", retrieval("999"), maturity_at=21.0)
        subject.mark_speech_end(speech_end_at=22.0)
        report = subject.finalize(mature, final_transcript_at=30.0, current_speech_end_to_answer_ms=304.0)

        self.assertEqual(hybrid.calls, [])
        self.assertEqual(report.latest_partial_at_dual_condition, PARTIAL)
        self.assertTrue(report.early_release_safe)

    def test_material_change_retrieves_once_and_requires_same_evidence_family(self) -> None:
        hybrid = FakeHybridRetriever([retrieval()])
        subject = benchmark(hybrid)
        mature = retrieval()
        subject.observe_partial(PARTIAL)
        subject.record_mature(PARTIAL, mature, maturity_at=10.0)
        subject.observe_partial("फिलाडेल्फिया लैंकेस्टर से कितनी दूर है")

        subject.mark_speech_end(speech_end_at=20.0)
        report = subject.finalize(mature, final_transcript_at=30.0, current_speech_end_to_answer_ms=304.0)

        self.assertEqual(hybrid.calls, ["फिलाडेल्फिया लैंकेस्टर से कितनी दूर है"])
        self.assertTrue(report.speech_end_partial_changed_materially)
        self.assertTrue(report.early_release_safe)

    def test_final_evidence_mismatch_is_a_false_early_release(self) -> None:
        subject = benchmark()
        mature = retrieval()
        subject.observe_partial(PARTIAL)
        subject.record_mature(PARTIAL, mature, maturity_at=10.0)
        subject.mark_speech_end(speech_end_at=20.0)

        report = subject.finalize(
            retrieval("999", provenance=("1", "0")),
            final_transcript_at=30.0,
            current_speech_end_to_answer_ms=304.0,
        )

        self.assertTrue(report.early_release_safe)
        self.assertFalse(report.early_release_matches_final)
        self.assertTrue(report.false_early_release)

    def test_matching_final_evidence_is_safe_and_reports_latency_savings(self) -> None:
        subject = benchmark()
        mature = retrieval()
        subject.observe_partial(PARTIAL)
        subject.record_mature(PARTIAL, mature, maturity_at=10.0)
        subject.mark_speech_end(speech_end_at=20.0)

        report = subject.finalize(mature, final_transcript_at=30.0, current_speech_end_to_answer_ms=304.0)

        self.assertTrue(report.early_release_safe)
        self.assertEqual(report.early_release_candidate_query_id, "232017")
        self.assertEqual(report.final_query_id, "232017")
        self.assertTrue(report.early_release_matches_final)
        self.assertFalse(report.false_early_release)
        self.assertIsNotNone(report.estimated_speech_end_to_answer_ms)
        self.assertIsNotNone(report.latency_saved_ms)

    def test_post_speech_end_recovery_compares_each_evaluated_partial_to_final_evidence(self) -> None:
        subject = benchmark()
        final = retrieval()
        subject.record_post_speech_end_evaluation(
            "pre-speech partial",
            retrieval("111", provenance=("1", "0")),
            decision("111", count=1, ratio=0.2),
            evaluated_at=9.0,
            semantic_concentrated=False,
            hybrid_concentrated=False,
        )
        subject.mark_speech_end(speech_end_at=10.0)
        subject.record_post_speech_end_evaluation(
            "पहला सही partial",
            final,
            decision("232017"),
            evaluated_at=10.100,
            semantic_concentrated=True,
            hybrid_concentrated=True,
        )
        subject.record_post_speech_end_evaluation(
            "दूसरा सही partial",
            final,
            decision("232017"),
            evaluated_at=10.150,
            semantic_concentrated=True,
            hybrid_concentrated=True,
        )
        subject.record_mature(PARTIAL, final, maturity_at=10.300)

        report = subject.finalize(final, final_transcript_at=10.350, current_speech_end_to_answer_ms=400.0)

        self.assertEqual(len(report.post_speech_end_partials), 2)
        first = report.post_speech_end_partials[0]
        self.assertEqual(first["semantic_top1_query_id"], "232017")
        self.assertEqual(first["hybrid_top1_provenance"], ["232017", "8", "0"])
        self.assertTrue(first["final_query_id_present_in_semantic_top_k"])
        self.assertTrue(first["final_query_id_present_in_bm25_top_k"])
        self.assertTrue(first["top1_exact_evidence_matches_final"])
        self.assertTrue(first["final_exact_evidence_present_in_hybrid_top_k"])
        self.assertAlmostEqual(report.post_speech_end_recovery["earliest_final_query_id_recovery_ms"], 100.0)
        self.assertAlmostEqual(report.post_speech_end_recovery["earliest_exact_final_evidence_recovery_ms"], 100.0)
        self.assertAlmostEqual(report.post_speech_end_recovery["first_consecutive_correct_evidence_ms"], 150.0)
        self.assertAlmostEqual(report.post_speech_end_recovery["maturity_reached_ms_after_speech_end"], 300.0)
        self.assertAlmostEqual(report.post_speech_end_recovery["maturity_delay_after_first_correct_evidence_ms"], 200.0)
        self.assertEqual(report.post_speech_end_recovery["conclusion"], "POLICY_LIMITED")

    def test_recovery_is_stt_limited_when_exact_evidence_is_not_available_before_maturity(self) -> None:
        subject = benchmark()
        final = retrieval()
        subject.mark_speech_end(speech_end_at=10.0)
        subject.record_post_speech_end_evaluation(
            "गलत partial",
            retrieval("111", provenance=("1", "0")),
            decision("111", count=1, ratio=0.2),
            evaluated_at=10.100,
            semantic_concentrated=False,
            hybrid_concentrated=False,
        )
        subject.record_mature(PARTIAL, final, maturity_at=10.200)

        report = subject.finalize(final, final_transcript_at=10.250, current_speech_end_to_answer_ms=300.0)

        self.assertIsNone(report.post_speech_end_recovery["earliest_exact_final_evidence_recovery_ms"])
        self.assertEqual(report.post_speech_end_recovery["conclusion"], "STT_LIMITED")

    def test_production_session_has_no_early_release_behavior_without_opt_in(self) -> None:
        """The normal session still has no benchmark report or early response path."""
        session = VoiceRAGSession(FakeHybridRetriever([retrieval()]), AnswerComposer(), "hin_Deva")
        session.mark_speech_end()
        session.handle_final(PARTIAL)

        self.assertIsNone(session.early_release_report())


if __name__ == "__main__":
    unittest.main()
