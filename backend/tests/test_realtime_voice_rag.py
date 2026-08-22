"""Unit tests for persistent hybrid retrieval and trusted realtime voice-RAG candidates."""

import asyncio
import threading
import time
import unittest
from types import SimpleNamespace

from app.rag.generation.answer_composer import AnswerComposer, ComposedAnswer
from app.rag.retrieval.bm25_store import BM25Store, LexicalDocument
from app.rag.retrieval.hybrid_retriever import HybridRetrievedChunk, HybridRetrievalResult, HybridRetriever
from app.rag.retrieval.maturity import MaturityPolicy, assess_retrieval_maturity
from app.rag.retrieval.retriever import RetrievedChunk
from app.services.voice_rag_session import VoiceRAGSession


EVIDENCE_TEXT = "लैंकेस्टर से फिलाडेल्फिया बहुत दूर नहीं है।"


def chunk(rank: int, query_id: str, text: str = EVIDENCE_TEXT, *, fused_score: float = 0.02) -> HybridRetrievedChunk:
    return HybridRetrievedChunk(
        rank=rank,
        text=text,
        metadata={"query_id": query_id, "passage_index": "0", "chunk_index": str(rank - 1), "target_lang": "hin_Deva"},
        semantic_score=0.8,
        lexical_score=2.0,
        fused_score=fused_score,
    )


def retrieval(*fused: HybridRetrievedChunk, semantic: list[HybridRetrievedChunk] | None = None, lexical: list[HybridRetrievedChunk] | None = None) -> HybridRetrievalResult:
    return HybridRetrievalResult(
        semantic=semantic if semantic is not None else list(fused),
        lexical=lexical if lexical is not None else list(fused),
        fused=list(fused),
        semantic_latency_ms=10.0,
        lexical_latency_ms=2.0,
        fusion_latency_ms=0.2,
    )


def mature_retrieval(query_id: str = "232017", *, semantic_score: float = 0.84) -> HybridRetrievalResult:
    """Build same-family evidence that passes the unchanged maturity policy."""
    chunks = [
        HybridRetrievedChunk(
            rank=rank,
            text=EVIDENCE_TEXT,
            metadata={"query_id": query_id, "passage_index": "0", "chunk_index": str(rank - 1), "target_lang": "hin_Deva"},
            semantic_score=semantic_score,
            lexical_score=2.5,
            fused_score=0.032018 - rank / 1_000_000,
        )
        for rank in range(1, 6)
    ]
    return retrieval(*chunks, semantic=list(chunks), lexical=list(chunks))


class FakeHybridRetriever:
    def __init__(self, results: list[HybridRetrievalResult]) -> None:
        self.results = results
        self.calls: list[str] = []

    def retrieve(self, query: str, **_kwargs) -> HybridRetrievalResult:
        self.calls.append(query)
        return self.results.pop(0)


class SlowHybridRetriever:
    """Thread-safe fake used to prove the session owns just one active evaluation."""

    def __init__(self, result: HybridRetrievalResult) -> None:
        self.result = result
        self.calls: list[str] = []
        self.active = 0
        self.maximum_active = 0
        self.started = threading.Event()
        self._lock = threading.Lock()

    def retrieve(self, query: str, *_args, **_kwargs) -> HybridRetrievalResult:
        with self._lock:
            self.calls.append(query)
            self.active += 1
            self.maximum_active = max(self.maximum_active, self.active)
            self.started.set()
        time.sleep(0.03)
        with self._lock:
            self.active -= 1
        return self.result


class CapturingComposer:
    def __init__(self, no_answer: bool = False) -> None:
        self.no_answer = no_answer
        self.calls: list[tuple[str, list[RetrievedChunk]]] = []

    def compose(self, query: str, chunks: list[RetrievedChunk], **_kwargs) -> ComposedAnswer:
        self.calls.append((query, chunks))
        return ComposedAnswer("उत्तर" if not self.no_answer else "कोई उत्तर नहीं", [], None, 0.1, self.no_answer)


class FakeQdrantClient:
    def __init__(self) -> None:
        self.scroll_calls = 0

    def collection_exists(self, _name: str) -> bool:
        return True

    def scroll(self, **_kwargs):
        self.scroll_calls += 1
        return [SimpleNamespace(payload={"text": EVIDENCE_TEXT, "query_id": "232017", "passage_index": 0, "chunk_index": 0, "target_lang": "hin_Deva"})], None


class FakeSemanticRetriever:
    def retrieve(self, _query: str, **_kwargs) -> list[RetrievedChunk]:
        return [RetrievedChunk(0.8, EVIDENCE_TEXT, {"query_id": "232017", "passage_index": 0, "chunk_index": 0, "target_lang": "hin_Deva"})]


class RealtimeVoiceRagTests(unittest.TestCase):
    def test_bm25_builds_once_and_reuses_current_qdrant_corpus(self) -> None:
        client = FakeQdrantClient()
        store = SimpleNamespace(
            client=client,
            collection_name="development",
            collection_exists=lambda: client.collection_exists("development"),
            scroll=client.scroll,
        )
        bm25 = BM25Store.from_vector_store(store, "hin_Deva")
        bm25.search("लैंकेस्टर दूर")
        bm25.search("फिलाडेल्फिया")
        self.assertEqual(client.scroll_calls, 1)

    def test_hybrid_retrieval_preserves_provenance(self) -> None:
        lexical = BM25Store([LexicalDocument(EVIDENCE_TEXT, {"query_id": "232017", "passage_index": 0, "chunk_index": 0, "target_lang": "hin_Deva"})])
        result = HybridRetriever(FakeSemanticRetriever(), lexical).retrieve("लैंकेस्टर दूर", target_lang="hin_Deva")
        self.assertEqual(result.fused[0].provenance, ("232017", "0", "0"))
        self.assertIsNotNone(result.fused[0].semantic_score)
        self.assertIsNotNone(result.fused[0].lexical_score)

    def test_composer_adapter_uses_semantic_score_not_rrf_score(self) -> None:
        hybrid_chunk = chunk(1, "232017", fused_score=0.032018)
        hybrid_chunk = HybridRetrievedChunk(
            rank=hybrid_chunk.rank,
            text=hybrid_chunk.text,
            metadata=hybrid_chunk.metadata,
            semantic_score=0.84,
            lexical_score=2.5,
            fused_score=hybrid_chunk.fused_score,
        )

        composer_chunk = hybrid_chunk.as_retrieved_chunk()

        self.assertIsNotNone(composer_chunk)
        assert composer_chunk is not None
        self.assertEqual(composer_chunk.score, 0.84)
        self.assertEqual(composer_chunk.metadata["semantic_score"], 0.84)
        self.assertEqual(composer_chunk.metadata["bm25_score"], 2.5)
        self.assertEqual(composer_chunk.metadata["fused_score"], 0.032018)

    def test_rrf_rank_is_independent_from_semantic_confidence(self) -> None:
        retriever = HybridRetriever(FakeSemanticRetriever(), BM25Store([]), rrf_k=60)
        shared_metadata = {"query_id": "both", "passage_index": "0", "chunk_index": "0"}
        semantic = [
            chunk(1, "semantic-only", fused_score=0.0),
            HybridRetrievedChunk(2, EVIDENCE_TEXT, shared_metadata, 0.84, None, 0.0),
        ]
        lexical = [HybridRetrievedChunk(1, EVIDENCE_TEXT, shared_metadata, None, 2.5, 0.0)]

        fused = retriever._fuse(semantic, lexical, top_k=2)

        self.assertEqual(fused[0].metadata["query_id"], "both")
        self.assertGreater(fused[0].fused_score, fused[1].fused_score)

    def test_semantically_scored_hybrid_evidence_passes_existing_composer_guardrail(self) -> None:
        hybrid_chunk = HybridRetrievedChunk(
            rank=1,
            text="फिलाडेल्फिया से लैंकेस्टर की दूरी लगभग 110 किलोमीटर है।",
            metadata={"query_id": "232017", "passage_index": "8", "chunk_index": "0"},
            semantic_score=0.84,
            lexical_score=2.5,
            fused_score=0.032018,
        )
        composer_chunk = hybrid_chunk.as_retrieved_chunk()
        assert composer_chunk is not None

        answer = AnswerComposer(min_retrieval_score=0.8).compose(
            "फिलाडेल्फिया लैंकेस्टर से कितनी दूर है?", [composer_chunk]
        )

        self.assertFalse(answer.is_no_answer)
        self.assertEqual(answer.evidence[0].retrieval_score, 0.84)

    def test_bm25_only_evidence_is_not_given_a_semantic_score(self) -> None:
        hybrid_chunk = HybridRetrievedChunk(
            rank=1,
            text=EVIDENCE_TEXT,
            metadata={"query_id": "232017", "passage_index": "8", "chunk_index": "0"},
            semantic_score=None,
            lexical_score=3.2,
            fused_score=0.016393,
        )

        self.assertIsNone(hybrid_chunk.as_retrieved_chunk())
        self.assertIsNone(hybrid_chunk.semantic_score)
        self.assertEqual(hybrid_chunk.bm25_score, 3.2)

    def test_a_like_partial_does_not_trigger_maturity(self) -> None:
        distributed = [chunk(1, "1", "दिल्ली भारत की राजधानी है"), chunk(2, "2", "मुंबई समुद्र के किनारे है")]
        decision = assess_retrieval_maturity("विला डेल सिया लैंक", distributed[:1], distributed[1:], distributed)
        self.assertFalse(decision.mature)

    def test_c_like_partial_triggers_maturity(self) -> None:
        semantic = [chunk(rank, "232017") for rank in range(1, 6)]
        lexical = [chunk(1, "9", "दूसरा दस्तावेज"), chunk(2, "232017")]
        decision = assess_retrieval_maturity("विला डेल सिया लैंकेस्टर से कितनी दूर", semantic, lexical, [chunk(1, "232017")])
        self.assertTrue(decision.mature)

    def test_first_mature_partial_creates_a_trusted_candidate(self) -> None:
        mature = mature_retrieval()
        hybrid = FakeHybridRetriever([mature])
        composer = CapturingComposer()
        session = VoiceRAGSession(hybrid, composer, "hin_Deva")
        self.assertTrue(session.handle_partial("विला डेल सिया लैंकेस्टर से कितनी दूर").mature)
        state = session.trusted_candidate_state()

        self.assertEqual(len(hybrid.calls), 1)
        self.assertTrue(state["trusted_candidate_exists"])
        self.assertEqual(state["trusted_top_query_id"], "232017")
        self.assertEqual(state["trusted_semantic_confidence"], 0.84)
        self.assertNotEqual(state["trusted_semantic_confidence"], mature.fused[0].fused_score)
        self.assertEqual(composer.calls[0][1][0].score, 0.84)

    def test_later_immature_partial_cannot_overwrite_trusted_candidate(self) -> None:
        initial = mature_retrieval("232017")
        weak = retrieval(chunk(1, "999"))
        session = VoiceRAGSession(FakeHybridRetriever([initial, weak]), CapturingComposer(), "hin_Deva")

        session.handle_partial("विला डेल सिया लैंकेस्टर से कितनी दूर")
        self.assertFalse(session.handle_partial("लैंकेस्टर दिल्ली नई जानकारी प्रश्न").mature)

        self.assertEqual(session.trusted_candidate_state()["trusted_top_query_id"], "232017")

    def test_later_mature_same_family_with_stronger_semantic_confidence_updates_candidate(self) -> None:
        initial = mature_retrieval("232017", semantic_score=0.84)
        stronger = mature_retrieval("232017", semantic_score=0.87)
        session = VoiceRAGSession(FakeHybridRetriever([initial, stronger]), CapturingComposer(), "hin_Deva")

        session.handle_partial("विला डेल सिया लैंकेस्टर से कितनी दूर")
        self.assertTrue(session.handle_partial("लैंकेस्टर दूरी मार्ग नया शहर प्रश्न").mature)
        state = session.trusted_candidate_state()

        self.assertEqual(state["trusted_top_query_id"], "232017")
        self.assertEqual(state["trusted_semantic_confidence"], 0.87)
        self.assertEqual(state["trusted_candidate_updates"], 1)

    def test_later_mature_conflicting_query_does_not_overwrite_trusted_candidate(self) -> None:
        initial = mature_retrieval("232017")
        conflict = mature_retrieval("999", semantic_score=0.95)
        session = VoiceRAGSession(FakeHybridRetriever([initial, conflict]), CapturingComposer(), "hin_Deva")

        session.handle_partial("विला डेल सिया लैंकेस्टर से कितनी दूर")
        self.assertTrue(session.handle_partial("लैंकेस्टर नया शहर अलग प्रश्न").mature)
        state = session.trusted_candidate_state()

        self.assertEqual(state["trusted_top_query_id"], "232017")
        self.assertEqual(state["trusted_candidate_conflicts"], 1)

    def test_repeated_identical_partial_does_not_retrigger_retrieval(self) -> None:
        session = VoiceRAGSession(FakeHybridRetriever([mature_retrieval()]), CapturingComposer(), "hin_Deva")

        self.assertIsNotNone(session.handle_partial("विला डेल सिया लैंकेस्टर से कितनी दूर"))
        self.assertIsNone(session.handle_partial("विला डेल सिया लैंकेस्टर से कितनी दूर"))
        self.assertEqual(len(session.hybrid_retriever.calls), 1)

    def test_minor_post_maturity_variations_do_not_trigger_retrieval(self) -> None:
        retriever = FakeHybridRetriever([mature_retrieval()])
        session = VoiceRAGSession(retriever, CapturingComposer(), "hin_Deva")
        trusted_text = "विला डेल सिया लैंकेस्टर से कितनी दूर"

        self.assertTrue(session.handle_partial(trusted_text).mature)
        self.assertIsNone(session.handle_partial(f"{trusted_text}!"))
        self.assertIsNone(session.handle_partial("विला डेल सिया लैंकेस्टर से दूर"))
        self.assertIsNone(session.handle_partial(f"{trusted_text} है"))
        metrics = session.partial_evaluation_metrics()

        self.assertEqual(len(retriever.calls), 1)
        self.assertEqual(metrics["post_maturity_partials_seen"], 3)
        self.assertEqual(metrics["post_maturity_partials_ignored"], 3)
        self.assertEqual(metrics["post_maturity_retrievals_triggered"], 0)

    def test_material_post_maturity_change_triggers_one_additional_retrieval(self) -> None:
        retriever = FakeHybridRetriever([mature_retrieval(), retrieval(chunk(1, "999"))])
        session = VoiceRAGSession(retriever, CapturingComposer(), "hin_Deva")

        session.handle_partial("विला डेल सिया लैंकेस्टर से कितनी दूर")
        self.assertFalse(session.handle_partial("लैंकेस्टर दिल्ली नई जानकारी प्रश्न").mature)
        metrics = session.partial_evaluation_metrics()

        self.assertEqual(len(retriever.calls), 2)
        self.assertEqual(metrics["post_maturity_retrievals_triggered"], 1)
        self.assertEqual(session.trusted_candidate_state()["trusted_top_query_id"], "232017")

    def test_mature_result_reused_when_final_evidence_agrees(self) -> None:
        trusted_chunk = HybridRetrievedChunk(
            rank=1,
            text=EVIDENCE_TEXT,
            metadata={"query_id": "232017", "passage_index": "8", "chunk_index": "0"},
            semantic_score=0.84,
            lexical_score=2.5,
            fused_score=0.032018,
        )
        mature = retrieval(
            trusted_chunk,
            semantic=[chunk(i, "232017") for i in range(1, 6)],
            lexical=[chunk(1, "232017")],
        )
        final = retrieval(trusted_chunk)
        composer = CapturingComposer()
        session = VoiceRAGSession(FakeHybridRetriever([mature, final]), composer, "hin_Deva")
        session.handle_partial("विला डेल सिया लैंकेस्टर से कितनी दूर")
        result = session.handle_final("फिलाडेल्फिया लैंकेस्टर से कितनी दूर है")
        self.assertTrue(result.mature_partial_used)
        self.assertFalse(result.final_rerun_required)
        self.assertEqual(len(composer.calls), 1)
        self.assertEqual(composer.calls[0][1][0].metadata["query_id"], "232017")
        self.assertEqual(composer.calls[0][1][0].score, 0.84)
        self.assertEqual(composer.calls[0][1][0].metadata["fused_score"], 0.032018)

    def test_mature_result_is_discarded_when_final_evidence_differs(self) -> None:
        mature = retrieval(chunk(1, "232017"), semantic=[chunk(i, "232017") for i in range(1, 6)], lexical=[chunk(1, "232017")])
        final = retrieval(chunk(1, "999"))
        composer = CapturingComposer()
        session = VoiceRAGSession(FakeHybridRetriever([mature, final]), composer, "hin_Deva")
        session.handle_partial("विला डेल सिया लैंकेस्टर से कितनी दूर")
        result = session.handle_final("एक अलग अंतिम प्रश्न")
        self.assertFalse(result.mature_partial_used)
        self.assertTrue(result.final_rerun_required)
        self.assertEqual(composer.calls[-1][1][0].metadata["query_id"], "999")

    def test_final_rerun_and_no_answer_paths_preserve_latency_fields(self) -> None:
        final = retrieval(chunk(1, "999", "संदर्भ"))
        composer = CapturingComposer(no_answer=True)
        session = VoiceRAGSession(FakeHybridRetriever([final]), composer, "hin_Deva")
        session.mark_speech_end()
        result = session.handle_final("अंतिम प्रश्न")
        self.assertTrue(result.final_rerun_required)
        self.assertTrue(result.answer.is_no_answer)
        self.assertIsNotNone(result.latency_ms.final_validation)
        self.assertIsNotNone(result.latency_ms.composer)
        self.assertIsNotNone(result.latency_ms.end_of_speech_to_answer)

    def test_policy_keeps_validated_thresholds(self) -> None:
        policy = MaturityPolicy()
        self.assertEqual(policy.semantic_dominant_min_count, 3)
        self.assertEqual(policy.hybrid_dominant_min_count, 3)
        self.assertEqual(policy.dominant_min_ratio, 0.60)
        self.assertEqual(policy.min_sanity_overlap_count, 1)


class RealtimeVoiceRagAsyncTests(unittest.IsolatedAsyncioTestCase):
    async def test_partial_evaluations_are_bounded_to_one_active_worker_and_one_pending_latest(self) -> None:
        slow_retriever = SlowHybridRetriever(mature_retrieval())
        session = VoiceRAGSession(slow_retriever, CapturingComposer(), "hin_Deva")

        self.assertTrue(session.submit_partial("विला डेल सिया लैंकेस्टर से कितनी दूर पहला"))
        await asyncio.to_thread(slow_retriever.started.wait, 1.0)
        self.assertTrue(session.partial_evaluation_active)
        self.assertTrue(session.submit_partial("लैंकेस्टर दिल्ली नई जानकारी दूसरा"))
        self.assertTrue(session.submit_partial("लैंकेस्टर मुंबई नई जानकारी तीसरा"))
        self.assertEqual(session.pending_partial_count, 1)

        await session.wait_for_partial_evaluations()

        self.assertLessEqual(slow_retriever.maximum_active, 1)
        self.assertEqual(len(slow_retriever.calls), 2)
        self.assertIn("तीसरा", slow_retriever.calls[-1])
        self.assertEqual(session.partial_evaluation_metrics()["stale_partial_evaluations_dropped"], 1)

    async def test_partial_submission_does_not_block_stt_reception(self) -> None:
        slow_retriever = SlowHybridRetriever(mature_retrieval())
        session = VoiceRAGSession(slow_retriever, CapturingComposer(), "hin_Deva")

        started_at = time.perf_counter()
        queued = session.submit_partial("विला डेल सिया लैंकेस्टर से कितनी दूर")
        submit_latency_ms = (time.perf_counter() - started_at) * 1_000

        self.assertTrue(queued)
        self.assertFalse(slow_retriever.started.is_set())
        self.assertLess(submit_latency_ms, 10.0)
        await session.wait_for_partial_evaluations()
