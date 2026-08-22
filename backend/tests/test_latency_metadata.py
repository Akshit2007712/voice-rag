"""Latency-contract coverage for the shared one-shot HTTP RAG path."""

from io import BytesIO
from types import SimpleNamespace
from unittest import IsolatedAsyncioTestCase, TestCase
from unittest.mock import AsyncMock, patch

from starlette.datastructures import UploadFile

from app.rag.generation.answer_composer import ComposedAnswer
from app.routes.voice import query_voice
from app.services.text_rag import RagLatency, TextRagResult, run_text_rag
from app.services.stt import TranscriptionResult


class _Chunk:
    def as_retrieved_chunk(self):
        return SimpleNamespace(score=0.9, text="evidence", metadata={})


class _Hybrid:
    def retrieve(self, *_args, **_kwargs):
        return SimpleNamespace(
            fused=[_Chunk()],
            embedding_latency_ms=700.0,
            qdrant_branch_wall_ms=1_000.0,
            worker_compute_ms=900.0,
            post_embed_parallel_wall_ms=1_100.0,
            fusion_latency_ms=10.0,
        )


def _app() -> SimpleNamespace:
    composer = SimpleNamespace(compose=lambda *_args, **_kwargs: ComposedAnswer("answer", [], 0.9, 0.1, False))
    return SimpleNamespace(
        state=SimpleNamespace(
            rag_runtime=SimpleNamespace(answer_composer=composer),
            hybrid_retrievers={"hin_Deva": _Hybrid(), "eng_Latn": _Hybrid()},
        )
    )


class TextLatencyContractTests(TestCase):
    def test_typed_rag_reports_real_stage_names_and_does_not_sum_parallel_branches(self) -> None:
        # The staged values deliberately overlap: total wall time must not be their sum.
        with patch("app.services.text_rag.time.perf_counter", side_effect=(10.0, 11.81, 11.91, 11.92)):
            result = run_text_rag("question", "en", "text", _app())
        self.assertEqual(result.input_mode, "text")
        self.assertEqual(result.latency.embedding_ms, 700.0)
        self.assertEqual(result.latency.qdrant_ms, 1_000.0)
        self.assertEqual(result.latency.bm25_ms, 900.0)
        self.assertEqual(result.latency.post_embedding_parallel_ms, 1_100.0)
        self.assertEqual(result.latency.rrf_ms, 10.0)
        self.assertEqual(result.latency.maturity_ms, 0.0)
        self.assertAlmostEqual(result.latency.composer_ms, 100.0)
        self.assertAlmostEqual(result.latency.rag_total_ms, 1_920.0)
        # Qdrant and BM25 are concurrent branches. Their sum is not a request
        # wall-clock contract; the explicit post-embedding wall time is.
        self.assertLess(
            result.latency.post_embedding_parallel_ms,
            result.latency.qdrant_ms + result.latency.bm25_ms,
        )
        critical_path_ms = (
            result.latency.embedding_ms
            + result.latency.post_embedding_parallel_ms
            + result.latency.rrf_ms
            + result.latency.maturity_ms
            + result.latency.composer_ms
        )
        self.assertGreaterEqual(result.latency.rag_total_ms, result.latency.embedding_ms)
        self.assertGreaterEqual(result.latency.rag_total_ms, result.latency.post_embedding_parallel_ms)
        self.assertAlmostEqual(result.latency.rag_total_ms, critical_path_ms, delta=20.0)


class VoiceLatencyContractTests(IsolatedAsyncioTestCase):
    async def test_voice_response_keeps_stt_out_of_rag_and_includes_voice_pipeline_total(self) -> None:
        app = _app()
        request = SimpleNamespace(app=app)
        upload = UploadFile(BytesIO(b"audio"), filename="sample.webm", headers={"content-type": "audio/webm"})
        rag_result = TextRagResult(
            query="question",
            answer=ComposedAnswer("answer", [], 0.9, 0.1, False),
            input_mode="voice",
            latency=RagLatency(10.0, 20.0, 15.0, 20.0, 1.0, 0.0, 2.0, 123.0),
        )
        with patch("app.routes.voice.validate_audio_upload", new=AsyncMock(return_value=b"audio")), patch(
            "app.routes.voice.stt_service.transcribe", new=AsyncMock(return_value=TranscriptionResult("question", 0.9, 1))
        ), patch("app.services.rag_harness.run_text_rag", return_value=rag_result), patch(
            # ``time`` is a shared module. In addition to route/STT timing, the
            # deterministic pre/post guardrails each make a local clock pair.
            "app.routes.voice.time.perf_counter", side_effect=(
                9.0, 10.0, 10.4,
                10.4, 10.4, 10.4, 10.4,
                10.7, 10.7,
            )
        ):
            response = await query_voice(request, upload, language="hi")
        self.assertEqual(response.latency.rag_total_ms, 123.0)
        self.assertIsNotNone(response.voice_latency)
        self.assertAlmostEqual(response.voice_latency.stt_ms, 400.0, delta=0.001)
        self.assertAlmostEqual(response.voice_latency.total_voice_pipeline_ms, 700.0, delta=0.001)
        self.assertIsNone(response.benchmark_latency.p50_ms)
        self.assertNotIn("transcript", response.model_dump())
        self.assertNotIn("diagnostics", response.model_dump())
        self.assertNotIn("benchmark_scope", response.model_dump())
        self.assertNotIn("benchmark_environment", response.model_dump())
        snapshot = app.state.diagnostics_registry.snapshot()
        self.assertEqual(snapshot["requests"]["successful"], 1)
        self.assertEqual(snapshot["requests"]["last_input_mode"], "voice")

    async def test_typed_response_has_no_voice_latency(self) -> None:
        from app.routes.voice import query_text
        from app.schemas.response import TextQueryRequest

        rag_result = TextRagResult(
            query="question",
            answer=ComposedAnswer("answer", [], 0.9, 0.1, False),
            input_mode="text",
            latency=RagLatency(1.0, 2.0, 3.0, 4.0, 5.0, 0.0, 6.0, 7.0),
        )
        request = SimpleNamespace(app=_app())
        with patch("app.services.rag_harness.run_text_rag", return_value=rag_result), patch(
            "app.routes.voice.stt_service.transcribe", new=AsyncMock()
        ) as stt:
            response = await query_text(request, TextQueryRequest(query="question", language="en"))
        self.assertIsNone(response.voice_latency)
        self.assertEqual(response.latency.rag_total_ms, 7.0)
        self.assertNotIn("diagnostics", response.model_dump())
        snapshot = request.app.state.diagnostics_registry.snapshot()
        self.assertEqual(snapshot["requests"]["successful"], 1)
        self.assertEqual(snapshot["requests"]["last_language"], "en")
        stt.assert_not_awaited()

    async def test_failed_typed_rag_records_only_a_sanitized_failure_category(self) -> None:
        from app.routes.voice import query_text
        from app.schemas.response import TextQueryRequest

        request = SimpleNamespace(app=_app())
        with patch("app.services.rag_harness.run_text_rag", side_effect=RuntimeError("query text must not be stored")):
            response = await query_text(request, TextQueryRequest(query="private input", language="hi"))
        self.assertEqual(response.status_code, 500)
        self.assertNotIn("query text must not be stored", response.body.decode())
        snapshot = request.app.state.diagnostics_registry.snapshot()
        self.assertEqual(snapshot["requests"]["failed"], 1)
        self.assertEqual(snapshot["requests"]["last_error_category"], "internal_error")
        self.assertNotIn("query text must not be stored", str(snapshot))


if __name__ == "__main__":
    import unittest

    unittest.main()
