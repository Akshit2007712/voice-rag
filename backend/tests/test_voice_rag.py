"""Route-level tests for STT to persistent deterministic-RAG integration."""

from io import BytesIO
from types import SimpleNamespace
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, patch

from starlette.datastructures import UploadFile

from app.rag.generation.answer_composer import AnswerEvidence, ComposedAnswer, NO_ANSWER_TEXT
from app.routes.voice import query_voice
from app.services.stt import STTServiceError, TranscriptionResult


class FakeRuntime:
    def __init__(self, answer: ComposedAnswer) -> None:
        self.answer_result = answer
        self.calls = 0
        self.last_args = ()
        self.last_kwargs = {}
        self.answer_composer = SimpleNamespace(compose=self.compose)

    def compose(self, *_args, **_kwargs) -> ComposedAnswer:
        self.calls += 1
        self.last_args = _args
        self.last_kwargs = _kwargs
        return self.answer_result


class FakeChunk:
    def as_retrieved_chunk(self):
        return SimpleNamespace(score=0.9, text="evidence", metadata={})


class FakeHybridRetriever:
    def __init__(self) -> None:
        self.calls = []

    def retrieve(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return SimpleNamespace(fused=[FakeChunk()])


def request_with_runtime(runtime: FakeRuntime) -> SimpleNamespace:
    return SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(
        rag_runtime=runtime,
        hybrid_retrievers={"hin_Deva": FakeHybridRetriever(), "eng_Latn": FakeHybridRetriever()},
    )))


class VoiceRagRouteTests(IsolatedAsyncioTestCase):
    async def test_webm_reuses_the_same_runtime_and_return_evidence(self) -> None:
        composed = ComposedAnswer(
            text="उत्तर प्रमाण से लिया गया है।",
            evidence=[AnswerEvidence(1, 2, 0, 0.9, "उत्तर प्रमाण से लिया गया है।")],
            confidence=0.9,
            latency_ms=1.0,
            is_no_answer=False,
        )
        runtime = FakeRuntime(composed)
        stt = AsyncMock(return_value=TranscriptionResult("प्रश्न", 0.9, 10))
        audio_bytes = b"\x1a\x45\xdf\xa3\x00"
        with patch("app.routes.voice.validate_audio_upload", new=AsyncMock(return_value=audio_bytes)), patch(
            "app.routes.voice.stt_service.transcribe", stt
        ):
            response = None
            for _ in range(2):
                upload = UploadFile(
                    BytesIO(audio_bytes),
                    filename="sample.webm",
                    headers={"content-type": "audio/webm;codecs=opus"},
                )
                response = await query_voice(request_with_runtime(runtime), upload, language="hi")
        self.assertIsNotNone(response)
        self.assertEqual(response.answer, composed.text)
        self.assertFalse(response.no_answer)
        self.assertEqual(response.evidence[0].query_id, 1)
        self.assertGreaterEqual(response.latency.rag_total_ms, 0)
        self.assertIsNotNone(response.voice_latency)
        self.assertGreaterEqual(response.voice_latency.stt_ms, 0)
        self.assertEqual(stt.await_count, 2)
        self.assertEqual(stt.await_args.args, (audio_bytes, "audio/webm;codecs=opus", "hi-IN"))
        self.assertEqual(runtime.calls, 2)
        self.assertEqual(runtime.last_kwargs["max_sentences"], 3)

    async def test_empty_transcript_is_rejected(self) -> None:
        runtime = FakeRuntime(ComposedAnswer("x", [], 0.0, 0.0, False))
        upload = UploadFile(BytesIO(b"audio"), filename="sample.webm", headers={"content-type": "audio/webm"})
        with patch("app.routes.voice.validate_audio_upload", new=AsyncMock(return_value=b"audio")), patch(
            "app.routes.voice.stt_service.transcribe",
            AsyncMock(return_value=TranscriptionResult("   ", 0.0, 0)),
        ):
            response = await query_voice(request_with_runtime(runtime), upload, language="hi")
        self.assertEqual(response.status_code, 422)
        self.assertIn(b'"code":"INVALID_REQUEST"', response.body)

    async def test_video_webm_mime_is_forwarded_to_stt_unchanged(self) -> None:
        runtime = FakeRuntime(ComposedAnswer("उत्तर", [], 0.9, 0.0, False))
        audio_bytes = b"\x1a\x45\xdf\xa3\x00"
        upload = UploadFile(
            BytesIO(audio_bytes),
            filename="sample.webm",
            headers={"content-type": "video/webm;codecs=opus"},
        )
        stt = AsyncMock(return_value=TranscriptionResult("प्रश्न", 0.9, 10))
        with patch("app.routes.voice.validate_audio_upload", new=AsyncMock(return_value=audio_bytes)), patch(
            "app.routes.voice.stt_service.transcribe", stt
        ):
            await query_voice(request_with_runtime(runtime), upload, language="hi")
        stt.assert_awaited_once_with(audio_bytes, "video/webm;codecs=opus", "hi-IN")

    async def test_stt_error_propagates_to_existing_handler(self) -> None:
        runtime = FakeRuntime(ComposedAnswer("x", [], 0.0, 0.0, False))
        upload = UploadFile(BytesIO(b"audio"), filename="sample.webm", headers={"content-type": "audio/webm"})
        with patch("app.routes.voice.validate_audio_upload", new=AsyncMock(return_value=b"audio")), patch(
            "app.routes.voice.stt_service.transcribe", AsyncMock(side_effect=STTServiceError("provider failed", 503)),
        ):
            response = await query_voice(request_with_runtime(runtime), upload, language="hi")
        self.assertEqual(response.status_code, 503)
        self.assertIn(b'"code":"STT_UNAVAILABLE"', response.body)

    async def test_no_answer_is_returned_without_evidence(self) -> None:
        runtime = FakeRuntime(ComposedAnswer(NO_ANSWER_TEXT, [], None, 0.1, True))
        upload = UploadFile(BytesIO(b"audio"), filename="sample.webm", headers={"content-type": "audio/webm"})
        with patch("app.routes.voice.validate_audio_upload", new=AsyncMock(return_value=b"audio")), patch(
            "app.routes.voice.stt_service.transcribe", AsyncMock(return_value=TranscriptionResult("प्रश्न", 0.9, 1)),
        ):
            response = await query_voice(request_with_runtime(runtime), upload, language="hi")
        self.assertTrue(response.no_answer)
        self.assertEqual(response.answer, NO_ANSWER_TEXT)
        self.assertEqual(response.evidence, [])

    async def test_unsupported_application_language_is_rejected_before_stt(self) -> None:
        runtime = FakeRuntime(ComposedAnswer("x", [], 0.0, 0.0, False))
        upload = UploadFile(BytesIO(b"audio"), filename="sample.webm", headers={"content-type": "audio/webm"})
        stt = AsyncMock()
        with patch("app.routes.voice.validate_audio_upload", new=AsyncMock(return_value=b"audio")), patch(
            "app.routes.voice.stt_service.transcribe", stt
        ):
            response = await query_voice(request_with_runtime(runtime), upload, language="ta")
        self.assertEqual(response.status_code, 422)
        self.assertIn(b'"code":"INVALID_REQUEST"', response.body)
        stt.assert_not_awaited()
