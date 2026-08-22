import asyncio
import json
import logging
import time

from fastapi import APIRouter, File, Form, Query, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

from app.rag.language_config import get_qdrant_target_lang, get_sarvam_language_code
from app.schemas.response import (
    BenchmarkLatencyResponse,
    EvidenceResponse,
    RagLatencyResponse,
    TextQueryRequest,
    VoicePipelineLatencyResponse,
    VoiceRagResponse,
)
from app.schemas.errors import HarnessErrorResponse, UserError
from app.services.benchmark_latency import pending_benchmark_latency
from app.services.diagnostics import get_diagnostics_registry, new_request_id
from app.services.audio_adapter import AudioConversionError, WebMToPcm16Adapter
from app.services.early_release_benchmark import EarlyReleaseBenchmark
from app.services.realtime_stt import RealtimeSTTError, SarvamRealtimeSTTService
from app.services.stt import SarvamSTTService
from app.services.validation import validate_audio_upload
from app.services.voice_rag_session import VoiceRAGSession
from app.services.rag_harness import HarnessFailure, HarnessSuccess, RAGRequestContext, get_rag_harness


router = APIRouter(tags=["voice"])
stt_service = SarvamSTTService()
realtime_stt_service = SarvamRealtimeSTTService()
logger = logging.getLogger("uvicorn.error")

_ERROR_RESPONSES = {
    413: {"model": HarnessErrorResponse, "description": "Audio upload exceeds the supported size."},
    415: {"model": HarnessErrorResponse, "description": "Unsupported audio container or MIME type."},
    422: {"model": HarnessErrorResponse, "description": "Invalid input, unsupported language, or guardrail rejection."},
    500: {"model": HarnessErrorResponse, "description": "Unexpected internal processing failure."},
    502: {"model": HarnessErrorResponse, "description": "Voice transcription provider failure."},
    503: {"model": HarnessErrorResponse, "description": "A required answer dependency is temporarily unavailable."},
    504: {"model": HarnessErrorResponse, "description": "The request timed out."},
}


@router.post(
    "/query-voice",
    response_model=VoiceRagResponse,
    summary="Answer a one-shot browser WebM voice query",
    description=(
        "Upload one browser-recorded WebM file. Supported base MIME types are "
        "audio/webm and video/webm (optional codecs parameters are accepted); the file must be under 10 MB. "
        "`voice_latency.stt_ms` measures transcription only, while `latency.rag_total_ms` excludes STT."
    ),
    responses=_ERROR_RESPONSES,
)
async def query_voice(
    request: Request,
    audio: UploadFile = File(
        ...,
        description="Required WebM microphone file. Use multipart field name `audio`.",
    ),
    language: str = Form("hi", description="Optional application language: `hi` (default) or `en`."),
) -> VoiceRagResponse | JSONResponse:
    """Transcribe browser WebM audio then answer using persistent deterministic RAG."""
    context = RAGRequestContext(new_request_id(), "voice", language, time.perf_counter())
    harness = get_rag_harness(request.app)
    try:
        audio_bytes = await validate_audio_upload(audio)
    except Exception as error:
        return _harness_response(harness.fail(context, request.app, error), request.app)
    try:
        sarvam_language_code = get_sarvam_language_code(language)
    except ValueError as error:
        return _harness_response(harness.fail(context, request.app, error), request.app)

    # Voice-pipeline timing starts with the STT call and excludes upload validation.
    voice_pipeline_started_at = time.perf_counter()
    context = RAGRequestContext(context.request_id, "voice", language, voice_pipeline_started_at)
    stt_started_at = voice_pipeline_started_at
    print("\n" + "=" * 60, flush=True)
    print(f">> [VOICE INPUT RECEIVED] Size: {len(audio_bytes)} bytes | Lang: {language}", flush=True)
    try:
        result = await stt_service.transcribe(audio_bytes, audio.content_type or "audio/webm", sarvam_language_code)
    except Exception as error:
        print(f">> [SARVAM STT ERROR]: {error}", flush=True)
        print("=" * 60 + "\n", flush=True)
        return _harness_response(harness.fail(context, request.app, error), request.app)
    stt_latency_ms = (time.perf_counter() - stt_started_at) * 1_000
    transcript = result.transcript.strip()
    print(f">> [SARVAM TRANSCRIPTION]: '{transcript}' (Confidence: {result.confidence:.2f}, Time: {stt_latency_ms:.1f}ms)", flush=True)
    if not transcript:
        print(">> [NO SPEECH DETECTED IN AUDIO]", flush=True)
        print("=" * 60 + "\n", flush=True)
        fallback_msg = "कोई आवाज़ नहीं मिली। कृपया अपने माइक में स्पष्ट रूप से बोलें।" if language == "hi" else "No speech detected. Please speak clearly into your microphone and try again."
        return VoiceRagResponse(
            transcript="(No speech detected)",
            answer=fallback_msg,
            no_answer=True,
            explanation=None,
            citations=[],
            latency={
                "stt_ms": stt_latency_ms,
                "retrieval_ms": 0.0,
                "generation_ms": 0.0,
                "guardrail_ms": 0.0,
                "guardrails": 0.0,
                "embedding": 0.0,
                "embedding_ms": 0.0,
                "dense": 0.0,
                "qdrant_ms": 0.0,
                "bm25": 0.0,
                "bm25_ms": 0.0,
                "fusion": 0.0,
                "rrf_ms": 0.0,
                "rerank": 0.0,
                "grounding": 0.0,
                "maturity_ms": 0.0,
                "composer_ms": 0.0,
                "rag_total_ms": 0.0,
                "total_voice_pipeline_ms": stt_latency_ms,
                "total_ms": stt_latency_ms,
            },
            voice_latency={
                "stt_ms": stt_latency_ms,
                "total_voice_pipeline_ms": stt_latency_ms,
            },
            guardrail={"triggered": True, "category": "empty_stt", "reason": "No speech detected in audio."},
            benchmark_latency={"p50_ms": None, "p70_ms": None, "p100_ms": None, "sample_count": None},
            input_mode="voice",
        )

    return _harness_response(harness.execute(context.with_query(transcript, stt_latency_ms), request.app), request.app)


@router.post(
    "/query-text",
    response_model=VoiceRagResponse,
    summary="Answer a typed Hindi or English query",
    description=(
        "Typed input bypasses STT and enters the same guarded bilingual RAG path as one-shot voice input. "
        "`latency.rag_total_ms` covers only text RAG; `voice_latency` is null."
    ),
    responses=_ERROR_RESPONSES,
)
async def query_text(request: Request, payload: TextQueryRequest) -> VoiceRagResponse | JSONResponse:
    """Answer a typed Hindi or English query through the same one-shot hybrid path."""
    print("\n" + "=" * 60, flush=True)
    print(f">> [TEXT INPUT RECEIVED]: '{payload.query}' (Language: {payload.language})", flush=True)
    context = RAGRequestContext(new_request_id(), "text", payload.language, time.perf_counter(), payload.query)
    return _harness_response(get_rag_harness(request.app).execute(context, request.app), request.app)


def _harness_response(outcome: HarnessSuccess | HarnessFailure, app) -> VoiceRagResponse | JSONResponse:
    """Adapt a typed harness outcome to either approved product response shape."""
    if isinstance(outcome, HarnessFailure):
        print(f">> [RAG HARNESS FAILURE]: {outcome.message} (Code: {outcome.error_code})", flush=True)
        print("=" * 60 + "\n", flush=True)
        return JSONResponse(
            status_code=outcome.status_code,
            content={
                "error": {
                    "code": outcome.error_code,
                    "message": outcome.message,
                },
                "request_id": outcome.request_id,
            },
        )
    return _one_shot_response(outcome, app)


def _one_shot_response(outcome: HarnessSuccess, app) -> VoiceRagResponse:
    """Map shared text-RAG output to the common voice/text response contract."""
    rag_result = outcome.result
    answer = rag_result.answer
    print(f">> [RAG RETRIEVAL]: {len(answer.evidence)} evidence passages retrieved", flush=True)
    print(f">> [GENERATED ANSWER]: {answer.text}", flush=True)
    timing_str = f">> [TIMINGS]: RAG = {rag_result.latency.rag_total_ms:.1f}ms"
    if outcome.context.stt_latency_ms is not None:
        timing_str += f" | STT = {outcome.context.stt_latency_ms:.1f}ms | Total Voice Pipeline = {outcome.total_voice_pipeline_ms:.1f}ms"
    print(timing_str, flush=True)
    print("=" * 60 + "\n", flush=True)

    return VoiceRagResponse(
        transcript=outcome.context.query_text,
        answer=answer.text,
        no_answer=answer.is_no_answer,
        evidence=[EvidenceResponse(query_id=item.query_id, passage_index=item.passage_index, chunk_index=item.chunk_index, retrieval_score=item.retrieval_score) for item in answer.evidence],
        latency=RagLatencyResponse(**rag_result.latency.__dict__),
        voice_latency=(
            VoicePipelineLatencyResponse(
                stt_ms=outcome.context.stt_latency_ms,
                total_voice_pipeline_ms=outcome.total_voice_pipeline_ms,
            )
            if outcome.context.stt_latency_ms is not None and outcome.total_voice_pipeline_ms is not None
            else None
        ),
        benchmark_latency=BenchmarkLatencyResponse(**_public_benchmark_latency(_benchmark_latency(app))),
        input_mode=rag_result.input_mode,
    )


def _benchmark_latency(app):
    """Use startup-loaded validated metadata; tests get an explicit pending value."""
    return getattr(app.state, "benchmark_latency", pending_benchmark_latency())


def _public_benchmark_latency(benchmark_latency) -> dict[str, object]:
    """Keep internal benchmark scope/environment exclusively in operator diagnostics."""
    benchmark = benchmark_latency or pending_benchmark_latency()
    return {
        "p50_ms": benchmark.p50_ms,
        "p70_ms": benchmark.p70_ms,
        "p100_ms": benchmark.p100_ms,
        "sample_count": benchmark.sample_count,
    }


def _record_stream_failure(app, request_id: str, language: str, error: BaseException) -> None:
    """Keep streaming lifecycle accounting separate from one-shot harness ownership."""
    get_diagnostics_registry(app).record_failure(request_id, "voice_stream", language, error)


@router.websocket("/query-voice-stream")
async def query_voice_stream(
    websocket: WebSocket,
    language: str = Query("hi", description="Application language: `hi` or `en`."),
    early_release_benchmark: bool = Query(False, description="Development-only early-release benchmark metadata."),
) -> None:
    """Run one WebM-browser-audio session through PCM Sarvam realtime STT and hybrid RAG.

    Client protocol: send binary WebM/Opus chunks after connecting, then send the
    JSON control message ``{"type": "end"}``. Partial events are sent back as
    JSON; the final event includes the grounded answer and provenance.
    """
    await websocket.accept()
    logger.warning("REALTIME VOICE: WebSocket accepted")
    request_id = new_request_id()
    stage = "websocket accepted"
    closed_normally = False
    chunk_number = 0
    connection = None
    adapter = WebMToPcm16Adapter()
    pcm_task: asyncio.Task[None] | None = None
    upstream_task: asyncio.Task[None] | None = None
    try:
        try:
            sarvam_language_code = get_sarvam_language_code(language)
            target_lang = get_qdrant_target_lang(language)
        except ValueError:
            _record_stream_failure(websocket.app, request_id, language, ValueError("unsupported language"))
            await _send_stream_error(websocket, "Unsupported application language.")
            return
        runtime = getattr(websocket.app.state, "rag_runtime", None)
        hybrid_retrievers = getattr(websocket.app.state, "hybrid_retrievers", {})
        hybrid_retriever = hybrid_retrievers.get(target_lang)
        if runtime is None or hybrid_retriever is None:
            _record_stream_failure(websocket.app, request_id, language, RuntimeError("runtime unavailable"))
            await _send_stream_error(websocket, "Realtime retrieval runtime is unavailable.")
            return

        benchmark = (
            EarlyReleaseBenchmark(
                hybrid_retriever,
                runtime.answer_composer,
                target_lang,
                top_k=5,
                started_at=time.perf_counter(),
            )
            if early_release_benchmark
            else None
        )
        session = VoiceRAGSession(
            hybrid_retriever,
            runtime.answer_composer,
            target_lang,
            early_release_benchmark=benchmark,
        )
        stage = "session created"
        logger.warning("REALTIME VOICE: session created")
        stage = "FFmpeg process start"
        await adapter.start()
        stage = "Sarvam realtime connection"
        connection = await realtime_stt_service.connect(sarvam_language_code)
        session.mark_connection_ready()
        pcm_task = asyncio.create_task(_forward_pcm_to_sarvam(adapter, connection))
        upstream_task = asyncio.create_task(
            _forward_sarvam_events(
                websocket,
                connection,
                session,
                request_id=request_id,
                language=language,
                include_benchmark=early_release_benchmark,
            )
        )

        while True:
            stage = "waiting for browser audio"
            message = await _receive_or_background_failure(websocket, pcm_task, upstream_task)
            if message is None:
                closed_normally = True
                return
            if message["type"] == "websocket.disconnect":
                logger.warning("REALTIME VOICE: client WebSocket disconnected")
                return
            binary_chunk = message.get("bytes")
            if binary_chunk is not None:
                chunk_number += 1
                stage = "binary WebM chunk received"
                logger.warning(
                    "REALTIME VOICE: binary WebM chunk received number=%d bytes=%d",
                    chunk_number,
                    len(binary_chunk),
                )
                stage = "chunk forwarding to audio adapter"
                await adapter.write_webm(binary_chunk)
                logger.warning("REALTIME VOICE: chunk forwarded to audio adapter number=%d", chunk_number)
                continue
            text = message.get("text")
            if text is None:
                continue
            try:
                control = json.loads(text)
            except ValueError:
                await _send_stream_error(websocket, "Streaming control messages must be JSON.")
                return
            if control.get("type") != "end":
                await _send_stream_error(websocket, "Unsupported streaming control message.")
                return
            stage = "end event received"
            logger.warning("REALTIME VOICE: end event received")
            stage = "FFmpeg end-of-utterance flush"
            await adapter.finish()
            await pcm_task
            stage = "Sarvam end-of-audio signal"
            logger.warning("REALTIME VOICE: Sarvam audio finalization started")
            await connection.finish_audio()
            await asyncio.wait_for(upstream_task, timeout=20)
            closed_normally = True
            return
    except (AudioConversionError, RealtimeSTTError, asyncio.TimeoutError) as error:
        _record_stream_failure(websocket.app, request_id, language, error)
        _log_realtime_error(stage, error)
        await _send_stream_error(websocket, str(error))
        await _close_stream_with_server_error(websocket)
    except WebSocketDisconnect:
        logger.warning("REALTIME VOICE: client WebSocket disconnected during stage=%s", stage)
    except Exception as error:
        _record_stream_failure(websocket.app, request_id, language, error)
        _log_realtime_error(stage, error)
        await _send_stream_error(websocket, "Realtime voice query failed.")
        await _close_stream_with_server_error(websocket)
    finally:
        if pcm_task is not None and not pcm_task.done():
            pcm_task.cancel()
        if upstream_task is not None and not upstream_task.done():
            upstream_task.cancel()
        try:
            await adapter.close()
        except Exception as error:
            _log_realtime_error("FFmpeg cleanup", error)
        if connection is not None:
            await connection.close()
        if closed_normally:
            logger.warning("REALTIME VOICE: WebSocket closed normally")


async def _forward_pcm_to_sarvam(adapter: WebMToPcm16Adapter, connection) -> None:
    """Forward the adapter's incremental 16 kHz mono linear16 output to Sarvam."""
    try:
        async for pcm_chunk in adapter.pcm_chunks():
            logger.warning("REALTIME VOICE: PCM bytes produced bytes=%d", len(pcm_chunk))
            await connection.send_audio(pcm_chunk)
            logger.warning("REALTIME VOICE: PCM forwarded to Sarvam realtime WebSocket bytes=%d", len(pcm_chunk))
    except Exception as error:
        _log_realtime_error("PCM forwarding to Sarvam", error)
        raise


async def _forward_sarvam_events(
    websocket: WebSocket,
    connection,
    session: VoiceRAGSession,
    request_id: str,
    language: str,
    include_benchmark: bool = False,
) -> None:
    """Process STT events while partial retrieval runs through the session's bounded worker."""
    try:
        while True:
            event = await connection.next_event()
            if event.kind == "speech_end":
                logger.warning("REALTIME VOICE: Sarvam end-of-speech event received")
                session.mark_speech_end()
                continue
            if event.kind == "error":
                raise RealtimeSTTError(event.transcript or "Sarvam realtime transcription failed.")
            if event.kind == "partial" and event.transcript:
                logger.warning("REALTIME VOICE: Sarvam partial received chars=%d", len(event.transcript))
                queued = session.submit_partial(event.transcript)
                logger.warning(
                    "REALTIME VOICE: partial evaluation queued=%s active=%s pending=%d",
                    queued,
                    session.partial_evaluation_active,
                    session.pending_partial_count,
                )
                await websocket.send_json(
                    {
                        "type": "partial",
                        "transcript": event.transcript,
                        "mature": False,
                    }
                )
                continue
            if event.kind == "final" and event.transcript:
                logger.warning("REALTIME VOICE: Sarvam final transcript received chars=%d", len(event.transcript))
                await session.wait_for_partial_evaluations()
                result = session.handle_final(event.transcript)
                logger.warning("REALTIME VOICE: AnswerComposer finished no_answer=%s", result.answer.is_no_answer)
                _record_stream_success(websocket.app, request_id, language, result)
                response = {
                        "type": "final",
                        "transcript": result.transcript,
                        "answer": result.answer.text,
                        "no_answer": result.answer.is_no_answer,
                        "evidence": [
                            {
                                "query_id": chunk.metadata.get("query_id"),
                                "passage_index": chunk.metadata.get("passage_index"),
                                "chunk_index": chunk.metadata.get("chunk_index"),
                                "retrieval_score": chunk.semantic_score,
                                "semantic_score": chunk.semantic_score,
                                "bm25_score": chunk.bm25_score,
                                "fused_score": chunk.fused_score,
                            }
                            for chunk in result.evidence
                        ],
                        "latency_ms": {
                            "maturity_at": result.latency_ms.maturity_at,
                            "hybrid_retrieval": result.latency_ms.hybrid_retrieval,
                            "final_stt": result.latency_ms.final_stt,
                            "final_validation": result.latency_ms.final_validation,
                            "composer": result.latency_ms.composer,
                            "end_of_speech_to_answer": result.latency_ms.end_of_speech_to_answer,
                        },
                        "speculation": {
                            "mature_partial_used": result.mature_partial_used,
                            "final_rerun_required": result.final_rerun_required,
                        },
                    }
                if include_benchmark:
                    benchmark_report = session.early_release_report()
                    response["early_release_benchmark"] = benchmark_report.as_dict() if benchmark_report else None
                await websocket.send_json(response)
                logger.warning("REALTIME VOICE: final JSON sent")
                return
    except Exception as error:
        _log_realtime_error("Sarvam event forwarding", error)
        raise


def _record_stream_success(app, request_id: str, language: str, result) -> None:
    """Store only available realtime timing values; never transcript or evidence."""
    latency = {
        "embedding_ms": None,
        "qdrant_ms": None,
        "bm25_ms": None,
        "post_embedding_parallel_ms": None,
        "rrf_ms": None,
        "maturity_ms": result.latency_ms.maturity_decision,
        "composer_ms": result.latency_ms.composer,
        "rag_total_ms": result.latency_ms.final_validation,
        "stt_ms": None,
        "total_voice_pipeline_ms": result.latency_ms.end_of_speech_to_answer,
    }
    get_diagnostics_registry(app).record_success(request_id, "voice_stream", language, latency)


async def _send_stream_error(websocket: WebSocket, detail: str) -> None:
    """Send a safe WebSocket error without provider secrets or transport internals."""
    try:
        await websocket.send_json({"type": "error", "detail": detail})
    except RuntimeError:
        pass


async def _receive_or_background_failure(
    websocket: WebSocket,
    pcm_task: asyncio.Task[None],
    upstream_task: asyncio.Task[None],
) -> dict | None:
    """Wait for browser input while surfacing audio/STT task failures immediately."""
    receive_task = asyncio.create_task(websocket.receive())
    done, _ = await asyncio.wait({receive_task, pcm_task, upstream_task}, return_when=asyncio.FIRST_COMPLETED)
    if receive_task in done:
        return receive_task.result()
    receive_task.cancel()
    await asyncio.gather(receive_task, return_exceptions=True)
    if upstream_task in done:
        exception = upstream_task.exception()
        if exception is not None:
            raise exception
        return None
    exception = pcm_task.exception()
    if exception is not None:
        raise exception
    raise AudioConversionError("PCM conversion ended before the browser stream completed.")


async def _close_stream_with_server_error(websocket: WebSocket) -> None:
    """Give clients a close frame instead of dropping the underlying TCP connection."""
    try:
        await websocket.close(code=1011, reason="Realtime voice processing failed.")
    except RuntimeError:
        pass


def _log_realtime_error(stage: str, error: BaseException) -> None:
    """Emit content-free realtime failure telemetry without exception details."""
    logger.error(
        "REALTIME VOICE ERROR\nstage: %s\nexception_type: %s",
        stage,
        type(error).__name__,
    )
