"""Sarvam realtime STT WebSocket adapter for 16 kHz mono linear16 PCM sessions."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import websockets


logger = logging.getLogger("uvicorn.error")

DEFAULT_REALTIME_URL = "wss://api.sarvam.ai/speech-to-text-realtime/ws"


class RealtimeSTTError(RuntimeError):
    """Safe failure for a single realtime STT session."""


@dataclass(frozen=True)
class RealtimeSTTEvent:
    """Provider-neutral partial/final/VAD event emitted by the Sarvam connection."""

    kind: str
    transcript: str | None = None
    request_id: str | None = None
    received_at: float | None = None


class SarvamRealtimeConnection:
    """One connected Sarvam realtime session; no reconnect or audio replay is attempted."""

    def __init__(self, socket: Any) -> None:
        self._socket = socket
        self._events: asyncio.Queue[RealtimeSTTEvent] = asyncio.Queue()
        self._end_of_audio_sent = False
        self._last_partial: RealtimeSTTEvent | None = None
        self._receiver_task = asyncio.create_task(self._receive_events())

    async def send_audio(self, pcm_linear16: bytes) -> None:
        """Forward raw 16 kHz mono signed-linear PCM without changing it."""
        if not pcm_linear16:
            return
        if len(pcm_linear16) % 2:
            raise RealtimeSTTError("Sarvam realtime PCM must contain 16-bit samples.")
        await self._socket.send(pcm_linear16)

    async def finish_audio(self) -> None:
        """Send Sarvam's documented graceful end event after the final PCM bytes."""
        self._end_of_audio_sent = True
        await self._socket.send(json.dumps({"event": "end"}))

    async def next_event(self) -> RealtimeSTTEvent:
        """Wait for the next normalized upstream event."""
        return await self._events.get()

    async def close(self) -> None:
        """Close the receiver and WebSocket after the voice session ends."""
        self._receiver_task.cancel()
        try:
            await self._receiver_task
        except asyncio.CancelledError:
            pass
        await self._socket.close()

    async def _receive_events(self) -> None:
        try:
            async for raw_message in self._socket:
                if self._end_of_audio_sent:
                    _log_inbound_event_after_end_of_audio(raw_message)
                event = _parse_event(raw_message)
                if event is None:
                    continue
                if event.kind == "partial" and event.transcript:
                    self._last_partial = event
                if event.kind == "session_end":
                    await self._handle_session_end(event)
                    continue
                await self._events.put(event)
        except Exception as exc:
            await self._events.put(RealtimeSTTEvent("error", transcript=str(exc), received_at=time.perf_counter()))

    async def _handle_session_end(self, event: RealtimeSTTEvent) -> None:
        """Finish on Sarvam's terminal session event when no separate final was sent.

        Sarvam documents ``session.end`` as the server's terminal usage/session event.
        A documented ``transcript.final`` still wins when it arrives. If the server
        has already ended the session after a non-empty partial but without a final,
        no later transcript can arrive; commit the last server-produced text instead
        of leaving the route waiting indefinitely.
        """
        if self._last_partial is None or not self._last_partial.transcript:
            await self._events.put(
                RealtimeSTTEvent(
                    "error",
                    transcript="Sarvam ended the realtime session without a transcript.",
                    request_id=event.request_id,
                    received_at=event.received_at,
                )
            )
            return
        logger.warning(
            "SARVAM REALTIME: session.end received after end-of-audio; "
            "committing the last non-empty partial transcript"
        )
        await self._events.put(
            RealtimeSTTEvent(
                "final",
                transcript=self._last_partial.transcript,
                request_id=self._last_partial.request_id or event.request_id,
                received_at=event.received_at,
            )
        )


class SarvamRealtimeSTTService:
    """Connect to the validated Saaras realtime endpoint for Hindi PCM streaming."""

    MODEL = "saaras:v3-realtime"
    MODE = "transcribe"
    ENCODING = "linear16"
    SAMPLE_RATE = 16_000

    async def connect(self, language_code: str) -> SarvamRealtimeConnection:
        """Open one authenticated session and configure server-side VAD/fast streaming."""
        api_key = os.getenv("SARVAM_API_KEY")
        if not api_key:
            raise RealtimeSTTError("Sarvam API key is not configured.")
        url = _realtime_connection_url(os.getenv("SARVAM_REALTIME_URL", DEFAULT_REALTIME_URL), language_code)
        headers = {"API-SUBSCRIPTION-KEY": api_key}
        logger.warning(
            "SARVAM REALTIME URL PARAMETERS:\n"
            "language_code=%s\nmodel=%s\nencoding=%s\nsample_rate=%s",
            language_code,
            self.MODEL,
            self.ENCODING,
            self.SAMPLE_RATE,
        )
        try:
            socket = await websockets.connect(url, additional_headers=headers)
            await socket.send(
                json.dumps(
                    {
                        "type": "session.start",
                        "model": self.MODEL,
                        "language_code": language_code,
                        "mode": self.MODE,
                        "encoding": self.ENCODING,
                        "sample_rate": self.SAMPLE_RATE,
                        "channels": 1,
                        "stream_type": "fast",
                        "endpointing": "vad",
                    }
                )
            )
            return SarvamRealtimeConnection(socket)
        except Exception as exc:
            logger.error(
                "SARVAM REALTIME CONNECT ERROR\nexception_type: %s\nmessage: %s",
                type(exc).__name__,
                str(exc),
            )
            raise RealtimeSTTError("Sarvam realtime transcription service is unavailable.") from exc


def _realtime_connection_url(base_url: str, language_code: str) -> str:
    """Attach required Sarvam realtime handshake parameters without exposing auth data."""
    if not isinstance(language_code, str) or not language_code.strip():
        raise ValueError("language_code must be non-empty")
    parsed = urlsplit(base_url)
    parameters = dict(parse_qsl(parsed.query, keep_blank_values=True))
    parameters.update(
        {
            "language_code": language_code.strip(),
            "model": SarvamRealtimeSTTService.MODEL,
            "mode": SarvamRealtimeSTTService.MODE,
            "encoding": SarvamRealtimeSTTService.ENCODING,
            "sample_rate": str(SarvamRealtimeSTTService.SAMPLE_RATE),
            "channels": "1",
            "stream_type": "fast",
            "endpointing": "vad",
        }
    )
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, urlencode(parameters), parsed.fragment))


def _parse_event(raw_message: str | bytes) -> RealtimeSTTEvent | None:
    """Normalize documented and compatible transcript/VAD payload variants."""
    if isinstance(raw_message, bytes):
        raw_message = raw_message.decode("utf-8", errors="replace")
    try:
        payload = json.loads(raw_message)
    except (TypeError, ValueError):
        return RealtimeSTTEvent("error", transcript="Sarvam sent an invalid realtime event.", received_at=time.perf_counter())
    if not isinstance(payload, dict):
        return None
    data = payload.get("data") if isinstance(payload.get("data"), dict) else payload
    event_type = str(
        payload.get("event") or payload.get("type") or data.get("event") or data.get("type") or ""
    ).lower()
    transcript = str(data.get("transcript") or data.get("text") or "").strip() or None
    request_id = data.get("request_id") or payload.get("request_id")
    if event_type in {"vad.speech_start", "speech_start", "start_speech"}:
        return RealtimeSTTEvent("speech_start", request_id=str(request_id) if request_id else None, received_at=time.perf_counter())
    if event_type in {"vad.speech_end", "speech_end", "end_speech"}:
        return RealtimeSTTEvent("speech_end", request_id=str(request_id) if request_id else None, received_at=time.perf_counter())
    if event_type == "session.end":
        return RealtimeSTTEvent("session_end", request_id=str(request_id) if request_id else None, received_at=time.perf_counter())
    if event_type in {"error", "session.error"}:
        return RealtimeSTTEvent("error", transcript=transcript or "Sarvam realtime error.", received_at=time.perf_counter())
    if transcript:
        finalized = bool(data.get("finalized") or data.get("is_final") or event_type in {"final", "final_transcript", "transcript.final"})
        return RealtimeSTTEvent("final" if finalized else "partial", transcript, str(request_id) if request_id else None, time.perf_counter())
    return None


def _log_inbound_event_after_end_of_audio(raw_message: str | bytes) -> None:
    """Log only safe event-shape diagnostics while finalization is in progress."""
    if isinstance(raw_message, bytes):
        raw_message = raw_message.decode("utf-8", errors="replace")
    try:
        payload = json.loads(raw_message)
    except (TypeError, ValueError):
        logger.warning("SARVAM REALTIME INBOUND AFTER END type=<invalid-json> top_level_keys=[]")
        return
    if not isinstance(payload, dict):
        logger.warning("SARVAM REALTIME INBOUND AFTER END type=<non-object> top_level_keys=[]")
        return
    nested_data = payload.get("data")
    event_type = payload.get("event") or payload.get("type")
    if not event_type and isinstance(nested_data, dict):
        event_type = nested_data.get("event") or nested_data.get("type")
    logger.warning(
        "SARVAM REALTIME INBOUND AFTER END type=%s top_level_keys=%s",
        str(event_type or "<missing>"),
        sorted(str(key) for key in payload),
    )
