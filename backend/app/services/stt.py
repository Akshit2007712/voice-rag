import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Any

import httpx


logger = logging.getLogger("uvicorn.error")
_MAX_LOGGED_RESPONSE_CHARS = 2_000


@dataclass(frozen=True)
class TranscriptionResult:
    transcript: str
    confidence: float
    duration_ms: int


class STTServiceError(Exception):
    """A safe, user-facing error returned by the speech-to-text provider."""

    def __init__(self, message: str, status_code: int = 502) -> None:
        self.status_code = status_code
        super().__init__(message)


def _filename_for_mime_type(mime_type: str) -> str:
    if "wav" in mime_type:
        return "audio.wav"
    if "mp3" in mime_type or "mpeg" in mime_type:
        return "audio.mp3"
    if "ogg" in mime_type:
        return "audio.ogg"
    if "flac" in mime_type:
        return "audio.flac"
    if "m4a" in mime_type or "mp4" in mime_type:
        return "audio.m4a"
    return "audio.webm"


class SarvamSTTService:

    """Sarvam REST speech-to-text adapter for short audio requests."""

    TIMEOUT_SECONDS = 15.0
    MODEL = "saaras:v3"
    MODE = "transcribe"

    async def transcribe(
        self,
        audio_bytes: bytes,
        mime_type: str,
        language_code: str,
    ) -> TranscriptionResult:
        api_key = os.getenv("SARVAM_API_KEY") or "sk_jwd8t1p0_fBNIRgvIPaNa0kYqtRXcl72r"
        api_url = os.getenv("SARVAM_API_URL") or "https://api.sarvam.ai/speech-to-text"


        base_mime_type = (mime_type or "audio/webm").split(";", 1)[0].strip().lower()
        if not base_mime_type:
            base_mime_type = "audio/webm"
        filename = _filename_for_mime_type(base_mime_type)

        headers = {
            "api-subscription-key": api_key,
            "Authorization": f"Bearer {api_key}",
        }
        files = {"file": (filename, audio_bytes, base_mime_type)}

        try:
            async with httpx.AsyncClient(timeout=self.TIMEOUT_SECONDS) as client:
                data = {
                    "model": self.MODEL,
                    "mode": self.MODE,
                    "language_code": language_code,
                }
                response = await client.post(api_url, headers=headers, data=data, files=files)
                if response.is_error:
                    logger.error(
                        "SARVAM STT ERROR\n"
                        "endpoint: %s\n"
                        "status_code: %d\n"
                        "response_body: %s\n"
                        "model: %s\n"
                        "request_config: %s\n"
                        "filename: %s\n"
                        "mime_type: %s\n"
                        "audio_bytes: %d",
                        api_url,
                        response.status_code,
                        _sanitize_response_body(response),
                        data.get("model", "<not sent>"),
                        _sanitize_for_log(data),
                        filename,
                        mime_type,
                        len(audio_bytes),
                    )

        except httpx.TimeoutException as exc:
            logger.error(
                "SARVAM TRANSPORT ERROR\nexception_type: %s\nmessage: %s",
                type(exc).__name__,
                _sanitize_text(str(exc)),
            )
            raise STTServiceError("Sarvam transcription request timed out.", 504) from exc

        except httpx.RequestError as exc:
            logger.error(
                "SARVAM TRANSPORT ERROR\nexception_type: %s\nmessage: %s",
                type(exc).__name__,
                _sanitize_text(str(exc)),
            )
            raise STTServiceError("Sarvam transcription service is unavailable.", 503) from exc

        if response.status_code in {401, 403}:
            raise STTServiceError("Sarvam API key is invalid or unauthorized.", 401)
        if response.status_code in {400, 415, 422}:
            raise STTServiceError("Sarvam does not support this audio file.", 422)
        if response.is_error:
            raise STTServiceError("Sarvam transcription service returned an error.")

        try:
            payload: dict[str, Any] = response.json()
        except ValueError as exc:
            raise STTServiceError("Sarvam returned an invalid transcription response.") from exc
        if not isinstance(payload, dict):
            raise STTServiceError("Sarvam returned an invalid transcription response.")

        transcript = str(payload.get("transcript") or "").strip()
        if not transcript:
            raise STTServiceError("Sarvam returned an empty transcription.")

        return TranscriptionResult(
            transcript=transcript,
            confidence=_confidence_from(payload),
            duration_ms=_duration_ms_from(payload),
        )


def _filename_for_mime_type(mime_type: str) -> str:
    base_mime_type = mime_type.split(";", 1)[0].strip().lower()
    extensions = {
        "audio/webm": ".webm",
        "video/webm": ".webm",
    }
    return f"audio{extensions.get(base_mime_type, '.webm')}"


def _confidence_from(payload: dict[str, Any]) -> float:
    """Use Sarvam's confidence field, falling back to language probability."""
    try:
        confidence = float(payload.get("confidence", payload.get("language_probability", 0.0)))
    except (TypeError, ValueError):
        return 0.0
    return min(max(confidence, 0.0), 1.0)


def _duration_ms_from(payload: dict[str, Any]) -> int:
    """Extract duration when supplied, otherwise derive it from word timestamps."""
    for key in ("duration_ms", "duration"):
        value = payload.get(key)
        if isinstance(value, (int, float)):
            return max(0, int(value if key == "duration_ms" else value * 1000))

    timestamps = payload.get("timestamps")
    if isinstance(timestamps, dict):
        end_times = timestamps.get("end_time_seconds")
        if isinstance(end_times, list) and end_times:
            try:
                return max(0, int(float(end_times[-1]) * 1000))
            except (TypeError, ValueError):
                pass
    return 0


def _sanitize_response_body(response: httpx.Response) -> str:
    """Log a bounded response body while redacting secret-like JSON fields."""
    try:
        return json.dumps(_sanitize_for_log(response.json()), ensure_ascii=False)[:_MAX_LOGGED_RESPONSE_CHARS]
    except ValueError:
        return _sanitize_text(response.text)[:_MAX_LOGGED_RESPONSE_CHARS]


def _sanitize_for_log(value: Any) -> Any:
    """Recursively redact keys that could contain credentials before development logging."""
    if isinstance(value, dict):
        return {
            key: "[REDACTED]" if any(secret in key.lower() for secret in ("api_key", "authorization", "token", "secret"))
            else _sanitize_for_log(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_sanitize_for_log(item) for item in value]
    return value


def _sanitize_text(value: str) -> str:
    """Redact token-like values from non-JSON provider text before logging it."""
    value = re.sub(r"(?i)(authorization\s*[:=]\s*)([^\s,;]+)", r"\1[REDACTED]", value)
    value = re.sub(r"(?i)((?:api[_ -]?key|token|secret)\s*[:=]\s*)([^\s,;]+)", r"\1[REDACTED]", value)
    return value
