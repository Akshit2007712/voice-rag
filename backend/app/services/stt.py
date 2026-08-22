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
    """Map a MIME type to an appropriate audio filename for the Sarvam multipart upload."""
    base = (mime_type or "audio/webm").split(";", 1)[0].strip().lower()
    mapping = {
        "audio/wav": "audio.wav",
        "audio/x-wav": "audio.wav",
        "audio/wave": "audio.wav",
        "audio/mp3": "audio.mp3",
        "audio/mpeg": "audio.mp3",
        "audio/ogg": "audio.ogg",
        "audio/flac": "audio.flac",
        "audio/m4a": "audio.m4a",
        "audio/mp4": "audio.m4a",
        "audio/webm": "audio.webm",
        "video/webm": "audio.webm",
    }
    return mapping.get(base, "audio.webm")


_STT_CLIENT: httpx.AsyncClient | None = None


def _get_stt_client() -> httpx.AsyncClient:
    global _STT_CLIENT
    if _STT_CLIENT is None or _STT_CLIENT.is_closed:
        _STT_CLIENT = httpx.AsyncClient(
            timeout=httpx.Timeout(25.0, connect=5.0),
            limits=httpx.Limits(max_keepalive_connections=20, max_connections=50, keepalive_expiry=60.0),
        )
    return _STT_CLIENT


class SarvamSTTService:
    """Sarvam REST speech-to-text adapter for short audio requests."""

    TIMEOUT_SECONDS = 25.0
    MODEL = "saarika:v2.5"
    MODE = "transcribe"

    async def transcribe(
        self,
        audio_bytes: bytes,
        mime_type: str,
        language_code: str,
    ) -> TranscriptionResult:
        api_key = os.getenv("SARVAM_API_KEY", "").strip()
        if not api_key:
            # Fallback key (for testing only — replace in Render env vars!)
            api_key = "sk_jwd8t1p0_fBNIRgvIPaNa0kYqtRXcl72r"

        api_url = os.getenv("SARVAM_API_URL", "https://api.sarvam.ai/speech-to-text").strip()

        base_mime_type = (mime_type or "audio/webm").split(";", 1)[0].strip().lower() or "audio/webm"
        filename = _filename_for_mime_type(base_mime_type)

        print(
            f"[STT] transcribe called | bytes={len(audio_bytes)} | mime={base_mime_type} | "
            f"file={filename} | lang={language_code} | api_url={api_url}",
            flush=True,
        )

        headers = {
            "api-subscription-key": api_key,
        }
        files = {"file": (filename, audio_bytes, base_mime_type)}
        data = {
            "model": self.MODEL,
            "mode": self.MODE,
            "language_code": language_code,
        }

        try:
            client = _get_stt_client()
            response = await client.post(
                api_url,
                headers=headers,
                data=data,
                files=files,
            )
            print(
                f"[STT] Sarvam response | status={response.status_code} | "
                f"body={response.text[:800]}",
                flush=True,
            )

        except httpx.TimeoutException as exc:
            logger.error(
                "SARVAM TRANSPORT TIMEOUT | %s: %s",
                type(exc).__name__,
                _sanitize_text(str(exc)),
            )
            raise STTServiceError("Voice transcription timed out — please try again.", 504) from exc

        except httpx.RequestError as exc:
            logger.error(
                "SARVAM REQUEST ERROR | %s: %s",
                type(exc).__name__,
                _sanitize_text(str(exc)),
            )
            raise STTServiceError(
                f"Cannot reach transcription service ({type(exc).__name__}). Check network.", 503
            ) from exc

        if response.status_code in {401, 403}:
            raise STTServiceError(
                f"Sarvam API key rejected (HTTP {response.status_code}). Set SARVAM_API_KEY in Render env vars.",
                401,
            )
        if response.status_code in {400, 415, 422}:
            body_preview = response.text[:300]
            raise STTServiceError(
                f"Sarvam rejected this audio file (HTTP {response.status_code}): {body_preview}",
                422,
            )
        if response.is_error:
            body_preview = response.text[:300]
            raise STTServiceError(
                f"Sarvam returned HTTP {response.status_code}: {body_preview}",
                502,
            )

        try:
            payload: dict[str, Any] = response.json()
        except ValueError as exc:
            raise STTServiceError("Sarvam returned an invalid (non-JSON) transcription response.") from exc

        if not isinstance(payload, dict):
            raise STTServiceError("Sarvam returned an unexpected transcription response format.")

        transcript = str(payload.get("transcript") or "").strip()
        print(f"[STT] Transcript extracted | len={len(transcript)} | preview={transcript[:100]!r}", flush=True)
        return TranscriptionResult(
            transcript=transcript,
            confidence=_confidence_from(payload) if transcript else 0.0,
            duration_ms=_duration_ms_from(payload) if transcript else 0,
        )


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
            key: "[REDACTED]" if any(
                secret in key.lower() for secret in ("api_key", "authorization", "token", "secret")
            )
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
