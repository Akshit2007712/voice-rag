"""Send one local browser WebM recording directly to Sarvam STT, without FastAPI or RAG."""

import argparse
import asyncio
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.rag.language_config import get_sarvam_language_code  # noqa: E402
from app.services.stt import SarvamSTTService, STTServiceError  # noqa: E402
from app.services.validation import is_webm_container  # noqa: E402


load_dotenv(BACKEND_ROOT / ".env")


async def run(file_path: Path, language: str) -> None:
    """Validate one WebM file and send it through the production STT adapter."""
    audio_bytes = file_path.read_bytes()
    provider_language_code = get_sarvam_language_code(language)
    is_valid_webm = is_webm_container(audio_bytes)

    print(f"API KEY PRESENT: {str(bool(os.getenv('SARVAM_API_KEY'))).lower()}")
    print(f"APPLICATION LANGUAGE: {language}")
    print(f"SARVAM LANGUAGE CODE: {provider_language_code}")
    print(f"SOURCE FILE: {file_path}")
    print("FILENAME: audio.webm")
    print("MIME TYPE: audio/webm")
    print(f"AUDIO BYTES: {len(audio_bytes)}")
    print(f"WEBM VALIDATION RESULT: {str(is_valid_webm).lower()}")

    if not is_valid_webm:
        raise ValueError("The supplied file does not begin with the WebM EBML signature")

    started_at = time.perf_counter()
    result = await SarvamSTTService().transcribe(audio_bytes, "audio/webm", provider_language_code)
    latency_ms = (time.perf_counter() - started_at) * 1_000
    print("STATUS: success")
    print(f"TRANSCRIPT: {result.transcript}")
    print(f"CONFIDENCE: {result.confidence}")
    print(f"DURATION_MS: {result.duration_ms}")
    print(f"REQUEST LATENCY_MS: {latency_ms:.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", required=True, type=Path, help="Path to a local browser WebM recording")
    parser.add_argument("--language", default="hi", help="Application language code (default: hi)")
    args = parser.parse_args()
    if not args.file.is_file():
        parser.error(f"WebM file not found: {args.file}")
    if args.file.suffix.lower() != ".webm":
        parser.error("This isolated diagnostic accepts a .webm file only")
    try:
        asyncio.run(run(args.file, args.language))
    except (OSError, STTServiceError, ValueError) as error:
        print(f"STATUS: failed")
        print(f"ERROR: {type(error).__name__}: {error}")
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
