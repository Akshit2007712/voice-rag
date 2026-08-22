"""Manually replay a WebM recording through the local realtime voice WebSocket route."""

import argparse
import asyncio
import json
from pathlib import Path

import websockets


def _short(text: object, limit: int = 46) -> str:
    value = str(text).replace("\n", " ")
    return value if len(value) <= limit else f"{value[: limit - 3]}..."


def _print_post_speech_end_recovery(benchmark: dict[str, object]) -> None:
    """Print the opt-in recovery report without changing the WebSocket protocol."""
    observations = benchmark.get("post_speech_end_partials", [])
    recovery = benchmark.get("post_speech_end_recovery", {})
    if not isinstance(observations, list) or not isinstance(recovery, dict):
        return

    print("\nPOST-SPEECH-END EVIDENCE RECOVERY")
    print(
        "OFFSET_MS | PARTIAL | SEMANTIC_TOP1_ID | HYBRID_TOP1_ID | "
        "FINAL_ID_FOUND | EXACT_FINAL_EVIDENCE_FOUND | SEMANTIC_CONCENTRATION | BM25_SUPPORT | MATURE"
    )
    for observation in observations:
        if not isinstance(observation, dict):
            continue
        concentration = (
            f"{observation.get('semantic_dominant_query_id')} "
            f"{observation.get('semantic_dominant_count')}/{observation.get('semantic_dominant_ratio')}"
        )
        print(
            f"{observation.get('offset_after_speech_end_ms')} | "
            f"{_short(observation.get('partial', ''))} | "
            f"{observation.get('semantic_top1_query_id')} | "
            f"{observation.get('hybrid_top1_query_id')} | "
            f"{observation.get('final_query_id_present_in_hybrid_top_k')} | "
            f"{observation.get('final_exact_evidence_present_in_hybrid_top_k')} | "
            f"{concentration} | "
            f"{observation.get('final_query_id_present_in_bm25_top_k')} | "
            f"{observation.get('mature')}"
        )

    print("\nPOST-SPEECH-END RECOVERY SUMMARY")
    for key in (
        "earliest_final_query_id_recovery_ms",
        "earliest_exact_final_evidence_recovery_ms",
        "first_consecutive_correct_evidence_ms",
        "maturity_reached_ms_after_speech_end",
        "transcript_final_ms_after_speech_end",
        "maturity_delay_after_first_correct_evidence_ms",
        "conclusion",
        "conclusion_reason",
    ):
        print(f"{key.upper()}: {recovery.get(key)}")


async def replay(file_path: Path, url: str, chunk_bytes: int, interval_ms: int) -> None:
    """Send a local browser-style WebM stream, then print partial and final server events."""
    async with websockets.connect(url, max_size=2**20) as websocket:
        with file_path.open("rb") as source:
            while chunk := source.read(chunk_bytes):
                await websocket.send(chunk)
                await asyncio.sleep(interval_ms / 1_000)
        await websocket.send(json.dumps({"type": "end"}))
        async for raw_event in websocket:
            print(raw_event)
            try:
                event = json.loads(raw_event)
            except ValueError:
                continue
            if event.get("type") in {"final", "error"}:
                benchmark = event.get("early_release_benchmark")
                if event.get("type") == "final" and isinstance(benchmark, dict):
                    _print_post_speech_end_recovery(benchmark)
                return


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", required=True, type=Path)
    parser.add_argument("--language", default="hi")
    parser.add_argument("--url", default="ws://127.0.0.1:8000/query-voice-stream")
    parser.add_argument("--chunk-bytes", type=int, default=8_000)
    parser.add_argument("--interval-ms", type=int, default=100)
    parser.add_argument(
        "--early-release-benchmark",
        action="store_true",
        help="Request opt-in speech-end instrumentation without changing normal response timing.",
    )
    args = parser.parse_args()
    if not args.file.is_file():
        parser.error(f"File not found: {args.file}")
    if args.chunk_bytes < 1 or args.interval_ms < 0:
        parser.error("--chunk-bytes must be positive and --interval-ms must not be negative")
    separator = "&" if "?" in args.url else "?"
    benchmark_query = "&early_release_benchmark=true" if args.early_release_benchmark else ""
    asyncio.run(
        replay(
            args.file,
            f"{args.url}{separator}language={args.language}{benchmark_query}",
            args.chunk_bytes,
            args.interval_ms,
        )
    )


if __name__ == "__main__":
    main()
