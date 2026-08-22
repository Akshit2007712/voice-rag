"""Replaceable browser-audio to Sarvam-PCM conversion boundary."""

from __future__ import annotations

import asyncio
import logging
import subprocess
from collections.abc import AsyncIterator
from typing import BinaryIO


logger = logging.getLogger("uvicorn.error")


class AudioConversionError(RuntimeError):
    """Raised when browser WebM/Opus cannot be converted to realtime PCM."""


class WebMToPcm16Adapter:
    """Run one Windows-safe FFmpeg process per voice session.

    Blocking pipe operations are always delegated to worker threads, so the
    FastAPI event loop remains available to receive browser audio and Sarvam
    events while FFmpeg is producing PCM.
    """

    SHUTDOWN_TIMEOUT_SECONDS = 5

    def __init__(self, ffmpeg_command: str = "ffmpeg") -> None:
        self.ffmpeg_command = ffmpeg_command
        self._process: subprocess.Popen[bytes] | None = None
        self._stderr_task: asyncio.Task[None] | None = None
        self._stderr_chunks: list[bytes] = []
        self._stdout_reader: asyncio.Task[object] | None = None
        self._stdout_eof = asyncio.Event()
        self._stdin_eof_sent = False
        self._stdout_eof_logged = False

    async def start(self) -> None:
        """Start a persistent WebM/Opus-to-PCM FFmpeg process for this session."""
        arguments = [
            self.ffmpeg_command,
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            "pipe:0",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-f",
            "s16le",
            "pipe:1",
        ]
        try:
            self._process = await asyncio.to_thread(
                subprocess.Popen,
                arguments,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0,
            )
        except FileNotFoundError as exc:
            raise AudioConversionError("FFmpeg is required to convert browser WebM/Opus audio to PCM.") from exc
        logger.warning("REALTIME VOICE: FFmpeg process started pid=%s", self._process.pid)
        self._stderr_task = asyncio.create_task(self._collect_stderr())

    async def write_chunk(self, audio_chunk: bytes) -> None:
        """Write one browser WebM chunk into the same persistent FFmpeg stdin pipe."""
        if not audio_chunk:
            return
        process = self._require_process()
        if process.poll() is not None:
            await self._raise_unexpected_exit("FFmpeg exited before the browser audio stream finished.")
        if process.stdin is None:
            raise AudioConversionError("FFmpeg stdin is unavailable.")
        try:
            await asyncio.to_thread(_write_and_flush, process.stdin, audio_chunk)
        except (BrokenPipeError, ConnectionResetError, OSError) as exc:
            await self._raise_unexpected_exit("FFmpeg stopped accepting browser audio.", exc)
        logger.warning("REALTIME VOICE: bytes written to FFmpeg stdin=%d", len(audio_chunk))

    async def read_pcm(self, read_size: int = 3_200) -> bytes:
        """Read available 16 kHz mono linear16 PCM without blocking the event loop."""
        if read_size < 1:
            raise ValueError("read_size must be at least 1")
        process = self._require_process()
        if process.stdout is None:
            raise AudioConversionError("FFmpeg stdout is unavailable.")
        self._claim_stdout_reader()
        chunk = await asyncio.to_thread(_read_available, process.stdout, read_size)
        if chunk and self._stdin_eof_sent:
            logger.warning("REALTIME VOICE: FFmpeg final PCM bytes=%d", len(chunk))
        if not chunk:
            self._stdout_eof.set()
            if not self._stdout_eof_logged:
                logger.warning("REALTIME VOICE: FFmpeg stdout EOF")
                self._stdout_eof_logged = True
        if not chunk and process.poll() not in (None, 0):
            await self._raise_unexpected_exit("FFmpeg exited while producing PCM.")
        return chunk

    async def close_input(self) -> None:
        """Close FFmpeg stdin at end-of-utterance so it flushes remaining PCM."""
        process = self._process
        if process is not None and process.stdin is not None and not process.stdin.closed:
            # Set this before closing so a concurrently unblocked stdout reader
            # classifies the very first flushed bytes as final PCM.
            self._stdin_eof_sent = True
            logger.warning("REALTIME VOICE: FFmpeg stdin EOF sent")
            await asyncio.to_thread(process.stdin.close)

    async def finish(self) -> None:
        """End input, let the existing PCM reader drain stdout, then await process exit."""
        process = self._require_process()
        await self.close_input()
        if self._stdout_reader is None:
            raise AudioConversionError("FFmpeg stdout consumer was not started before finish.")
        logger.warning("REALTIME VOICE: FFmpeg final PCM flush started")
        try:
            await asyncio.wait_for(self._stdout_eof.wait(), timeout=self.SHUTDOWN_TIMEOUT_SECONDS)
        except asyncio.TimeoutError as exc:
            logger.error("REALTIME VOICE: FFmpeg stdout did not reach EOF before shutdown timeout")
            if process.poll() is None:
                process.terminate()
                await asyncio.to_thread(process.wait)
            await self._await_stderr_reader()
            raise AudioConversionError("FFmpeg did not flush PCM stdout before shutdown timeout.") from exc
        await self._wait_for_exit(process, terminate_on_timeout=True)
        if process.returncode not in (0, None):
            await self._raise_unexpected_exit("FFmpeg audio conversion failed.")

    async def close(self) -> None:
        """Release FFmpeg during normal route cleanup or after an interrupted session."""
        process = self._process
        if process is None:
            return
        await self.close_input()
        if process.poll() is None:
            process.terminate()
            try:
                await self._wait_for_exit(process, terminate_on_timeout=True)
            except AudioConversionError:
                pass
        await self._await_stderr_reader()

    async def write_webm(self, audio_chunk: bytes) -> None:
        """Compatibility name used by the existing streaming route."""
        await self.write_chunk(audio_chunk)

    async def pcm_chunks(self, read_size: int = 3_200) -> AsyncIterator[bytes]:
        """Yield continuous PCM from the one persistent FFmpeg process."""
        self._claim_stdout_reader()
        while True:
            chunk = await self.read_pcm(read_size)
            if not chunk:
                return
            yield chunk

    def _require_process(self) -> subprocess.Popen[bytes]:
        if self._process is None:
            raise AudioConversionError("Audio converter is not running.")
        return self._process

    def _claim_stdout_reader(self) -> None:
        """Enforce one stdout consumer so final PCM cannot be split or lost."""
        current_task = asyncio.current_task()
        if current_task is None:
            raise AudioConversionError("FFmpeg stdout must be read from an asyncio task.")
        if self._stdout_reader is None:
            self._stdout_reader = current_task
            return
        if self._stdout_reader is not current_task:
            raise AudioConversionError("FFmpeg stdout already has a consumer for this voice session.")

    async def _wait_for_exit(self, process: subprocess.Popen[bytes], terminate_on_timeout: bool) -> None:
        try:
            await asyncio.wait_for(asyncio.to_thread(process.wait), timeout=self.SHUTDOWN_TIMEOUT_SECONDS)
        except asyncio.TimeoutError as exc:
            if terminate_on_timeout and process.poll() is None:
                process.kill()
                await asyncio.to_thread(process.wait)
            raise AudioConversionError("FFmpeg did not finish converting the audio stream.") from exc
        finally:
            await self._await_stderr_reader()
        logger.warning("REALTIME VOICE: FFmpeg exited returncode=%s", process.returncode)

    async def _collect_stderr(self) -> None:
        """Drain stderr continuously so FFmpeg cannot block on a full error pipe."""
        process = self._process
        if process is None or process.stderr is None:
            return
        while True:
            chunk = await asyncio.to_thread(_read_available, process.stderr, 1_024)
            if not chunk:
                return
            if sum(len(item) for item in self._stderr_chunks) < 4_096:
                self._stderr_chunks.append(chunk)

    async def _await_stderr_reader(self) -> None:
        if self._stderr_task is None:
            return
        await self._stderr_task
        self._stderr_task = None

    async def _raise_unexpected_exit(self, message: str, cause: BaseException | None = None) -> None:
        await self._await_stderr_reader()
        process = self._process
        stderr = b"".join(self._stderr_chunks).decode(errors="replace")[:1_000]
        logger.error(
            "REALTIME VOICE: FFmpeg exited unexpectedly returncode=%s stderr=%s",
            process.returncode if process is not None else None,
            stderr or "<empty>",
        )
        error = AudioConversionError(f"{message} {stderr or 'FFmpeg produced no stderr output.'}")
        if cause is not None:
            raise error from cause
        raise error


def _write_and_flush(stream: BinaryIO, data: bytes) -> None:
    """Perform one blocking pipe write away from the asyncio event loop."""
    stream.write(data)
    stream.flush()


def _read_available(stream: BinaryIO, size: int) -> bytes:
    """Read an available pipe chunk, preferring read1 to avoid waiting for a full buffer."""
    read_one = getattr(stream, "read1", None)
    return read_one(size) if callable(read_one) else stream.read(size)
