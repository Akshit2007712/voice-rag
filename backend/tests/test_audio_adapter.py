"""Windows-safe FFmpeg adapter tests using mocked long-lived Popen processes."""

import asyncio
import queue
import threading
from io import BytesIO
from unittest import IsolatedAsyncioTestCase
from unittest.mock import patch

from app.services.audio_adapter import AudioConversionError, WebMToPcm16Adapter


class FakePopen:
    """Minimal binary-pipe process double for adapter lifecycle tests."""

    def __init__(self, stdout: bytes = b"", stderr: bytes = b"") -> None:
        self.pid = 1234
        self.stdin = BytesIO()
        self.stdout = BytesIO(stdout)
        self.stderr = BytesIO(stderr)
        self.returncode: int | None = None
        self.terminated = False
        self.killed = False

    def poll(self) -> int | None:
        return self.returncode

    def wait(self) -> int:
        if self.returncode is None:
            self.returncode = 0
        return self.returncode

    def terminate(self) -> None:
        self.terminated = True
        self.returncode = -15

    def kill(self) -> None:
        self.killed = True
        self.returncode = -9


class BlockingPipe:
    """A pipe whose next read blocks until the fake process produces data or EOF."""

    def __init__(self) -> None:
        self._chunks: queue.Queue[bytes | None] = queue.Queue()

    def feed(self, data: bytes) -> None:
        self._chunks.put(data)

    def end(self) -> None:
        self._chunks.put(None)

    def read(self, _size: int) -> bytes:
        item = self._chunks.get()
        return b"" if item is None else item


class ClosingInput(BytesIO):
    """Emit final decoded PCM only when FFmpeg stdin receives EOF."""

    def __init__(self, on_close) -> None:
        super().__init__()
        self._on_close = on_close
        self._closed_once = False

    def close(self) -> None:
        if not self._closed_once:
            self._closed_once = True
            self._on_close()
        super().close()


class FinalFlushPopen(FakePopen):
    """Simulate FFmpeg buffering final PCM until the route closes stdin."""

    def __init__(self) -> None:
        super().__init__()
        self.stdout = BlockingPipe()
        self.stdin = ClosingInput(self._flush_final_pcm)

    def _flush_final_pcm(self) -> None:
        self.stdout.feed(b"final-pcm")
        self.stdout.end()


class HungPopen(FakePopen):
    """Wait blocks until the adapter's fallback kill path releases the process."""

    def __init__(self) -> None:
        super().__init__()
        self._release_wait = threading.Event()

    def wait(self) -> int:
        self._release_wait.wait(timeout=1)
        return self.returncode if self.returncode is not None else 0

    def kill(self) -> None:
        super().kill()
        self._release_wait.set()


class WebMToPcm16AdapterTests(IsolatedAsyncioTestCase):
    async def test_start_uses_one_unbuffered_popen_process_and_bridged_pipe_io(self) -> None:
        process = FakePopen(stdout=b"\x01\x02\x03\x04")
        with patch("app.services.audio_adapter.subprocess.Popen", return_value=process) as popen:
            adapter = WebMToPcm16Adapter("fake-ffmpeg")
            await adapter.start()
            await adapter.write_chunk(b"webm-chunk")
            pcm = await adapter.read_pcm(4)
            self.assertEqual(process.stdin.getvalue(), b"webm-chunk")
            self.assertEqual(pcm, b"\x01\x02\x03\x04")
            self.assertEqual(popen.call_count, 1)
            self.assertEqual(popen.call_args.kwargs["bufsize"], 0)
            self.assertEqual(await adapter.read_pcm(4), b"")
            await adapter.finish()

    async def test_finish_drains_final_pcm_through_one_stdout_consumer_before_exit(self) -> None:
        process = FinalFlushPopen()
        with patch("app.services.audio_adapter.subprocess.Popen", return_value=process):
            adapter = WebMToPcm16Adapter()
            await adapter.start()
            drained: list[bytes] = []

            async def consume_pcm() -> None:
                async for pcm in adapter.pcm_chunks():
                    drained.append(pcm)

            consumer = asyncio.create_task(consume_pcm())
            await asyncio.sleep(0)
            await adapter.finish()
            await consumer
        self.assertEqual(drained, [b"final-pcm"])
        self.assertEqual(process.returncode, 0)

    async def test_stdout_rejects_a_second_competing_consumer(self) -> None:
        process = FakePopen(stdout=b"")
        with patch("app.services.audio_adapter.subprocess.Popen", return_value=process):
            adapter = WebMToPcm16Adapter()
            await adapter.start()
            self.assertEqual(await adapter.read_pcm(), b"")
            with self.assertRaisesRegex(AudioConversionError, "already has a consumer"):
                await asyncio.create_task(adapter.read_pcm())
            await adapter.finish()

    async def test_unexpected_ffmpeg_exit_includes_stderr_without_audio_data(self) -> None:
        process = FakePopen(stderr=b"invalid WebM stream")
        process.returncode = 1
        with patch("app.services.audio_adapter.subprocess.Popen", return_value=process):
            adapter = WebMToPcm16Adapter()
            await adapter.start()
            with self.assertRaisesRegex(AudioConversionError, "invalid WebM stream"):
                await adapter.write_chunk(b"ignored")

    async def test_hung_ffmpeg_is_killed_only_after_stdout_eof(self) -> None:
        process = HungPopen()
        with patch("app.services.audio_adapter.subprocess.Popen", return_value=process):
            adapter = WebMToPcm16Adapter()
            adapter.SHUTDOWN_TIMEOUT_SECONDS = 0.01
            await adapter.start()
            self.assertEqual(await adapter.read_pcm(), b"")
            with self.assertRaisesRegex(AudioConversionError, "did not finish converting"):
                await adapter.finish()
        self.assertTrue(process.killed)

    async def test_close_terminates_a_live_process(self) -> None:
        process = FakePopen()
        with patch("app.services.audio_adapter.subprocess.Popen", return_value=process):
            adapter = WebMToPcm16Adapter()
            await adapter.start()
            await adapter.close()
        self.assertTrue(process.terminated)
