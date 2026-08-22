"""Handshake construction tests for the Sarvam realtime WebSocket adapter."""

import json
import unittest
from urllib.parse import parse_qs, urlsplit

from app.services.realtime_stt import (
    SarvamRealtimeConnection,
    SarvamRealtimeSTTService,
    _parse_event,
    _realtime_connection_url,
)


class SarvamRealtimeHandshakeTests(unittest.TestCase):
    def test_required_realtime_settings_are_in_the_handshake_query(self) -> None:
        url = _realtime_connection_url("wss://api.sarvam.ai/speech-to-text-realtime/ws", "hi-IN")
        parameters = parse_qs(urlsplit(url).query)
        self.assertEqual(parameters["language_code"], ["hi-IN"])
        self.assertEqual(parameters["model"], [SarvamRealtimeSTTService.MODEL])
        self.assertEqual(parameters["mode"], ["transcribe"])
        self.assertEqual(parameters["encoding"], ["linear16"])
        self.assertEqual(parameters["sample_rate"], ["16000"])
        self.assertEqual(parameters["channels"], ["1"])

    def test_existing_non_secret_query_parameters_are_preserved(self) -> None:
        url = _realtime_connection_url("wss://example.test/realtime?region=in", "hi-IN")
        parameters = parse_qs(urlsplit(url).query)
        self.assertEqual(parameters["region"], ["in"])
        self.assertEqual(parameters["language_code"], ["hi-IN"])

    def test_empty_language_code_is_rejected_before_connecting(self) -> None:
        with self.assertRaisesRegex(ValueError, "language_code"):
            _realtime_connection_url("wss://example.test/realtime", " ")

    def test_documented_final_event_is_classified_as_final(self) -> None:
        event = _parse_event('{"event": "transcript.final", "text": "अंतिम"}')
        self.assertIsNotNone(event)
        self.assertEqual(event.kind, "final")
        self.assertEqual(event.transcript, "अंतिम")

    def test_documented_session_end_event_is_classified_separately(self) -> None:
        event = _parse_event('{"event": "session.end", "request_id": "request-1"}')
        self.assertIsNotNone(event)
        self.assertEqual(event.kind, "session_end")
        self.assertEqual(event.request_id, "request-1")


class _FakeSocket:
    def __init__(self, messages: list[str]) -> None:
        self._messages = messages
        self.sent: list[str | bytes] = []

    def __aiter__(self):
        return self

    async def __anext__(self) -> str:
        if not self._messages:
            raise StopAsyncIteration
        return self._messages.pop(0)

    async def send(self, message: str | bytes) -> None:
        self.sent.append(message)

    async def close(self) -> None:
        return None


class SarvamRealtimeSessionEndTests(unittest.IsolatedAsyncioTestCase):
    async def test_session_end_commits_last_non_empty_partial(self) -> None:
        socket = _FakeSocket(
            [
                '{"event": "transcript.partial", "text": "पूरा वाक्य"}',
                '{"event": "session.end", "request_id": "request-1"}',
            ]
        )
        connection = SarvamRealtimeConnection(socket)

        await connection.finish_audio()
        partial_event = await connection.next_event()
        final_event = await connection.next_event()

        self.assertEqual(json.loads(socket.sent[0]), {"event": "end"})
        self.assertEqual(partial_event.kind, "partial")
        self.assertEqual(final_event.kind, "final")
        self.assertEqual(final_event.transcript, "पूरा वाक्य")
        self.assertEqual(final_event.request_id, "request-1")
        await connection.close()

    async def test_session_end_without_transcript_is_a_clear_error(self) -> None:
        socket = _FakeSocket(['{"event": "session.end"}'])
        connection = SarvamRealtimeConnection(socket)

        await connection.finish_audio()
        event = await connection.next_event()

        self.assertEqual(event.kind, "error")
        self.assertIn("without a transcript", event.transcript or "")
        await connection.close()
