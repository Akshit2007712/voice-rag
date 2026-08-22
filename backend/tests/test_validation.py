"""Focused validation tests for the WebM-first voice upload boundary."""

from io import BytesIO
from unittest import IsolatedAsyncioTestCase

from fastapi import HTTPException
from starlette.datastructures import UploadFile

from app.services.validation import WEBM_EBML_SIGNATURE, validate_audio_upload


def upload(filename: str, content_type: str, contents: bytes) -> UploadFile:
    """Build an in-memory multipart upload for validation tests."""
    return UploadFile(BytesIO(contents), filename=filename, headers={"content-type": content_type})


class WebMUploadValidationTests(IsolatedAsyncioTestCase):
    async def test_webm_mime_aliases_are_accepted(self) -> None:
        contents = WEBM_EBML_SIGNATURE + b"browser-audio"
        for content_type in (
            "audio/webm",
            "audio/webm;codecs=opus",
            "video/webm",
            "video/webm;codecs=opus",
        ):
            with self.subTest(content_type=content_type):
                result = await validate_audio_upload(upload("recording.webm", content_type, contents))
                self.assertEqual(result, contents)

    async def test_fake_webm_container_is_rejected(self) -> None:
        with self.assertRaises(HTTPException) as raised:
            await validate_audio_upload(upload("recording.webm", "audio/webm", b"not-webm"))
        self.assertEqual(raised.exception.status_code, 422)

    async def test_wav_and_mp3_are_rejected(self) -> None:
        for filename, content_type in (("recording.wav", "audio/wav"), ("recording.mp3", "audio/mpeg")):
            with self.subTest(filename=filename), self.assertRaises(HTTPException) as raised:
                await validate_audio_upload(upload(filename, content_type, WEBM_EBML_SIGNATURE))
            self.assertEqual(raised.exception.status_code, 415)

    async def test_non_webm_mime_is_rejected_even_with_webm_extension(self) -> None:
        with self.assertRaises(HTTPException) as raised:
            await validate_audio_upload(upload("recording.webm", "audio/ogg", WEBM_EBML_SIGNATURE))
        self.assertEqual(raised.exception.status_code, 415)

    async def test_empty_webm_is_rejected(self) -> None:
        with self.assertRaises(HTTPException) as raised:
            await validate_audio_upload(upload("recording.webm", "audio/webm", b""))
        self.assertEqual(raised.exception.status_code, 422)
