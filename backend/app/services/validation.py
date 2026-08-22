from pathlib import Path

from fastapi import HTTPException, UploadFile, status


ALLOWED_EXTENSIONS = {".webm"}
ALLOWED_MIME_TYPES = {"audio/webm", "video/webm"}
WEBM_EBML_SIGNATURE = b"\x1a\x45\xdf\xa3"
MAX_AUDIO_SIZE_BYTES = 10 * 1024 * 1024


async def validate_audio_upload(upload: UploadFile) -> bytes:
    """Validate upload metadata, read the file once, and enforce its size limit."""
    extension = Path(upload.filename or "").suffix.lower()
    mime = normalize_mime_type(upload.content_type)

    if (extension and extension not in ALLOWED_EXTENSIONS) or mime not in ALLOWED_MIME_TYPES:
        raise HTTPException(status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail="Unsupported audio container.")

    contents = await upload.read()
    if len(contents) >= MAX_AUDIO_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="Audio file must be smaller than 10 MB.",
        )

    if not contents or not is_webm_container(contents):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Audio file is empty or invalid WebM format.",
        )

    return contents


def is_webm_container(contents: bytes) -> bool:
    """Perform a lightweight EBML signature check for browser WebM containers."""
    return len(contents) >= len(WEBM_EBML_SIGNATURE) and contents.startswith(WEBM_EBML_SIGNATURE)


def normalize_mime_type(content_type: str | None) -> str:
    """Return a lowercase base MIME type without optional parameters."""
    return (content_type or "").split(";", 1)[0].strip().lower()
