"""Small environment-backed configuration for the public HTTP API boundary."""

from __future__ import annotations

import os
from dataclasses import dataclass
from urllib.parse import urlparse

from fastapi.middleware.cors import CORSMiddleware


LOCAL_DEVELOPMENT_ORIGINS = (
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:3005",
    "http://127.0.0.1:3005",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
)


@dataclass(frozen=True)
class CorsSettings:
    """Explicit browser origins; wildcard origins are intentionally unsupported."""

    origins: tuple[str, ...]

    @classmethod
    def from_environment(cls) -> "CorsSettings":
        """Parse comma-separated ``FRONTEND_ORIGINS`` once during app import."""
        configured = os.getenv("FRONTEND_ORIGINS", "").strip()
        origins = LOCAL_DEVELOPMENT_ORIGINS if not configured else tuple(
            origin.strip().rstrip("/") for origin in configured.split(",") if origin.strip()
        )
        if not origins:
            raise RuntimeError("FRONTEND_ORIGINS must contain at least one origin")
        normalized: list[str] = []
        for origin in origins:
            if origin == "*":
                raise RuntimeError("FRONTEND_ORIGINS must not contain '*'")
            parsed = urlparse(origin)
            if parsed.scheme not in {"http", "https"} or not parsed.netloc or parsed.path not in {"", "/"}:
                raise RuntimeError("FRONTEND_ORIGINS must contain comma-separated http(s) origins only")
            if origin not in normalized:
                normalized.append(origin)
        return cls(tuple(normalized))


def configure_cors(app, settings: CorsSettings) -> None:
    """Install explicit-origin CORS for public browser endpoints only."""
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.origins),
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        # Diagnostics remains operator-only; its credential is intentionally not
        # exposed as a browser CORS request header.
        allow_headers=["Content-Type"],
    )
