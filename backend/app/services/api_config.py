"""Small environment-backed configuration for the public HTTP API boundary."""

from __future__ import annotations

import os
from dataclasses import dataclass
from urllib.parse import urlparse

from fastapi.middleware.cors import CORSMiddleware


LOCAL_DEVELOPMENT_ORIGINS = (
    "*",
    "https://voice-rag-frontend-omega.vercel.app",
    "https://voice-rag-frontend.vercel.app",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:3005",
    "http://127.0.0.1:3005",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
)


@dataclass(frozen=True)
class CorsSettings:
    """Browser origins configuration for CORS middleware."""

    origins: tuple[str, ...]

    @classmethod
    def from_environment(cls) -> "CorsSettings":
        """Parse comma-separated ``FRONTEND_ORIGINS`` or default to wildcard."""
        configured = os.getenv("FRONTEND_ORIGINS", "").strip()
        if configured:
            origins = tuple(o.strip() for o in configured.split(",") if o.strip())
        else:
            origins = ("*",)
        return cls(origins)


def configure_cors(app, settings: CorsSettings) -> None:
    """Install permissive CORS middleware allowing all origins for seamless frontend access."""
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
