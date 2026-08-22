"""Replaceable authentication boundary for developer-only diagnostics."""

from __future__ import annotations

import os
import secrets
from typing import Annotated

from fastapi import Header, HTTPException, status


async def require_diagnostics_access(
    supplied_key: Annotated[str | None, Header(alias="X-Diagnostics-Key")] = None,
) -> None:
    """Authorize diagnostics with its dedicated secret and no provider credentials."""
    configured_key = os.getenv("DIAGNOSTICS_API_KEY")
    if not configured_key:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Diagnostics unavailable.")
    if supplied_key is None or not secrets.compare_digest(supplied_key, configured_key):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")
