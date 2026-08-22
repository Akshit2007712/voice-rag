"""Small user-facing error schema, separate from diagnostics schemas."""

from pydantic import BaseModel


class UserError(BaseModel):
    code: str
    message: str


class HarnessErrorResponse(BaseModel):
    """Stable content-free error response returned by one-shot HTTP endpoints."""

    error: UserError
    request_id: str
