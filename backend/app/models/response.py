from typing import Any

from pydantic import BaseModel


class APIError(BaseModel):
    code: str
    message: str
    retryable: bool = False


class APIResponse(BaseModel):
    success: bool
    message: str
    data: Any | None = None
    error: APIError | None = None
