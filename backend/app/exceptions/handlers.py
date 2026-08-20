from fastapi import HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.core.errors import AppError
from app.core.logger import logger


def _error_payload(code: str, message: str, retryable: bool = False) -> dict:
    return {
        "success": False,
        "message": message,
        "detail": message,
        "data": None,
        "error": {
            "code": code,
            "message": message,
            "retryable": retryable,
        },
    }


async def app_exception_handler(request: Request, exc: AppError) -> JSONResponse:
    log_method = logger.warning if exc.status_code < 500 else logger.error
    log_method("%s %s failed: %s (%s)", request.method, request.url.path, exc.message, exc.code)
    return JSONResponse(
        status_code=exc.status_code,
        content=_error_payload(exc.code, exc.message, exc.retryable),
    )


async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    message = str(exc.detail)
    return JSONResponse(
        status_code=exc.status_code,
        content=_error_payload(f"http_{exc.status_code}", message),
        headers=exc.headers,
    )


async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    logger.info("Request validation failed for %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=422,
        content={
            **_error_payload("validation_error", "Request validation failed"),
            "details": exc.errors(),
        },
    )


async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled error for %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content=_error_payload(
            "internal_server_error",
            "An unexpected server error occurred",
            retryable=True,
        ),
    )
