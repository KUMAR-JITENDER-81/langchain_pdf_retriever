from __future__ import annotations

from collections import defaultdict, deque
import re
from threading import RLock
import time
from uuid import uuid4

from fastapi import Request
from fastapi.responses import JSONResponse

from app.core.config import settings
from app.core.logger import logger


REQUEST_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{8,80}$")
_RATE_LIMITS: dict[tuple[str, str], deque[float]] = defaultdict(deque)
_RATE_LIMIT_LOCK = RLock()


async def request_middleware(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID", "")
    if not REQUEST_ID_PATTERN.fullmatch(request_id):
        request_id = uuid4().hex
    request.state.request_id = request_id

    retry_after = _rate_limit_retry_after(request)
    if retry_after is not None:
        response = JSONResponse(
            status_code=429,
            content={
                "success": False,
                "message": "Too many requests; retry shortly",
                "detail": "Too many requests; retry shortly",
                "data": None,
                "error": {
                    "code": "application_rate_limited",
                    "message": "Too many requests; retry shortly",
                    "retryable": True,
                },
            },
            headers={"Retry-After": str(retry_after)},
        )
    else:
        started = time.perf_counter()
        response = await call_next(request)
        duration_ms = (time.perf_counter() - started) * 1000
        if response.status_code >= 400 or duration_ms >= 1000:
            logger.info(
                "%s %s -> %s in %.1f ms request_id=%s",
                request.method,
                request.url.path,
                response.status_code,
                duration_ms,
                request_id,
            )

    response.headers["X-Request-ID"] = request_id
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
    return response


def _rate_limit_retry_after(request: Request) -> int | None:
    if not settings.RATE_LIMIT_ENABLED:
        return None
    limit = _request_limit(request)
    if limit is None or limit <= 0:
        return None

    client_host = request.client.host if request.client else "unknown"
    bucket_name = _bucket_name(request)
    key = (client_host, bucket_name)
    now = time.monotonic()
    cutoff = now - 60
    with _RATE_LIMIT_LOCK:
        bucket = _RATE_LIMITS[key]
        while bucket and bucket[0] <= cutoff:
            bucket.popleft()
        if len(bucket) >= limit:
            return max(1, int(60 - (now - bucket[0])) + 1)
        bucket.append(now)
    return None


def _request_limit(request: Request) -> int | None:
    if request.method != "POST":
        return None
    path = request.url.path.rstrip("/")
    if path.startswith("/chat"):
        return settings.CHAT_RATE_LIMIT_PER_MINUTE
    if path == "/upload":
        return settings.UPLOAD_RATE_LIMIT_PER_MINUTE
    if path.endswith("/index"):
        return settings.INDEX_RATE_LIMIT_PER_MINUTE
    return None


def _bucket_name(request: Request) -> str:
    path = request.url.path.rstrip("/")
    if path.startswith("/chat"):
        return "chat"
    if path == "/upload":
        return "upload"
    return "index"
