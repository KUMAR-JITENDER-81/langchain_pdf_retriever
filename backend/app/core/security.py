import secrets

from fastapi import Header, HTTPException, status

from app.core.config import settings


def require_api_token(authorization: str | None = Header(default=None)) -> None:
    """Require a bearer token when API_AUTH_TOKEN is configured."""
    if not settings.API_AUTH_TOKEN:
        return

    expected = f"Bearer {settings.API_AUTH_TOKEN}"
    if authorization is None or not secrets.compare_digest(authorization, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API token",
            headers={"WWW-Authenticate": "Bearer"},
        )
