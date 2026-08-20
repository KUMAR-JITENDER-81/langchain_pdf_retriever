from functools import lru_cache

from langchain_core.embeddings import Embeddings
from langchain_ollama import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings
from openai import (
    APIConnectionError,
    APIStatusError,
    AuthenticationError,
    RateLimitError,
)

from app.core.config import settings
from app.core.errors import (
    AppError,
    ConfigurationError,
    ProviderQuotaError,
    ProviderUnavailableError,
)


def embedding_identity() -> tuple[str, str]:
    provider = settings.EMBEDDING_PROVIDER.strip().lower()
    model = (
        settings.OLLAMA_EMBEDDING_MODEL
        if provider == "ollama"
        else settings.EMBEDDING_MODEL
    )
    return provider, model


def get_embedding_model() -> Embeddings:
    """Return a cached embedding client for the configured provider."""
    provider, model = embedding_identity()
    if provider == "openai":
        if not settings.OPENAI_API_KEY:
            raise ConfigurationError(
                "OPENAI_API_KEY is required for the OpenAI embedding provider"
            )
        return _create_embedding_model(
            provider,
            model,
            settings.OPENAI_API_KEY,
            "",
            settings.EMBEDDING_BATCH_SIZE,
        )
    if provider == "ollama":
        return _create_embedding_model(
            provider,
            model,
            "",
            settings.OLLAMA_BASE_URL,
            settings.EMBEDDING_BATCH_SIZE,
        )
    raise ConfigurationError("EMBEDDING_PROVIDER must be either openai or ollama")


@lru_cache(maxsize=8)
def _create_embedding_model(
    provider: str,
    model: str,
    api_key: str,
    base_url: str,
    batch_size: int,
) -> Embeddings:
    if provider == "openai":
        return OpenAIEmbeddings(
            model=model,
            api_key=api_key,
            chunk_size=batch_size,
            max_retries=3,
            request_timeout=60,
        )
    return OllamaEmbeddings(
        model=model,
        base_url=base_url,
        validate_model_on_init=False,
    )


def clear_embedding_cache() -> None:
    _create_embedding_model.cache_clear()


def translate_embedding_error(exc: Exception) -> AppError:
    """Convert provider exceptions into safe, actionable application errors."""
    if isinstance(exc, AppError):
        return exc
    if isinstance(exc, RateLimitError):
        error_code = _openai_error_code(exc)
        if error_code in {"insufficient_quota", "billing_hard_limit_reached"}:
            return ProviderQuotaError(
                "OpenAI embedding quota is unavailable; check billing and project limits"
            )
        return ProviderUnavailableError(
            "OpenAI embeddings are rate limited; retry later",
            code="openai_rate_limited",
        )
    if isinstance(exc, AuthenticationError):
        return ConfigurationError("The configured OpenAI API key was rejected")
    if isinstance(exc, APIConnectionError):
        return ProviderUnavailableError(
            "Could not connect to the OpenAI embedding service",
            code="openai_connection_error",
        )
    if isinstance(exc, APIStatusError):
        return ProviderUnavailableError(
            f"The embedding provider returned status {exc.status_code}",
            code="embedding_provider_error",
        )

    message = str(exc).lower()
    if "connection refused" in message or "failed to connect" in message:
        return ProviderUnavailableError(
            "Could not connect to Ollama; start Ollama or select OpenAI embeddings",
            code="ollama_unavailable",
        )
    if "model" in message and "not found" in message:
        return ConfigurationError(
            f"Embedding model '{embedding_identity()[1]}' is not installed or available"
        )
    return ProviderUnavailableError(
        "Embedding generation failed",
        code="embedding_failed",
    )


def _openai_error_code(exc: RateLimitError) -> str | None:
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        nested = body.get("error")
        if isinstance(nested, dict):
            return nested.get("code")
        code = body.get("code")
        return str(code) if code else None
    return None
