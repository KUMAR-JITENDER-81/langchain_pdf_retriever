from __future__ import annotations

from functools import lru_cache
from typing import Literal

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
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


AnswerMode = Literal["quick", "balanced", "deep"]


def generation_identity() -> tuple[str, str]:
    provider = settings.GENERATION_PROVIDER.strip().lower()
    model = settings.OLLAMA_CHAT_MODEL if provider == "ollama" else settings.MODEL_NAME
    return provider, model


def get_chat_model(mode: AnswerMode = "balanced", *, streaming: bool = False) -> BaseChatModel:
    provider, model = generation_identity()
    if provider == "openai" and not settings.OPENAI_API_KEY:
        raise ConfigurationError(
            "OPENAI_API_KEY is required for the OpenAI generation provider"
        )
    if provider not in {"openai", "ollama"}:
        raise ConfigurationError("GENERATION_PROVIDER must be either openai or ollama")
    return _create_chat_model(
        provider,
        model,
        settings.OPENAI_API_KEY if provider == "openai" else "",
        settings.OLLAMA_BASE_URL if provider == "ollama" else "",
        mode,
        streaming,
        settings.TEMPERATURE,
        settings.MAX_ANSWER_TOKENS,
        settings.GENERATION_TIMEOUT_SECONDS,
    )


@lru_cache(maxsize=24)
def _create_chat_model(
    provider: str,
    model: str,
    api_key: str,
    base_url: str,
    mode: AnswerMode,
    streaming: bool,
    temperature: float,
    configured_max_tokens: int,
    timeout: float,
) -> BaseChatModel:
    output_tokens = _mode_output_tokens(mode, configured_max_tokens)
    if provider == "ollama":
        return ChatOllama(
            model=model,
            base_url=base_url,
            temperature=temperature,
            num_predict=output_tokens,
            validate_model_on_init=False,
        )

    kwargs = {
        "model": model,
        "api_key": api_key,
        "max_tokens": output_tokens,
        "streaming": streaming,
        "stream_usage": streaming,
        "max_retries": 2,
        "request_timeout": timeout,
        "use_responses_api": True,
        "store": False,
    }
    if model.startswith(("gpt-5", "o1", "o3", "o4")):
        kwargs["reasoning_effort"] = {
            "quick": "low",
            "balanced": "medium",
            "deep": "high",
        }[mode]
    else:
        kwargs["temperature"] = temperature
    return ChatOpenAI(**kwargs)


def clear_generation_cache() -> None:
    _create_chat_model.cache_clear()


def translate_generation_error(exc: Exception) -> AppError:
    if isinstance(exc, AppError):
        return exc
    if isinstance(exc, RateLimitError):
        error_code = _openai_error_code(exc)
        if error_code in {"insufficient_quota", "billing_hard_limit_reached"}:
            return ProviderQuotaError(
                "OpenAI answer-generation quota is unavailable; check billing and limits"
            )
        return ProviderUnavailableError(
            "Answer generation is rate limited; retry later",
            code="generation_rate_limited",
        )
    if isinstance(exc, AuthenticationError):
        return ConfigurationError("The configured OpenAI API key was rejected")
    if isinstance(exc, APIConnectionError):
        return ProviderUnavailableError(
            "Could not connect to the answer-generation provider",
            code="generation_connection_error",
        )
    if isinstance(exc, APIStatusError):
        return ProviderUnavailableError(
            f"The answer-generation provider returned status {exc.status_code}",
            code="generation_provider_error",
        )

    message = str(exc).lower()
    if "connection refused" in message or "failed to connect" in message:
        return ProviderUnavailableError(
            "Could not connect to Ollama; start Ollama or select OpenAI generation",
            code="ollama_unavailable",
        )
    return ProviderUnavailableError(
        "Answer generation failed",
        code="generation_failed",
    )


def _mode_output_tokens(mode: AnswerMode, configured: int) -> int:
    limits = {
        "quick": min(configured, 450),
        "balanced": configured,
        "deep": max(configured, 1500),
    }
    return max(128, limits[mode])


def _openai_error_code(exc: RateLimitError) -> str | None:
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        nested = body.get("error")
        if isinstance(nested, dict):
            return nested.get("code")
        code = body.get("code")
        return str(code) if code else None
    return None
