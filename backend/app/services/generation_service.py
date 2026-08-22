from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
from functools import lru_cache
import json
from threading import BoundedSemaphore, RLock, Thread
import time
from typing import Literal
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_ollama import ChatOllama

from app.core.config import settings
from app.core.errors import (
    AppError,
    ConfigurationError,
    ProviderUnavailableError,
)
from app.core.logger import logger


AnswerMode = Literal["quick", "balanced", "deep"]
_STATUS_LOCK = RLock()
_STATUS_CACHE: tuple[float, dict[str, object]] | None = None
_WARMUP_LOCK = RLock()
_WARMUP_STATUS: dict[str, object] = {
    "state": "idle",
    "model": None,
    "duration_ms": None,
    "message": "The local AI model has not been warmed yet",
    "updated_at": None,
}
_GENERATION_SLOTS = BoundedSemaphore(
    max(1, settings.OLLAMA_MAX_CONCURRENT_GENERATIONS)
)


def generation_identity(mode: AnswerMode = "deep") -> tuple[str, str]:
    provider = settings.GENERATION_PROVIDER.strip().lower()
    model = (
        settings.LOCAL_ANSWER_MODEL
        if provider == "local"
        else generation_model_for_mode(mode)
    )
    return provider, model


def generation_model_for_mode(mode: AnswerMode) -> str:
    """Use a small text model for responsive answers and the larger model for Deep."""
    if mode in {"quick", "balanced"}:
        return settings.OLLAMA_FAST_MODEL.strip() or settings.OLLAMA_CHAT_MODEL
    return settings.OLLAMA_CHAT_MODEL


def effective_generation_model(
    mode: AnswerMode,
    *,
    status: dict[str, object] | None = None,
) -> str:
    """Resolve an installed text model, falling back between fast and deep models."""
    configured = generation_model_for_mode(mode)
    if not settings.OLLAMA_AUTO_MODEL_FALLBACK:
        return configured
    current_status = status if status is not None else ollama_status()
    installed_models = [str(model) for model in current_status.get("installed_models", [])]
    return _resolve_installed_generation_model(mode, installed_models) or configured


def uses_local_answer_provider() -> bool:
    return generation_identity()[0] == "local"


def local_answer_fallback_enabled() -> bool:
    return bool(settings.LOCAL_ANSWER_FALLBACK)


def generation_timeout_for_mode(mode: AnswerMode) -> float:
    if mode == "deep":
        return min(
            settings.GENERATION_TIMEOUT_SECONDS,
            settings.OLLAMA_DEEP_TIMEOUT_SECONDS,
        )
    return min(
        settings.GENERATION_TIMEOUT_SECONDS,
        settings.OLLAMA_BALANCED_TIMEOUT_SECONDS,
    )


@contextmanager
def ollama_generation_slot():
    """Prevent overlapping CPU generations from making every request unusably slow."""
    acquired = _GENERATION_SLOTS.acquire(
        timeout=max(0.0, settings.OLLAMA_QUEUE_TIMEOUT_SECONDS)
    )
    if not acquired:
        raise ProviderUnavailableError(
            "The local AI is busy; used the instant evidence fallback",
            code="ollama_busy",
        )
    try:
        yield
    finally:
        _GENERATION_SLOTS.release()


def get_chat_model(
    mode: AnswerMode = "balanced",
    *,
    streaming: bool = False,
) -> BaseChatModel:
    provider, _ = generation_identity(mode)
    if provider == "local":
        raise ConfigurationError(
            "The local extractive provider does not expose a chat model"
        )
    if provider != "ollama":
        raise ConfigurationError("GENERATION_PROVIDER must be either ollama or local")
    model = effective_generation_model(mode)
    return _create_chat_model(
        model,
        settings.OLLAMA_BASE_URL,
        mode,
        streaming,
        settings.TEMPERATURE,
        settings.MAX_ANSWER_TOKENS,
        generation_timeout_for_mode(mode),
        settings.OLLAMA_KEEP_ALIVE,
        settings.OLLAMA_NUM_CTX,
    )


@lru_cache(maxsize=12)
def _create_chat_model(
    model: str,
    base_url: str,
    mode: AnswerMode,
    streaming: bool,
    temperature: float,
    configured_max_tokens: int,
    timeout: float,
    keep_alive: str,
    num_ctx: int,
) -> BaseChatModel:
    del streaming  # ChatOllama supports invoke and stream on the same client.
    return ChatOllama(
        model=model,
        base_url=base_url,
        temperature=temperature,
        num_predict=_mode_output_tokens(mode, configured_max_tokens),
        num_ctx=max(2048, num_ctx),
        keep_alive=keep_alive,
        reasoning=False,
        seed=42,
        validate_model_on_init=False,
        client_kwargs={"timeout": timeout},
    )


def clear_generation_cache() -> None:
    global _STATUS_CACHE
    _create_chat_model.cache_clear()
    with _STATUS_LOCK:
        _STATUS_CACHE = None


def ollama_status(*, force: bool = False) -> dict[str, object]:
    """Return Ollama/server/model readiness without making health itself fail."""
    global _STATUS_CACHE
    now = time.monotonic()
    with _STATUS_LOCK:
        if not force and _STATUS_CACHE and now - _STATUS_CACHE[0] < 5:
            return dict(_STATUS_CACHE[1])

    target_models = {
        "fast": settings.OLLAMA_FAST_MODEL.strip() or settings.OLLAMA_CHAT_MODEL,
        "deep": settings.OLLAMA_CHAT_MODEL,
        "ocr": settings.OLLAMA_OCR_MODEL,
    }
    request = Request(
        f"{settings.OLLAMA_BASE_URL.rstrip('/')}/api/tags",
        headers={"Accept": "application/json"},
        method="GET",
    )
    status: dict[str, object]
    try:
        with urlopen(request, timeout=2.5) as response:
            payload = json.loads(response.read().decode("utf-8"))
        installed_models = sorted(
            {
                str(item.get("model") or item.get("name") or "")
                for item in payload.get("models", [])
                if isinstance(item, dict)
            }
            - {""}
        )
        model_readiness = {
            role: _model_is_installed(model, installed_models)
            for role, model in target_models.items()
        }
        effective_fast_model = (
            _resolve_installed_generation_model("balanced", installed_models)
            if settings.OLLAMA_AUTO_MODEL_FALLBACK
            else target_models["fast"]
        ) or target_models["fast"]
        effective_deep_model = (
            _resolve_installed_generation_model("deep", installed_models)
            if settings.OLLAMA_AUTO_MODEL_FALLBACK
            else target_models["deep"]
        ) or target_models["deep"]
        loaded_models = _ollama_loaded_models()
        status = {
            "available": True,
            "model": target_models["deep"],
            "model_installed": model_readiness["deep"],
            "fast_model": target_models["fast"],
            "fast_model_installed": model_readiness["fast"],
            "effective_fast_model": effective_fast_model,
            "effective_fast_model_installed": _model_is_installed(
                effective_fast_model, installed_models
            ),
            "fast_model_fallback": effective_fast_model != target_models["fast"],
            "deep_model": target_models["deep"],
            "deep_model_installed": model_readiness["deep"],
            "effective_deep_model": effective_deep_model,
            "effective_deep_model_installed": _model_is_installed(
                effective_deep_model, installed_models
            ),
            "deep_model_fallback": effective_deep_model != target_models["deep"],
            "ocr_model": target_models["ocr"],
            "ocr_model_installed": model_readiness["ocr"],
            "all_required_models_installed": all(model_readiness.values()),
            "installed_models": installed_models,
            "loaded_models": loaded_models,
            "base_url": settings.OLLAMA_BASE_URL,
            "error": None,
        }
    except (HTTPError, URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
        status = {
            "available": False,
            "model": target_models["deep"],
            "model_installed": False,
            "fast_model": target_models["fast"],
            "fast_model_installed": False,
            "effective_fast_model": target_models["fast"],
            "effective_fast_model_installed": False,
            "fast_model_fallback": False,
            "deep_model": target_models["deep"],
            "deep_model_installed": False,
            "effective_deep_model": target_models["deep"],
            "effective_deep_model_installed": False,
            "deep_model_fallback": False,
            "ocr_model": target_models["ocr"],
            "ocr_model_installed": False,
            "all_required_models_installed": False,
            "installed_models": [],
            "loaded_models": [],
            "base_url": settings.OLLAMA_BASE_URL,
            "error": _safe_connection_message(exc),
        }

    with _STATUS_LOCK:
        _STATUS_CACHE = (now, status)
    return dict(status)


def ollama_warmup_status() -> dict[str, object]:
    with _WARMUP_LOCK:
        return dict(_WARMUP_STATUS)


def warmup_ollama_model(mode: AnswerMode = "balanced") -> dict[str, object]:
    """Load the selected Ollama model into memory without generating an answer."""
    if settings.GENERATION_PROVIDER.strip().lower() != "ollama":
        raise ConfigurationError("Ollama generation is not enabled")

    status = ollama_status(force=True)
    if not status.get("available"):
        error = ProviderUnavailableError(
            "Ollama is not reachable; start Ollama and try again",
            code="ollama_unavailable",
        )
        _set_warmup_status("error", None, None, error.message)
        raise error

    model = effective_generation_model(mode, status=status)
    if not _model_is_installed(model, [str(value) for value in status["installed_models"]]):
        error = ConfigurationError(
            f"The Ollama model '{model}' is not installed"
        )
        _set_warmup_status("error", model, None, error.message)
        raise error

    _set_warmup_status("warming", model, None, f"Loading {model} into memory")
    payload = json.dumps(
        {
            "model": model,
            "prompt": "",
            "stream": False,
            "keep_alive": settings.OLLAMA_KEEP_ALIVE,
            "options": {"num_ctx": max(2048, settings.OLLAMA_NUM_CTX)},
        }
    ).encode("utf-8")
    request = Request(
        f"{settings.OLLAMA_BASE_URL.rstrip('/')}/api/generate",
        data=payload,
        headers={"Accept": "application/json", "Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    try:
        with urlopen(request, timeout=settings.OLLAMA_WARMUP_TIMEOUT_SECONDS) as response:
            response_payload = json.loads(response.read().decode("utf-8"))
        if response_payload.get("done") is not True:
            raise ValueError("Ollama did not confirm that the model loaded")
    except (HTTPError, URLError, TimeoutError, OSError, ValueError, json.JSONDecodeError) as exc:
        duration_ms = round((time.perf_counter() - started) * 1000)
        message = _safe_connection_message(exc)
        _set_warmup_status("error", model, duration_ms, message)
        raise ProviderUnavailableError(message, code="ollama_warmup_failed") from exc

    duration_ms = round((time.perf_counter() - started) * 1000)
    clear_generation_cache()
    result = _set_warmup_status(
        "ready",
        model,
        duration_ms,
        f"{model} is loaded and ready",
    )
    logger.info("Ollama model %s warmed in %d ms", model, duration_ms)
    return result


def start_ollama_warmup() -> bool:
    """Start non-blocking model warmup once during application startup."""
    if (
        not settings.OLLAMA_WARMUP_ON_START
        or settings.GENERATION_PROVIDER.strip().lower() != "ollama"
    ):
        return False
    with _WARMUP_LOCK:
        if _WARMUP_STATUS["state"] == "warming":
            return False
        _WARMUP_STATUS.update(
            state="warming",
            message="Connecting to Ollama and loading the Balanced model",
            updated_at=_utc_now(),
        )

    def run() -> None:
        try:
            warmup_ollama_model("balanced")
        except AppError as exc:
            logger.warning("Ollama startup warmup skipped: %s", exc.message)

    Thread(target=run, name="ollama-warmup", daemon=True).start()
    return True


def translate_generation_error(exc: Exception) -> AppError:
    if isinstance(exc, AppError):
        return exc
    message = str(exc).lower()
    if "connection refused" in message or "failed to connect" in message:
        return ProviderUnavailableError(
            "Ollama is not running; the local evidence-only fallback will be used",
            code="ollama_unavailable",
        )
    if "model" in message and ("not found" in message or "404" in message):
        return ConfigurationError(
            "A configured Ollama model is not installed; run the model setup command"
        )
    if "timed out" in message or "timeout" in message:
        return ProviderUnavailableError(
            "The local Ollama model took too long to answer",
            code="ollama_timeout",
        )
    return ProviderUnavailableError(
        "Local Ollama answer generation failed",
        code="generation_failed",
    )


def _mode_output_tokens(mode: AnswerMode, configured: int) -> int:
    limits = {
        "quick": min(configured, 120),
        "balanced": min(configured, 190),
        "deep": min(configured, 220),
    }
    return max(96, limits[mode])


def _model_is_installed(target: str, installed: list[str]) -> bool:
    normalized_target = target.removesuffix(":latest")
    for model in installed:
        normalized_model = model.removesuffix(":latest")
        if normalized_model == normalized_target:
            return True
    return False


def _resolve_installed_generation_model(
    mode: AnswerMode,
    installed: list[str],
) -> str | None:
    fast_model = settings.OLLAMA_FAST_MODEL.strip() or settings.OLLAMA_CHAT_MODEL
    deep_model = settings.OLLAMA_CHAT_MODEL.strip() or fast_model
    candidates = (
        [deep_model, fast_model]
        if mode == "deep"
        else [fast_model, deep_model]
    )
    for model in dict.fromkeys(candidate for candidate in candidates if candidate):
        if _model_is_installed(model, installed):
            return model
    return None


def _ollama_loaded_models() -> list[str]:
    request = Request(
        f"{settings.OLLAMA_BASE_URL.rstrip('/')}/api/ps",
        headers={"Accept": "application/json"},
        method="GET",
    )
    try:
        with urlopen(request, timeout=1.5) as response:
            payload = json.loads(response.read().decode("utf-8"))
        return sorted(
            {
                str(item.get("model") or item.get("name") or "")
                for item in payload.get("models", [])
                if isinstance(item, dict)
            }
            - {""}
        )
    except (HTTPError, URLError, TimeoutError, OSError, json.JSONDecodeError):
        return []


def _set_warmup_status(
    state: str,
    model: str | None,
    duration_ms: int | None,
    message: str,
) -> dict[str, object]:
    with _WARMUP_LOCK:
        _WARMUP_STATUS.update(
            state=state,
            model=model,
            duration_ms=duration_ms,
            message=message,
            updated_at=_utc_now(),
        )
        return dict(_WARMUP_STATUS)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_connection_message(exc: Exception) -> str:
    if isinstance(exc, HTTPError):
        return f"Ollama returned status {exc.code}"
    if isinstance(exc, TimeoutError):
        return "Ollama health check timed out"
    return "Ollama is not reachable"
