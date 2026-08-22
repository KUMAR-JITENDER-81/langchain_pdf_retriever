from functools import lru_cache
from pathlib import Path
from threading import RLock

from chromadb.utils.embedding_functions import (
    DefaultEmbeddingFunction,
    ONNXMiniLM_L6_V2,
)
from langchain_core.embeddings import Embeddings
from langchain_ollama import OllamaEmbeddings

from app.core.config import settings
from app.core.errors import (
    AppError,
    ConfigurationError,
    ProviderUnavailableError,
)


_LOCAL_MODEL_LOCK = RLock()


class LocalMiniLMEmbeddings(Embeddings):
    """LangChain adapter for Chroma's free, local ONNX MiniLM model."""

    def __init__(self) -> None:
        self._embedding_function = DefaultEmbeddingFunction()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        with _LOCAL_MODEL_LOCK:
            vectors = self._embedding_function(texts)
        return [[float(value) for value in vector] for vector in vectors]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


def embedding_identity() -> tuple[str, str]:
    provider = settings.EMBEDDING_PROVIDER.strip().lower()
    model = (
        settings.OLLAMA_EMBEDDING_MODEL
        if provider == "ollama"
        else settings.LOCAL_EMBEDDING_MODEL
    )
    return provider, model


def get_embedding_model() -> Embeddings:
    """Return a cached free local or Ollama embedding client."""
    provider, model = embedding_identity()
    if provider not in {"local", "ollama"}:
        raise ConfigurationError("EMBEDDING_PROVIDER must be either local or ollama")
    return _create_embedding_model(
        provider,
        model,
        settings.OLLAMA_BASE_URL if provider == "ollama" else "",
        settings.EMBEDDING_BATCH_SIZE,
    )


@lru_cache(maxsize=4)
def _create_embedding_model(
    provider: str,
    model: str,
    base_url: str,
    batch_size: int,
) -> Embeddings:
    del batch_size  # Batching is controlled by the indexing service.
    if provider == "local":
        if model != ONNXMiniLM_L6_V2.MODEL_NAME:
            raise ConfigurationError(
                "LOCAL_EMBEDDING_MODEL must be all-MiniLM-L6-v2 for the bundled ONNX runtime"
            )
        return LocalMiniLMEmbeddings()
    return OllamaEmbeddings(
        model=model,
        base_url=base_url,
        validate_model_on_init=False,
    )


def clear_embedding_cache() -> None:
    _create_embedding_model.cache_clear()


def local_embedding_status() -> dict[str, object]:
    cache_path = Path(ONNXMiniLM_L6_V2.DOWNLOAD_PATH)
    model_files = list(cache_path.rglob("*.onnx")) if cache_path.is_dir() else []
    return {
        "model": ONNXMiniLM_L6_V2.MODEL_NAME,
        "cached": bool(model_files),
        "cache_path": str(cache_path),
    }


def warm_local_embedding_model() -> dict[str, object]:
    """Download/load the local model once and return its vector dimension."""
    provider, model = embedding_identity()
    if provider != "local":
        return {"provider": provider, "model": model, "skipped": True}
    vector = get_embedding_model().embed_query("local PDF retrieval readiness check")
    return {
        "provider": provider,
        "model": model,
        "dimension": len(vector),
        "cached": True,
        "skipped": False,
    }


def translate_embedding_error(exc: Exception) -> AppError:
    """Convert local-provider exceptions into safe, actionable errors."""
    if isinstance(exc, AppError):
        return exc

    provider, model = embedding_identity()
    message = str(exc).lower()
    if provider == "local":
        if any(term in message for term in ("download", "network", "url", "ssl")):
            return ProviderUnavailableError(
                "The free MiniLM model could not be downloaded. Connect once and retry indexing.",
                code="local_embedding_download_failed",
            )
        return ProviderUnavailableError(
            "The local MiniLM embedding model could not process this document",
            code="local_embedding_failed",
        )

    if "connection refused" in message or "failed to connect" in message:
        return ProviderUnavailableError(
            "Could not connect to Ollama; start Ollama or use local embeddings",
            code="ollama_unavailable",
        )
    if "model" in message and ("not found" in message or "404" in message):
        return ConfigurationError(
            f"Embedding model '{model}' is not installed in Ollama"
        )
    return ProviderUnavailableError(
        "Ollama embedding generation failed",
        code="embedding_failed",
    )
