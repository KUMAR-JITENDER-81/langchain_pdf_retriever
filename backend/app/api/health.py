from pathlib import Path

from fastapi import APIRouter, Depends

from app.core.config import settings
from app.core.security import require_api_token
from app.models.response import APIResponse
from app.services.embedding_service import embedding_identity, local_embedding_status
from app.services.generation_service import (
    generation_identity,
    ollama_status,
    ollama_warmup_status,
    warmup_ollama_model,
)
from app.services.metadata_service import initialize_metadata_store
from app.services.indexing_service import index_queue_status
from app.services.ocr_service import resolve_tessdata_path
from app.services.reranker_service import reranker_status, warmup_reranker


router = APIRouter(tags=["Health"])


@router.get("/health", response_model=APIResponse)
def health() -> APIResponse:
    initialize_metadata_store()
    Path(settings.UPLOAD_DIR).mkdir(parents=True, exist_ok=True)
    Path(settings.CHROMA_DIR).mkdir(parents=True, exist_ok=True)

    tessdata_path = resolve_tessdata_path()
    embedding_provider, embedding_model = embedding_identity()
    generation_provider, generation_model = generation_identity("deep")
    embedding_status = local_embedding_status()
    ollama = ollama_status()
    fast_model_ready = bool(
        ollama["available"] and ollama["effective_fast_model_installed"]
    )
    deep_model_ready = bool(
        ollama["available"] and ollama["effective_deep_model_installed"]
    )
    quick_ready = bool(
        generation_provider == "local" or settings.QUICK_MODE_LOCAL or fast_model_ready
    )
    balanced_ready = bool(
        generation_provider == "local" or settings.BALANCED_MODE_LOCAL or fast_model_ready
    )
    deep_ready = bool(
        generation_provider == "local"
        or deep_model_ready
        or settings.LOCAL_ANSWER_FALLBACK
    )
    return APIResponse(
        success=True,
        message="Service is healthy",
        data={
            "status": "healthy",
            "cost_mode": "free-local",
            "embedding_provider": embedding_provider,
            "embedding_model": embedding_model,
            "embedding_model_cached": bool(embedding_status["cached"]),
            "generation_provider": generation_provider,
            "generation_model": generation_model,
            "generation_ready": quick_ready and balanced_ready and deep_ready,
            "local_answer_fallback": settings.LOCAL_ANSWER_FALLBACK,
            "quick_mode_local": settings.QUICK_MODE_LOCAL,
            "balanced_mode_local": settings.BALANCED_MODE_LOCAL,
            "answer_modes": {
                "quick": {
                    "ready": quick_ready,
                    "engine": "extractive" if settings.QUICK_MODE_LOCAL else "ollama",
                    "model": settings.LOCAL_ANSWER_MODEL
                    if settings.QUICK_MODE_LOCAL
                    else ollama["effective_fast_model"],
                },
                "balanced": {
                    "ready": balanced_ready,
                    "engine": "extractive" if settings.BALANCED_MODE_LOCAL else "ollama",
                    "model": settings.LOCAL_ANSWER_MODEL
                    if settings.BALANCED_MODE_LOCAL
                    else ollama["effective_fast_model"],
                    "using_fallback_model": bool(ollama["fast_model_fallback"]),
                },
                "deep": {
                    "ready": deep_ready,
                    "engine": "ollama" if deep_model_ready else "extractive-fallback",
                    "model": ollama["effective_deep_model"],
                    "using_fallback_model": bool(ollama["deep_model_fallback"]),
                },
            },
            "ollama": ollama,
            "ollama_warmup": ollama_warmup_status(),
            "reranker": reranker_status(),
            "index_queue": index_queue_status(),
            "answer_cache": {
                "enabled": settings.ANSWER_CACHE_ENABLED,
                "ttl_hours": settings.ANSWER_CACHE_TTL_HOURS,
                "max_entries": settings.ANSWER_CACHE_MAX_ENTRIES,
            },
            "ocr_enabled": settings.OCR_ENABLED,
            "ocr_provider": settings.OCR_PROVIDER,
            "ocr_vision_model": settings.OLLAMA_OCR_MODEL,
            "ocr_vision_fallback": settings.OLLAMA_OCR_FALLBACK,
            "tesseract_available": tessdata_path is not None,
            "tesseract_data_path": str(tessdata_path) if tessdata_path else None,
            "limits": {
                "upload_mb": settings.MAX_UPLOAD_SIZE_MB,
                "pages": settings.MAX_PDF_PAGES,
            },
        },
    )


@router.post(
    "/ollama/warmup",
    response_model=APIResponse,
    dependencies=[Depends(require_api_token)],
)
def warmup_ollama() -> APIResponse:
    result = warmup_ollama_model("balanced")
    return APIResponse(
        success=True,
        message=str(result["message"]),
        data=result,
    )


@router.post(
    "/reranker/warmup",
    response_model=APIResponse,
    dependencies=[Depends(require_api_token)],
)
def warmup_local_reranker() -> APIResponse:
    result = warmup_reranker()
    return APIResponse(
        success=bool(result["loaded"]),
        message=(
            "Semantic reranker is ready"
            if result["loaded"]
            else str(result.get("error") or "Semantic reranker is unavailable")
        ),
        data=result,
    )
