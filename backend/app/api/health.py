from pathlib import Path

from fastapi import APIRouter

from app.core.config import settings
from app.models.response import APIResponse
from app.services.metadata_service import initialize_metadata_store
from app.services.ocr_service import resolve_tessdata_path


router = APIRouter(tags=["Health"])


@router.get("/health", response_model=APIResponse)
def health() -> APIResponse:
    initialize_metadata_store()
    Path(settings.UPLOAD_DIR).mkdir(parents=True, exist_ok=True)
    Path(settings.CHROMA_DIR).mkdir(parents=True, exist_ok=True)

    tessdata_path = resolve_tessdata_path()
    return APIResponse(
        success=True,
        message="Service is healthy",
        data={
            "status": "healthy",
            "embedding_provider": settings.EMBEDDING_PROVIDER,
            "embedding_model": (
                settings.OLLAMA_EMBEDDING_MODEL
                if settings.EMBEDDING_PROVIDER.lower() == "ollama"
                else settings.EMBEDDING_MODEL
            ),
            "generation_provider": settings.GENERATION_PROVIDER,
            "generation_model": settings.MODEL_NAME,
            "openai_key_configured": bool(settings.OPENAI_API_KEY),
            "ocr_enabled": settings.OCR_ENABLED,
            "ocr_provider": settings.OCR_PROVIDER,
            "tesseract_available": tessdata_path is not None,
            "tesseract_data_path": str(tessdata_path) if tessdata_path else None,
            "limits": {
                "upload_mb": settings.MAX_UPLOAD_SIZE_MB,
                "pages": settings.MAX_PDF_PAGES,
            },
        },
    )
