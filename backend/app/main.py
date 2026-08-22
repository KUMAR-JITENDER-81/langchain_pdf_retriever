from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware

from app.api.chat import router as chat_router
from app.api.documents import router as document_router
from app.api.health import router as health_router
from app.api.quality import router as quality_router
from app.api.upload import router as upload_router
from app.core.config import settings
from app.core.errors import AppError
from app.core.middleware import request_middleware
from app.exceptions.handlers import (
    app_exception_handler,
    global_exception_handler,
    http_exception_handler,
    validation_exception_handler,
)
from app.services.metadata_service import initialize_metadata_store, mark_interrupted_work
from app.services.quality_service import initialize_quality_store


@asynccontextmanager
async def lifespan(app: FastAPI):
    initialize_metadata_store()
    initialize_quality_store()
    mark_interrupted_work()
    from app.services.indexing_service import (
        retry_interrupted_jobs,
        retry_legacy_provider_failures,
    )
    from app.services.generation_service import start_ollama_warmup

    retry_legacy_provider_failures()
    retry_interrupted_jobs()
    start_ollama_warmup()
    try:
        yield
    finally:
        from app.services.indexing_service import shutdown_index_executor

        shutdown_index_executor()

app = FastAPI(
    title="LangChain PDF Retriever API",
    description="Upload, OCR, index, search, and chat with PDF documents.",
    version="0.3.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in settings.FRONTEND_ORIGINS.split(",")],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.middleware("http")(request_middleware)


@app.get("/", include_in_schema=False)
def service_home() -> dict[str, object]:
    return {
        "service": "LangChain PDF Retriever API",
        "status": "running",
        "cost_mode": "free-local",
        "health": "/health",
        "documentation": "/docs",
        "frontend": "http://127.0.0.1:5173",
    }

app.add_exception_handler(AppError, app_exception_handler)
app.add_exception_handler(HTTPException, http_exception_handler)
app.add_exception_handler(RequestValidationError, validation_exception_handler)
app.add_exception_handler(Exception, global_exception_handler)

app.include_router(health_router)
app.include_router(upload_router)
app.include_router(document_router)
app.include_router(chat_router)
app.include_router(quality_router)
