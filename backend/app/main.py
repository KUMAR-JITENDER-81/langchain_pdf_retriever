from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.upload import router as upload_router
from app.api.chat import router as chat_router
from app.api.documents import router as document_router

from app.exceptions.handlers import global_exception_handler
from app.core.config import settings

app = FastAPI(
    title="Financial RAG API"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in settings.FRONTEND_ORIGINS.split(",")],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_exception_handler(
    Exception,
    global_exception_handler
)

app.include_router(upload_router)
app.include_router(chat_router)
app.include_router(document_router)
