from fastapi import FastAPI

from app.api.upload import router as upload_router
from app.api.chat import router as chat_router
from app.api.documents import router as document_router

from app.exceptions.handlers import global_exception_handler

app = FastAPI(
    title="Financial RAG API"
)

app.add_exception_handler(
    Exception,
    global_exception_handler
)

app.include_router(upload_router)
app.include_router(chat_router)
app.include_router(document_router)