from collections.abc import Callable
import hashlib
from pathlib import Path
import re
from threading import RLock

from langchain_chroma import Chroma
from langchain_core.documents import Document

from app.core.config import settings
from app.core.errors import AppError
from app.rag.splitter import split_pdf_text
from app.services.embedding_service import (
    embedding_identity,
    get_embedding_model,
    translate_embedding_error,
)
from app.services.metadata_service import get_document


VECTOR_STORE_LOCK = RLock()
VECTOR_STORES: dict[tuple[str, str, str, str, str], Chroma] = {}
IndexProgressCallback = Callable[[str, float], None]
INDEX_SCHEMA_VERSION = "chunks-v2-table-bbox"


def current_collection_name() -> str:
    provider, model = embedding_identity()
    readable_model = re.sub(r"[^a-zA-Z0-9_-]", "_", model)[:28]
    identity_hash = hashlib.sha256(f"{provider}:{model}".encode()).hexdigest()[:8]
    return f"pdf_{provider}_{readable_model}_{identity_hash}"[:63]


def get_vector_store(collection_name: str | None = None) -> Chroma:
    """Open the persistent Chroma collection used by the application."""
    chroma_directory = Path(settings.CHROMA_DIR)
    chroma_directory.mkdir(parents=True, exist_ok=True)

    provider, model = embedding_identity()
    selected_collection = collection_name or current_collection_name()
    runtime_marker = settings.OLLAMA_BASE_URL if provider == "ollama" else model
    credential_marker = hashlib.sha256(runtime_marker.encode("utf-8")).hexdigest()[:8]
    key = (
        str(chroma_directory.resolve()),
        selected_collection,
        provider,
        model,
        credential_marker,
    )
    with VECTOR_STORE_LOCK:
        if key not in VECTOR_STORES:
            VECTOR_STORES[key] = Chroma(
                collection_name=selected_collection,
                persist_directory=str(chroma_directory),
                embedding_function=get_embedding_model(),
            )
        return VECTOR_STORES[key]


def index_document(
    document_id: str,
    *,
    force: bool = False,
    progress_callback: IndexProgressCallback | None = None,
) -> dict[str, object]:
    """Chunk a PDF and add its chunks to the persistent vector store."""
    document_record = get_document(document_id)
    if document_record is None:
        raise AppError("Document metadata was not found", code="document_not_found", status_code=404)

    def extraction_progress(stage: str, progress: float) -> None:
        if progress_callback:
            progress_callback(stage, progress * 0.5)

    chunked_document = split_pdf_text(
        document_id,
        force_extraction=force,
        progress_callback=extraction_progress,
    )
    chunks = chunked_document["chunks"]
    if not chunks:
        raise AppError(
            "No extractable text was found. Enable OCR or upload a clearer PDF.",
            code="no_extractable_text",
            status_code=422,
        )

    provider, model = embedding_identity()
    collection_name = current_collection_name()
    fingerprint = _index_fingerprint(chunked_document, provider, model)
    if (
        not force
        and document_record.get("index_fingerprint") == fingerprint
        and document_record.get("vector_collection") == collection_name
        and int(document_record.get("chunk_count") or 0) > 0
    ):
        if progress_callback:
            progress_callback("ready", 1.0)
        return {
            "document_id": document_id,
            "chunk_count": document_record.get("chunk_count", len(chunks)),
            "collection": collection_name,
            "embedding_provider": provider,
            "embedding_model": model,
            "index_fingerprint": fingerprint,
            "skipped": True,
        }

    documents = [
        Document(
            page_content=chunk["text"],
            metadata={
                "chunk_id": chunk["chunk_id"],
                "document_id": chunk["document_id"],
                "page": chunk["page"],
                "page_chunk_index": chunk["page_chunk_index"],
                "filename": document_record.get("original_filename", ""),
                "section": chunk.get("section") or "",
                "extraction_method": chunk["extraction_method"],
                "ocr_confidence": (
                    chunk["ocr_confidence"]
                    if chunk["ocr_confidence"] is not None
                    else -1.0
                ),
                "handwritten": bool(chunk["handwritten"]),
                "content_type": str(chunk.get("content_type") or "text"),
                "table_index": int(chunk.get("table_index") or -1),
                "bbox_x0": _bbox_value(chunk.get("bbox"), 0),
                "bbox_y0": _bbox_value(chunk.get("bbox"), 1),
                "bbox_x1": _bbox_value(chunk.get("bbox"), 2),
                "bbox_y1": _bbox_value(chunk.get("bbox"), 3),
                "page_text_quality": float(chunk.get("text_quality") or -1.0),
                "image_coverage": float(chunk.get("image_coverage") or 0.0),
                "page_rotation": int(chunk.get("page_rotation") or 0),
            },
        )
        for chunk in chunks
    ]
    chunk_ids = [chunk["chunk_id"] for chunk in chunks]

    vector_store = get_vector_store(collection_name)
    old_collection = document_record.get("vector_collection")
    try:
        if old_collection and old_collection != collection_name:
            _delete_from_collection(str(old_collection), document_id)
        vector_store.delete(where={"document_id": document_id})

        batch_size = max(1, settings.EMBEDDING_BATCH_SIZE)
        for offset in range(0, len(documents), batch_size):
            batch_documents = documents[offset : offset + batch_size]
            batch_ids = chunk_ids[offset : offset + batch_size]
            vector_store.add_documents(documents=batch_documents, ids=batch_ids)
            if progress_callback:
                completed = min(offset + len(batch_documents), len(documents))
                progress_callback("embedding", 0.55 + (completed / len(documents)) * 0.4)
    except Exception as exc:
        try:
            vector_store.delete(where={"document_id": document_id})
        except Exception:
            pass
        if isinstance(exc, AppError):
            raise
        raise translate_embedding_error(exc) from exc

    return {
        "document_id": document_id,
        "chunk_count": len(chunk_ids),
        "collection": collection_name,
        "chroma_directory": str(Path(settings.CHROMA_DIR)),
        "embedding_provider": provider,
        "embedding_model": model,
        "index_fingerprint": fingerprint,
        "skipped": False,
    }


def delete_document_vectors(document_id: str) -> None:
    """Delete all indexed chunks belonging to one document."""
    record = get_document(document_id)
    collections = {current_collection_name()}
    if record and record.get("vector_collection"):
        collections.add(str(record["vector_collection"]))
    for collection in collections:
        _delete_from_collection(collection, document_id)


def clear_vector_store_cache() -> None:
    with VECTOR_STORE_LOCK:
        VECTOR_STORES.clear()


def _delete_from_collection(collection_name: str, document_id: str) -> None:
    try:
        Chroma(
            collection_name=collection_name,
            persist_directory=str(Path(settings.CHROMA_DIR)),
            embedding_function=None,
        ).delete(where={"document_id": document_id})
    except ValueError:
        # Chroma may report that a not-yet-created collection has no matching rows.
        return


def _index_fingerprint(
    chunked_document: dict[str, object],
    provider: str,
    model: str,
) -> str:
    payload = "|".join(
        [
            str(chunked_document.get("extraction_fingerprint") or ""),
            str(chunked_document.get("chunk_size") or ""),
            str(chunked_document.get("chunk_overlap") or ""),
            INDEX_SCHEMA_VERSION,
            provider,
            model,
        ]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _bbox_value(value: object, index: int) -> float:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return -1.0
    try:
        return float(value[index])
    except (TypeError, ValueError, IndexError):
        return -1.0
