from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document

from app.core.config import settings
from app.rag.splitter import split_pdf_text
from app.services.embedding_service import get_embedding_model


COLLECTION_NAME = "pdf_documents"


def get_vector_store() -> Chroma:
    """Open the persistent Chroma collection used by the application."""
    chroma_directory = Path(settings.CHROMA_DIR)
    chroma_directory.mkdir(parents=True, exist_ok=True)

    return Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=str(chroma_directory),
        embedding_function=get_embedding_model(),
    )


def index_document(document_id: str) -> dict[str, object]:
    """Chunk a PDF and add its chunks to the persistent vector store."""
    chunked_document = split_pdf_text(document_id)
    chunks = chunked_document["chunks"]

    documents = [
        Document(
            page_content=chunk["text"],
            metadata={
                "document_id": chunk["document_id"],
                "page": chunk["page"],
                "page_chunk_index": chunk["page_chunk_index"],
            },
        )
        for chunk in chunks
    ]
    chunk_ids = [chunk["chunk_id"] for chunk in chunks]

    vector_store = get_vector_store()
    if chunk_ids:
        vector_store.delete(ids=chunk_ids)
        vector_store.add_documents(documents=documents, ids=chunk_ids)

    return {
        "document_id": document_id,
        "chunk_count": len(chunk_ids),
        "collection": COLLECTION_NAME,
        "chroma_directory": str(Path(settings.CHROMA_DIR)),
    }


def delete_document_vectors(document_id: str) -> None:
    """Delete all indexed chunks belonging to one document."""
    get_vector_store().delete(where={"document_id": document_id})
