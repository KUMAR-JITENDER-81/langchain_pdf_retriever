import hashlib

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.config import settings
from app.services.pdf_service import extract_pdf_text


DEFAULT_CHUNK_SIZE = settings.CHUNK_SIZE_TOKENS
DEFAULT_CHUNK_OVERLAP = settings.CHUNK_OVERLAP_TOKENS


def split_pdf_text(
    document_id: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    *,
    force_extraction: bool = False,
    progress_callback=None,
) -> dict[str, object]:
    """Split each extracted PDF page into overlapping chunks."""
    if chunk_size <= 0:
        raise ValueError("Chunk size must be greater than zero")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("Chunk overlap must be between zero and chunk size")

    extracted_document = extract_pdf_text(
        document_id,
        force=force_extraction,
        progress_callback=progress_callback,
    )
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks: list[dict[str, object]] = []
    page_details = extracted_document.get("page_details", [])
    for page_number, page_text in enumerate(extracted_document["pages"], start=1):
        page_detail = page_details[page_number - 1] if page_details else {}
        for page_chunk_index, text in enumerate(splitter.split_text(page_text)):
            chunk_digest = hashlib.sha256(
                f"{document_id}:{page_number}:{page_chunk_index}:{text}".encode("utf-8")
            ).hexdigest()[:16]
            chunks.append(
                {
                    "chunk_id": f"{document_id}-{chunk_digest}",
                    "document_id": document_id,
                    "page": page_number,
                    "page_chunk_index": page_chunk_index,
                    "text": text,
                    "section": page_detail.get("heading"),
                    "extraction_method": page_detail.get("method", "native"),
                    "ocr_confidence": page_detail.get("ocr_confidence"),
                    "handwritten": page_detail.get("handwritten"),
                }
            )

    return {
        "document_id": document_id,
        "chunk_count": len(chunks),
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "chunk_unit": "tokens",
        "chunks": chunks,
        "source_sha256": extracted_document.get("source_sha256"),
        "extraction_fingerprint": extracted_document.get("extraction_fingerprint"),
    }
