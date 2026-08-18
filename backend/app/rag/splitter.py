from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.services.pdf_service import extract_pdf_text


DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200


def split_pdf_text(
    document_id: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> dict[str, object]:
    """Split each extracted PDF page into overlapping chunks."""
    if chunk_size <= 0:
        raise ValueError("Chunk size must be greater than zero")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("Chunk overlap must be between zero and chunk size")

    extracted_document = extract_pdf_text(document_id)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )

    chunks: list[dict[str, object]] = []
    for page_number, page_text in enumerate(extracted_document["pages"], start=1):
        for page_chunk_index, text in enumerate(splitter.split_text(page_text)):
            chunks.append(
                {
                    "chunk_id": f"{document_id}-{len(chunks):04d}",
                    "document_id": document_id,
                    "page": page_number,
                    "page_chunk_index": page_chunk_index,
                    "text": text,
                }
            )

    return {
        "document_id": document_id,
        "chunk_count": len(chunks),
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "chunks": chunks,
    }
