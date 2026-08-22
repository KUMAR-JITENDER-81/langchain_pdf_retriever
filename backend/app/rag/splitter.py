import hashlib
import re
from typing import Any

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
    table_count = 0
    page_details = extracted_document.get("page_details", [])
    for page_number, page_text in enumerate(extracted_document["pages"], start=1):
        page_detail = page_details[page_number - 1] if page_details else {}
        tables = list(page_detail.get("tables") or [])
        table_count += len(tables)
        prose_text = str(page_text)
        for table in tables:
            markdown = str(table.get("markdown") or "")
            if markdown:
                prose_text = prose_text.replace(markdown, "", 1)

        page_chunks: list[dict[str, Any]] = []
        for text in splitter.split_text(prose_text.strip()):
            page_chunks.append(
                {
                    "text": text,
                    "content_type": "text",
                    "table_index": None,
                    "bbox": _matching_block_bbox(text, list(page_detail.get("blocks") or [])),
                }
            )
        for table_index, table in enumerate(tables, start=1):
            for text in splitter.split_text(str(table.get("markdown") or "").strip()):
                page_chunks.append(
                    {
                        "text": text,
                        "content_type": "table",
                        "table_index": table_index,
                        "bbox": _valid_bbox(table.get("bbox")),
                    }
                )

        for page_chunk_index, chunk in enumerate(page_chunks):
            text = str(chunk["text"])
            chunk_digest = hashlib.sha256(
                (
                    f"{document_id}:{page_number}:{page_chunk_index}:"
                    f"{chunk['content_type']}:{text}"
                ).encode("utf-8")
            ).hexdigest()[:16]
            chunks.append(
                {
                    "chunk_id": f"{document_id}-{chunk_digest}",
                    "document_id": document_id,
                    "page": page_number,
                    "page_chunk_index": page_chunk_index,
                    "text": text,
                    "content_type": chunk["content_type"],
                    "table_index": chunk["table_index"],
                    "bbox": chunk["bbox"],
                    "section": page_detail.get("heading"),
                    "extraction_method": page_detail.get("method", "native"),
                    "ocr_confidence": page_detail.get("ocr_confidence"),
                    "handwritten": page_detail.get("handwritten"),
                    "text_quality": page_detail.get("text_quality"),
                    "image_coverage": page_detail.get("image_coverage"),
                    "page_rotation": page_detail.get("rotation", 0),
                }
            )

    return {
        "document_id": document_id,
        "chunk_count": len(chunks),
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "chunk_unit": "tokens",
        "table_count": table_count,
        "chunks": chunks,
        "source_sha256": extracted_document.get("source_sha256"),
        "extraction_fingerprint": extracted_document.get("extraction_fingerprint"),
    }


def _matching_block_bbox(
    chunk_text: str,
    blocks: list[dict[str, Any]],
) -> list[float] | None:
    chunk_normalized = " ".join(chunk_text.casefold().split())
    chunk_tokens = set(re.findall(r"\w+", chunk_normalized, re.UNICODE))
    matched: list[list[float]] = []
    for block in blocks:
        block_text = " ".join(str(block.get("text") or "").casefold().split())
        if not block_text:
            continue
        block_tokens = set(re.findall(r"\w+", block_text, re.UNICODE))
        overlap = len(chunk_tokens & block_tokens)
        strong_overlap = overlap >= 2 and overlap / max(len(block_tokens), 1) >= 0.45
        if block_text in chunk_normalized or strong_overlap:
            bbox = _valid_bbox(block.get("bbox"))
            if bbox:
                matched.append(bbox)
    if not matched:
        return None
    return [
        round(min(item[0] for item in matched), 2),
        round(min(item[1] for item in matched), 2),
        round(max(item[2] for item in matched), 2),
        round(max(item[3] for item in matched), 2),
    ]


def _valid_bbox(value: Any) -> list[float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        bbox = [float(item) for item in value]
    except (TypeError, ValueError):
        return None
    if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
        return None
    return [round(item, 2) for item in bbox]
