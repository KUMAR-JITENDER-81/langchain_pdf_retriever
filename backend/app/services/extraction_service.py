from __future__ import annotations

from collections.abc import Callable
import hashlib
import json
from pathlib import Path
from typing import Any

import pymupdf

from app.core.config import settings
from app.core.errors import AppError, DocumentLimitError, InvalidPDFError
from app.services.metadata_service import get_document, update_document
from app.services.ocr_service import (
    meaningful_character_count,
    needs_ocr,
    normalize_text,
    run_ocr,
)


EXTRACTOR_VERSION = "hybrid-v1"
ProgressCallback = Callable[[str, float], None]


def extract_document(
    document_id: str,
    *,
    force: bool = False,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Extract native text and selectively OCR pages that have no usable text."""
    # Imported lazily to keep PDF persistence independent from extraction providers.
    from app.services.pdf_service import get_pdf_path

    pdf_path = get_pdf_path(document_id)
    document_record = get_document(document_id)
    source_sha256 = (
        str(document_record.get("sha256"))
        if document_record and document_record.get("sha256")
        else _sha256_file(pdf_path)
    )
    fingerprint = _extraction_fingerprint(source_sha256)

    if not force:
        cached = _read_cache(document_id, fingerprint)
        if cached is not None:
            if progress_callback:
                progress_callback("extracted", 1.0)
            return cached

    try:
        pdf = pymupdf.open(pdf_path)
    except (pymupdf.FileDataError, RuntimeError, ValueError) as exc:
        raise InvalidPDFError() from exc

    try:
        if pdf.needs_pass:
            raise InvalidPDFError(
                "Password-protected PDFs must be unlocked before processing"
            )
        page_count = pdf.page_count
        if page_count > settings.MAX_PDF_PAGES:
            raise DocumentLimitError(
                f"PDF has {page_count} pages; the limit is {settings.MAX_PDF_PAGES}"
            )

        analyses: list[dict[str, Any]] = []
        native_character_total = 0
        for page_index in range(page_count):
            page = pdf.load_page(page_index)
            native_text = normalize_text(page.get_text("text", sort=True))
            if len(native_text) > settings.MAX_PAGE_CHARACTERS:
                raise DocumentLimitError(
                    f"Page {page_index + 1} exceeds the extracted-text safety limit"
                )
            native_character_total += len(native_text)
            if native_character_total > settings.MAX_EXTRACTED_CHARACTERS:
                raise DocumentLimitError(
                    "The PDF exceeds the total extracted-text safety limit"
                )
            should_ocr, reason = needs_ocr(page, native_text)
            analyses.append(
                {
                    "page": page_index + 1,
                    "native_text": native_text,
                    "needs_ocr": should_ocr,
                    "ocr_reason": reason,
                    "native_blocks": _native_blocks(page),
                    "tables": _extract_tables(page) if settings.EXTRACT_TABLES else [],
                }
            )
            if progress_callback:
                progress_callback("extracting_native_text", (page_index + 1) / max(page_count, 1) * 0.3)

        ocr_page_count_requested = sum(bool(item["needs_ocr"]) for item in analyses)
        if ocr_page_count_requested > settings.OCR_MAX_PAGES:
            raise DocumentLimitError(
                f"PDF needs OCR on {ocr_page_count_requested} pages; "
                f"the OCR limit is {settings.OCR_MAX_PAGES}"
            )

        page_details: list[dict[str, Any]] = []
        warnings: list[str] = []
        native_page_count = 0
        ocr_page_count = 0
        handwritten_page_count = 0
        attempted_ocr_pages = 0
        final_character_total = 0

        for page_index, analysis in enumerate(analyses):
            native_text = analysis["native_text"]
            final_text = native_text
            method = "native" if native_text else "empty"
            confidence: float | None = None
            handwritten: bool | None = None
            page_warnings: list[str] = []
            blocks = analysis["native_blocks"]
            tables = analysis["tables"]

            if analysis["needs_ocr"]:
                attempted_ocr_pages += 1
                try:
                    result = run_ocr(pdf.load_page(page_index), page_index + 1)
                    page_warnings.extend(result.warnings)
                    if _prefer_ocr_text(result.text, native_text):
                        final_text = result.text
                        method = result.method
                        confidence = result.confidence
                        handwritten = result.handwritten
                        blocks = []
                        if result.text:
                            ocr_page_count += 1
                        if result.handwritten:
                            handwritten_page_count += 1
                    elif result.text:
                        page_warnings.append(
                            "OCR text was lower quality than the native text and was not used"
                        )
                except AppError as exc:
                    page_warnings.append(exc.message)
                    if settings.OCR_PROVIDER.lower() in {"openai", "tesseract"}:
                        raise

            if final_text and method == "native":
                native_page_count += 1

            if tables:
                table_text = "\n\n".join(table["markdown"] for table in tables)
                if table_text and table_text not in final_text:
                    final_text = f"{final_text}\n\n{table_text}".strip()

            if len(final_text) > settings.MAX_PAGE_CHARACTERS:
                raise DocumentLimitError(
                    f"Page {page_index + 1} exceeds the extracted-text safety limit"
                )
            final_character_total += len(final_text)
            if final_character_total > settings.MAX_EXTRACTED_CHARACTERS:
                raise DocumentLimitError(
                    "The PDF exceeds the total extracted-text safety limit"
                )

            page_details.append(
                {
                    "page": page_index + 1,
                    "text": final_text,
                    "method": method,
                    "native_character_count": len(native_text),
                    "character_count": len(final_text),
                    "ocr_reason": analysis["ocr_reason"],
                    "ocr_confidence": confidence,
                    "handwritten": handwritten,
                    "heading": _page_heading(final_text),
                    "blocks": blocks,
                    "tables": tables,
                    "warnings": page_warnings,
                }
            )
            warnings.extend(
                f"Page {page_index + 1}: {warning}" for warning in page_warnings
            )
            if progress_callback:
                progress_callback(
                    "ocr" if analysis["needs_ocr"] else "extracting_text",
                    0.3 + (page_index + 1) / max(page_count, 1) * 0.7,
                )

        pages = [str(page["text"]) for page in page_details]
        combined_text = "\n\n".join(text for text in pages if text).strip()
        extraction_method = _document_method(native_page_count, ocr_page_count, page_count)
        extracted = {
            "document_id": document_id,
            "page_count": page_count,
            "pages": pages,
            "page_details": page_details,
            "text": combined_text,
            "character_count": len(combined_text),
            "extraction_method": extraction_method,
            "native_page_count": native_page_count,
            "ocr_page_count": ocr_page_count,
            "ocr_pages_attempted": attempted_ocr_pages,
            "handwritten_page_count": handwritten_page_count,
            "warnings": warnings,
            "source_sha256": source_sha256,
            "extraction_fingerprint": fingerprint,
        }
    finally:
        pdf.close()

    _write_cache(document_id, extracted)
    if document_record:
        next_status = (
            document_record["status"]
            if document_record["status"] in {"ready", "processing", "queued"}
            else "extracted"
        )
        update_document(
            document_id,
            status=next_status,
            stage="extracted" if next_status == "extracted" else document_record["stage"],
            progress=1.0 if next_status == "extracted" else document_record["progress"],
            page_count=page_count,
            extraction_method=extraction_method,
            native_page_count=native_page_count,
            ocr_page_count=ocr_page_count,
            character_count=len(combined_text),
            error_code=None,
            error_message=None,
        )
    return extracted


def delete_extraction_cache(document_id: str) -> None:
    _cache_path(document_id).unlink(missing_ok=True)


def _prefer_ocr_text(ocr_text: str, native_text: str) -> bool:
    ocr_characters = meaningful_character_count(ocr_text)
    native_characters = meaningful_character_count(native_text)
    if ocr_characters < 5:
        return False
    return native_characters < settings.OCR_MIN_TEXT_CHARACTERS or ocr_characters >= int(
        native_characters * 0.7
    )


def _native_blocks(page: pymupdf.Page) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    for block in page.get_text("blocks", sort=True):
        if len(block) < 7 or int(block[6]) != 0:
            continue
        text = normalize_text(str(block[4]))
        if not text:
            continue
        blocks.append(
            {
                "bbox": [round(float(value), 2) for value in block[:4]],
                "text": text,
            }
        )
    return blocks


def _extract_tables(page: pymupdf.Page) -> list[dict[str, Any]]:
    tables: list[dict[str, Any]] = []
    try:
        detected_tables = page.find_tables()
    except (AttributeError, RuntimeError, ValueError):
        return tables

    for table in detected_tables.tables:
        rows = table.extract()
        if not rows or not any(any(cell for cell in row) for row in rows):
            continue
        normalized_rows = [
            [normalize_text(str(cell or "")).replace("|", "\\|") for cell in row]
            for row in rows
        ]
        column_count = max(len(row) for row in normalized_rows)
        normalized_rows = [
            row + [""] * (column_count - len(row)) for row in normalized_rows
        ]
        header = normalized_rows[0]
        body = normalized_rows[1:]
        markdown_lines = [
            "| " + " | ".join(header) + " |",
            "| " + " | ".join(["---"] * column_count) + " |",
            *("| " + " | ".join(row) + " |" for row in body),
        ]
        tables.append(
            {
                "bbox": [round(float(value), 2) for value in table.bbox],
                "markdown": "[Table]\n" + "\n".join(markdown_lines),
                "row_count": len(normalized_rows),
                "column_count": column_count,
            }
        )
    return tables


def _page_heading(text: str) -> str | None:
    for line in text.splitlines():
        candidate = line.strip().strip("#")
        if candidate and len(candidate) <= 160:
            return candidate
    return None


def _document_method(native_pages: int, ocr_pages: int, total_pages: int) -> str:
    if ocr_pages and native_pages:
        return "hybrid"
    if ocr_pages:
        return "ocr"
    if native_pages:
        return "native"
    return "empty" if total_pages else "none"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _extraction_fingerprint(source_sha256: str) -> str:
    payload = "|".join(
        [
            EXTRACTOR_VERSION,
            source_sha256,
            str(settings.OCR_ENABLED),
            settings.OCR_PROVIDER,
            settings.OCR_LANGUAGES,
            str(settings.OCR_DPI),
            str(settings.OCR_MIN_TEXT_CHARACTERS),
            str(settings.OPENAI_OCR_FALLBACK),
            settings.OPENAI_OCR_MODEL,
            str(settings.EXTRACT_TABLES),
        ]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _cache_path(document_id: str) -> Path:
    directory = Path(settings.EXTRACTION_CACHE_DIR)
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{document_id}.json"


def _read_cache(document_id: str, fingerprint: str) -> dict[str, Any] | None:
    path = _cache_path(document_id)
    if not path.is_file():
        return None
    try:
        cached = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if cached.get("extraction_fingerprint") != fingerprint:
        return None
    return cached


def _write_cache(document_id: str, extracted: dict[str, Any]) -> None:
    path = _cache_path(document_id)
    temporary_path = path.with_suffix(".tmp")
    temporary_path.write_text(
        json.dumps(extracted, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    temporary_path.replace(path)
