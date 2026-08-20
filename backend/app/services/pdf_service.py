from datetime import UTC, datetime
import hashlib
from pathlib import Path
import re
from uuid import uuid4

from fastapi import UploadFile
from pypdf import PdfReader
from pypdf.errors import PdfReadError

from app.core.config import settings
from app.core.errors import (
    AppError,
    DocumentLimitError,
    DocumentNotFoundError,
    InvalidPDFError,
    UploadTooLargeError,
)
from app.services.metadata_service import (
    create_document,
    delete_document_record,
    find_document_by_hash,
    get_document,
    list_document_records,
)

PDF_SIGNATURE = b"%PDF-"
COPY_BUFFER_SIZE = 1024 * 1024
ALLOWED_PDF_CONTENT_TYPES = {
    "",
    "application/octet-stream",
    "application/pdf",
    "application/x-pdf",
}


def save_pdf(upload: UploadFile) -> dict[str, object]:
    """Validate and persist one uploaded PDF file."""
    original_name = Path(upload.filename or "").name

    if not original_name:
        raise AppError("A filename is required", code="filename_required")

    if Path(original_name).suffix.lower() != ".pdf":
        raise AppError("Only PDF files are supported", code="unsupported_file_type")

    content_type = (upload.content_type or "").lower()
    if content_type not in ALLOWED_PDF_CONTENT_TYPES:
        raise AppError(
            "The uploaded file does not have a supported PDF content type",
            code="unsupported_content_type",
            status_code=415,
        )

    upload_directory = Path(settings.UPLOAD_DIR)
    upload_directory.mkdir(parents=True, exist_ok=True)

    document_id = uuid4().hex
    stored_name = f"{document_id}.pdf"
    destination = upload_directory / stored_name
    max_upload_bytes = int(settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024)

    try:
        upload.file.seek(0)
        header = upload.file.read(1024)
        if PDF_SIGNATURE not in header:
            raise InvalidPDFError("The uploaded file does not contain a valid PDF header")
        upload.file.seek(0)

        total_bytes = 0
        digest = hashlib.sha256()
        with destination.open("wb") as output_file:
            while chunk := upload.file.read(COPY_BUFFER_SIZE):
                total_bytes += len(chunk)
                if total_bytes > max_upload_bytes:
                    raise UploadTooLargeError(
                        f"PDF exceeds the {settings.MAX_UPLOAD_SIZE_MB:g} MB size limit"
                    )
                digest.update(chunk)
                output_file.write(chunk)

        try:
            reader = PdfReader(str(destination), strict=False)
            if reader.is_encrypted and reader.decrypt("") == 0:
                raise InvalidPDFError(
                    "Password-protected PDFs must be unlocked before upload"
                )
            page_count = len(reader.pages)
        except InvalidPDFError:
            raise
        except (PdfReadError, OSError, ValueError) as exc:
            raise InvalidPDFError() from exc

        if page_count == 0:
            raise InvalidPDFError("The PDF does not contain any pages")
        if page_count > settings.MAX_PDF_PAGES:
            raise DocumentLimitError(
                f"PDF has {page_count} pages; the limit is {settings.MAX_PDF_PAGES}"
            )

        sha256 = digest.hexdigest()
        duplicate = find_document_by_hash(sha256)
        if duplicate and (upload_directory / duplicate["stored_filename"]).is_file():
            destination.unlink(missing_ok=True)
            return {**_public_document(duplicate), "duplicate": True}

        existing_document_count = sum(
            1 for path in upload_directory.glob("*.pdf") if path != destination
        )
        if existing_document_count >= settings.MAX_TOTAL_DOCUMENTS:
            raise DocumentLimitError(
                f"The document limit of {settings.MAX_TOTAL_DOCUMENTS} has been reached"
            )

        document = create_document(
            {
                "document_id": document_id,
                "original_filename": original_name,
                "stored_filename": stored_name,
                "sha256": sha256,
                "size_bytes": total_bytes,
                "content_type": content_type or "application/pdf",
                "page_count": page_count,
            }
        )
    except Exception:
        destination.unlink(missing_ok=True)
        raise

    return {**_public_document(document), "duplicate": False}


def get_pdf_path(document_id: str) -> Path:
    """Resolve a stored PDF path from its generated document ID."""
    if not re.fullmatch(r"[0-9a-f]{32}", document_id):
        raise AppError("Invalid document ID", code="invalid_document_id")

    upload_directory = Path(settings.UPLOAD_DIR).resolve()
    pdf_path = (upload_directory / f"{document_id}.pdf").resolve()

    if pdf_path.parent != upload_directory:
        raise AppError("Invalid document path", code="invalid_document_id")
    if not pdf_path.is_file():
        raise DocumentNotFoundError()

    return pdf_path


def extract_pdf_text(
    document_id: str,
    *,
    force: bool = False,
    progress_callback=None,
) -> dict[str, object]:
    """Extract native and OCR text from every page of a stored PDF."""
    from app.services.extraction_service import extract_document

    return extract_document(
        document_id,
        force=force,
        progress_callback=progress_callback,
    )


def list_pdfs() -> list[dict[str, object]]:
    """List stored PDFs with persistent processing metadata."""
    upload_directory = Path(settings.UPLOAD_DIR)
    if not upload_directory.is_dir():
        return []

    known_documents = {
        record["document_id"]: record for record in list_document_records()
    }
    for pdf_path in sorted(upload_directory.glob("*.pdf")):
        if not re.fullmatch(r"[0-9a-f]{32}", pdf_path.stem):
            continue
        if pdf_path.stem not in known_documents:
            known_documents[pdf_path.stem] = _register_existing_pdf(pdf_path)

    return [
        _public_document(record)
        for record in sorted(
            known_documents.values(),
            key=lambda item: item.get("created_at") or "",
            reverse=True,
        )
        if (upload_directory / record["stored_filename"]).is_file()
    ]


def get_document_info(document_id: str) -> dict[str, object]:
    get_pdf_path(document_id)
    record = get_document(document_id)
    if record is None:
        pdf_path = get_pdf_path(document_id)
        record = _register_existing_pdf(pdf_path)
    return _public_document(record)


def delete_pdf(document_id: str) -> None:
    """Delete one stored PDF by its generated document ID."""
    get_pdf_path(document_id).unlink()
    from app.services.extraction_service import delete_extraction_cache

    delete_extraction_cache(document_id)
    delete_document_record(document_id)


def _register_existing_pdf(pdf_path: Path) -> dict[str, object]:
    """Register PDFs created by older application versions without metadata."""
    digest = hashlib.sha256()
    with pdf_path.open("rb") as pdf_file:
        while chunk := pdf_file.read(COPY_BUFFER_SIZE):
            digest.update(chunk)

    page_count: int | None = None
    status = "uploaded"
    try:
        page_count = len(PdfReader(str(pdf_path), strict=False).pages)
    except (PdfReadError, OSError, ValueError):
        status = "failed"

    file_info = pdf_path.stat()
    modified_at = datetime.fromtimestamp(file_info.st_mtime, UTC).isoformat()
    return create_document(
        {
            "document_id": pdf_path.stem,
            "original_filename": pdf_path.name,
            "stored_filename": pdf_path.name,
            "sha256": digest.hexdigest(),
            "size_bytes": file_info.st_size,
            "content_type": "application/pdf",
            "page_count": page_count,
            "status": status,
            "created_at": modified_at,
            "updated_at": modified_at,
        }
    )


def _public_document(record: dict[str, object]) -> dict[str, object]:
    """Return document fields that are safe and useful to API clients."""
    return {
        "document_id": record["document_id"],
        "filename": record.get("original_filename"),
        "stored_filename": record.get("stored_filename"),
        "size_bytes": record.get("size_bytes", 0),
        "page_count": record.get("page_count"),
        "status": record.get("status", "uploaded"),
        "stage": record.get("stage", "uploaded"),
        "progress": record.get("progress", 0),
        "extraction_method": record.get("extraction_method"),
        "native_page_count": record.get("native_page_count", 0),
        "ocr_page_count": record.get("ocr_page_count", 0),
        "character_count": record.get("character_count", 0),
        "chunk_count": record.get("chunk_count", 0),
        "embedding_provider": record.get("embedding_provider"),
        "embedding_model": record.get("embedding_model"),
        "error_code": record.get("error_code"),
        "error_message": record.get("error_message"),
        "created_at": record.get("created_at"),
        "updated_at": record.get("updated_at"),
        "indexed_at": record.get("indexed_at"),
    }
