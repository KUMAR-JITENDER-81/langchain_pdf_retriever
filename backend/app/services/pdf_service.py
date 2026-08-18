from pathlib import Path
import re
from shutil import copyfileobj
from uuid import uuid4

from fastapi import UploadFile
from pypdf import PdfReader

from app.core.config import settings


def save_pdf(upload: UploadFile) -> dict[str, str]:
    """Validate and persist one uploaded PDF file."""
    original_name = Path(upload.filename or "").name

    if not original_name:
        raise ValueError("A filename is required")

    if Path(original_name).suffix.lower() != ".pdf":
        raise ValueError("Only PDF files are supported")

    upload_directory = Path(settings.UPLOAD_DIR)
    upload_directory.mkdir(parents=True, exist_ok=True)

    document_id = uuid4().hex
    stored_name = f"{document_id}.pdf"
    destination = upload_directory / stored_name

    with destination.open("wb") as output_file:
        copyfileobj(upload.file, output_file)

    return {
        "document_id": document_id,
        "filename": original_name,
        "stored_filename": stored_name,
        "path": str(destination),
    }


def _pdf_path(document_id: str) -> Path:
    """Resolve a stored PDF path from its generated document ID."""
    if not re.fullmatch(r"[0-9a-f]{32}", document_id):
        raise ValueError("Invalid document ID")

    upload_directory = Path(settings.UPLOAD_DIR).resolve()
    pdf_path = (upload_directory / f"{document_id}.pdf").resolve()

    if pdf_path.parent != upload_directory:
        raise ValueError("Invalid document path")
    if not pdf_path.is_file():
        raise FileNotFoundError("Document not found")

    return pdf_path


def extract_pdf_text(document_id: str) -> dict[str, object]:
    """Extract text from every page of a stored PDF."""
    pdf_path = _pdf_path(document_id)
    reader = PdfReader(str(pdf_path))

    pages = [page.extract_text() or "" for page in reader.pages]

    return {
        "document_id": document_id,
        "page_count": len(pages),
        "pages": pages,
        "text": "\n\n".join(pages),
    }


def list_pdfs() -> list[dict[str, object]]:
    """List stored PDFs and basic filesystem metadata."""
    upload_directory = Path(settings.UPLOAD_DIR)
    if not upload_directory.is_dir():
        return []

    documents = []
    for pdf_path in sorted(upload_directory.glob("*.pdf")):
        if not re.fullmatch(r"[0-9a-f]{32}", pdf_path.stem):
            continue

        file_info = pdf_path.stat()
        documents.append(
            {
                "document_id": pdf_path.stem,
                "stored_filename": pdf_path.name,
                "size_bytes": file_info.st_size,
                "modified_at": file_info.st_mtime,
            }
        )

    return documents


def delete_pdf(document_id: str) -> None:
    """Delete one stored PDF by its generated document ID."""
    _pdf_path(document_id).unlink()
