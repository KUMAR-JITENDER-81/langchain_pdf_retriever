from pathlib import Path
from shutil import copyfileobj
from uuid import uuid4

from fastapi import UploadFile

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
