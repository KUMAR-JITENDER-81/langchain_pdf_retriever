from fastapi import APIRouter, HTTPException
from pypdf.errors import PdfReadError

from app.models.response import APIResponse
from app.rag.splitter import split_pdf_text
from app.services.pdf_service import extract_pdf_text

router = APIRouter(
    prefix="/documents",
    tags=["Documents"]
)
@router.get(
    "/",
    response_model=APIResponse
)

def document_home():
    return APIResponse(
        success=True,
        message="Documents API Working"
    )


@router.get("/{document_id}/text", response_model=APIResponse)
def document_text(document_id: str):
    try:
        extracted_document = extract_pdf_text(document_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except PdfReadError as exc:
        raise HTTPException(status_code=422, detail="The file is not a readable PDF") from exc
    except OSError as exc:
        raise HTTPException(status_code=500, detail="Could not read the PDF") from exc

    return APIResponse(
        success=True,
        message="PDF text extracted successfully",
        data=extracted_document,
    )


@router.get("/{document_id}/chunks", response_model=APIResponse)
def document_chunks(document_id: str):
    try:
        chunked_document = split_pdf_text(document_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except PdfReadError as exc:
        raise HTTPException(status_code=422, detail="The file is not a readable PDF") from exc
    except OSError as exc:
        raise HTTPException(status_code=500, detail="Could not read the PDF") from exc

    return APIResponse(
        success=True,
        message="PDF text chunked successfully",
        data=chunked_document,
    )
