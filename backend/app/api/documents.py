from fastapi import APIRouter, Depends, Query
from fastapi.responses import FileResponse, Response

from app.core.errors import AppError
from app.core.security import require_api_token
from app.models.request import BulkIndexRequest, SearchRequest
from app.models.response import APIResponse
from app.rag.retriever import search_documents
from app.rag.splitter import split_pdf_text
from app.services.indexing_service import (
    cancel_document_index,
    cancel_index_job,
    submit_index_job,
    wait_for_index_job,
)
from app.services.metadata_service import get_document, get_index_job, get_latest_index_job
from app.services.pdf_service import (
    delete_pdf,
    extract_pdf_text,
    get_document_info,
    get_pdf_path,
    list_pdfs,
)
from app.services.preview_service import render_page_preview
from app.services.vector_service import delete_document_vectors

router = APIRouter(
    prefix="/documents",
    tags=["Documents"],
    dependencies=[Depends(require_api_token)],
)
@router.get(
    "/",
    response_model=APIResponse
)

def document_home():
    return APIResponse(
        success=True,
        message="Documents listed successfully",
        data={"documents": list_pdfs()},
    )


@router.post("/search", response_model=APIResponse)
def document_search(request: SearchRequest):
    search_results = search_documents(
        request.question,
        request.k,
        request.document_ids,
        hybrid=request.hybrid,
    )

    return APIResponse(
        success=True,
        message="Document search completed",
        data=search_results,
    )


@router.get("/index-jobs/{job_id}", response_model=APIResponse)
def index_job_status(job_id: str):
    job = get_index_job(job_id)
    if job is None:
        raise AppError("Index job not found", code="index_job_not_found", status_code=404)

    return APIResponse(
        success=True,
        message="Index job status retrieved",
        data=job,
    )


@router.post("/index-jobs/{job_id}/cancel", response_model=APIResponse)
def index_job_cancel(job_id: str):
    job = cancel_index_job(job_id)
    return APIResponse(
        success=True,
        message="Index cancellation requested",
        data={"job": job},
    )


@router.post("/index", response_model=APIResponse, status_code=202)
def documents_bulk_index(request: BulkIndexRequest):
    jobs: list[dict[str, object]] = []
    for document_id in request.document_ids:
        get_document_info(document_id)
        job, created = submit_index_job(document_id, force=request.force)
        jobs.append({"document_id": document_id, "job": job, "created": created})
    return APIResponse(
        success=True,
        message=f"Queued {sum(bool(item['created']) for item in jobs)} indexing job(s)",
        data={"jobs": jobs},
    )


@router.get("/{document_id}/status", response_model=APIResponse)
def document_status(document_id: str):
    document = get_document_info(document_id)
    return APIResponse(
        success=True,
        message="Document status retrieved",
        data={
            "document": document,
            "latest_job": get_latest_index_job(document_id),
        },
    )


@router.get("/{document_id}/file")
def document_file(document_id: str):
    document = get_document_info(document_id)
    return FileResponse(
        path=get_pdf_path(document_id),
        media_type="application/pdf",
        filename=str(document.get("filename") or f"{document_id}.pdf"),
        content_disposition_type="inline",
    )


@router.get("/{document_id}/pages/{page_number}/preview")
def document_page_preview(
    document_id: str,
    page_number: int,
    x0: float | None = Query(default=None),
    y0: float | None = Query(default=None),
    x1: float | None = Query(default=None),
    y1: float | None = Query(default=None),
) -> Response:
    coordinates = [x0, y0, x1, y1]
    if any(value is not None for value in coordinates) and not all(
        value is not None for value in coordinates
    ):
        raise AppError(
            "All four highlight coordinates are required",
            code="invalid_highlight",
            status_code=422,
        )
    preview, details = render_page_preview(
        document_id,
        page_number,
        bbox=[float(value) for value in coordinates if value is not None] or None,
    )
    return Response(
        content=preview,
        media_type="image/png",
        headers={
            "Cache-Control": "private, max-age=60",
            "X-PDF-Page": str(details["page"]),
            "X-Evidence-Highlighted": str(details["highlighted"]).lower(),
        },
    )


@router.get("/{document_id}/text", response_model=APIResponse)
def document_text(document_id: str, force: bool = False):
    extracted_document = extract_pdf_text(document_id, force=force)

    return APIResponse(
        success=True,
        message="PDF text extracted successfully",
        data=extracted_document,
    )


@router.get("/{document_id}/chunks", response_model=APIResponse)
def document_chunks(document_id: str):
    chunked_document = split_pdf_text(document_id)

    return APIResponse(
        success=True,
        message="PDF text chunked successfully",
        data=chunked_document,
    )


@router.post("/{document_id}/index", response_model=APIResponse, status_code=202)
def document_index(document_id: str, force: bool = False, wait: bool = False):
    get_document_info(document_id)
    job, created = submit_index_job(document_id, force=force)
    if wait:
        job = wait_for_index_job(job["job_id"])

    return APIResponse(
        success=True,
        message=(
            "Indexing job queued"
            if created
            else "An indexing job is already active for this document"
        ),
        data={"job": job, "created": created},
    )


@router.post("/{document_id}/index/cancel", response_model=APIResponse)
def document_index_cancel(document_id: str):
    job = cancel_document_index(document_id)
    return APIResponse(
        success=True,
        message="Index cancellation requested",
        data={"job": job},
    )


@router.delete("/{document_id}", response_model=APIResponse)
def document_delete(document_id: str):
    record = get_document(document_id)
    if record and record["status"] in {"queued", "processing"}:
        raise AppError(
            "Wait for indexing to finish before deleting this document",
            code="document_busy",
            status_code=409,
            retryable=True,
        )
    delete_document_vectors(document_id)
    delete_pdf(document_id)

    return APIResponse(
        success=True,
        message="Document deleted successfully",
        data={"document_id": document_id},
    )
