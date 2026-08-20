from fastapi import APIRouter, Depends, File, UploadFile

from app.models.response import APIResponse
from app.core.security import require_api_token
from app.services.pdf_service import save_pdf

router = APIRouter(
    prefix="/upload",
    tags=["Upload"],
    dependencies=[Depends(require_api_token)],
)
@router.post("/", response_model=APIResponse)
def upload_pdf_endpoint(file: UploadFile = File(...)):
    document = save_pdf(file)

    return APIResponse(
        success=True,
        message=(
            "This PDF was already uploaded"
            if document.get("duplicate")
            else "PDF uploaded successfully"
        ),
        data=document,
    )
