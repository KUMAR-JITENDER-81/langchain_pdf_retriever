from fastapi import APIRouter, File, HTTPException, UploadFile

from app.models.response import APIResponse
from app.services.pdf_service import save_pdf

router = APIRouter(
    prefix="/upload",
    tags=["Upload"]
)
@router.post("/", response_model=APIResponse)
def upload_pdf_endpoint(file: UploadFile = File(...)):
    try:
        document = save_pdf(file)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except OSError as exc:
        raise HTTPException(status_code=500, detail="Could not save the uploaded PDF") from exc

    return APIResponse(
        success=True,
        message="PDF uploaded successfully",
        data=document,
    )
