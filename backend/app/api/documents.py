from fastapi import APIRouter

from app.models.response import APIResponse

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