from fastapi import APIRouter
from app.models.request import ChatRequest
from app.models.response import APIResponse
router = APIRouter(
    prefix="/chat",
    tags=["Chat"]
)
@router.post(
    "/",
    response_model=APIResponse
)
def chat(
    request: ChatRequest
):
    return APIResponse(
        success=True,
        message="Question Received",
        data={
            "question": request.question
        }
    )