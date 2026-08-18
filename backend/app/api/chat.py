from fastapi import APIRouter, Depends, HTTPException

from app.core.security import require_api_token
from app.models.request import ChatRequest
from app.models.response import APIResponse
from app.rag.chain import answer_question
router = APIRouter(
    prefix="/chat",
    tags=["Chat"],
    dependencies=[Depends(require_api_token)],
)
@router.post(
    "/",
    response_model=APIResponse
)
def chat(
    request: ChatRequest
):
    try:
        result = answer_question(request.question, k=request.k)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail="Chat generation failed") from exc

    return APIResponse(
        success=True,
        message="Answer generated successfully",
        data=result,
    )
