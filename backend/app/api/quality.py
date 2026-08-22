from fastapi import APIRouter, Depends, Query

from app.core.security import require_api_token
from app.models.request import FeedbackRequest
from app.models.response import APIResponse
from app.services.quality_service import quality_summary, recent_answer_runs, save_feedback


router = APIRouter(
    prefix="/quality",
    tags=["Quality"],
    dependencies=[Depends(require_api_token)],
)


@router.post("/feedback", response_model=APIResponse)
def answer_feedback(request: FeedbackRequest) -> APIResponse:
    feedback = save_feedback(
        request.answer_id,
        request.rating,
        request.reasons,
        request.comment,
    )
    return APIResponse(
        success=True,
        message="Feedback saved locally",
        data=feedback,
    )


@router.get("/summary", response_model=APIResponse)
def answer_quality_summary() -> APIResponse:
    return APIResponse(
        success=True,
        message="Answer quality summary retrieved",
        data=quality_summary(),
    )


@router.get("/runs", response_model=APIResponse)
def answer_quality_runs(
    limit: int = Query(default=20, ge=1, le=200),
) -> APIResponse:
    return APIResponse(
        success=True,
        message="Recent answer diagnostics retrieved",
        data={"runs": recent_answer_runs(limit)},
    )
