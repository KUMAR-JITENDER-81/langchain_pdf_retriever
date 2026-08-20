import json

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from app.core.security import require_api_token
from app.models.request import ChatRequest
from app.models.response import APIResponse
from app.rag.chain import answer_question, prepare_answer, stream_prepared_answer
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
    result = answer_question(
        request.question,
        k=request.k,
        document_ids=request.document_ids,
        mode=request.mode,
        history=[message.model_dump() for message in request.history],
    )

    return APIResponse(
        success=True,
        message="Answer generated successfully",
        data=result,
    )


@router.post("/stream")
def chat_stream(request: ChatRequest):
    prepared = prepare_answer(
        request.question,
        k=request.k,
        document_ids=request.document_ids,
        mode=request.mode,
        history=[message.model_dump() for message in request.history],
    )

    def event_stream():
        for item in stream_prepared_answer(prepared):
            payload = json.dumps(item["data"], ensure_ascii=False)
            yield f"event: {item['event']}\ndata: {payload}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
        },
    )
