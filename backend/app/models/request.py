from pydantic import BaseModel
from pydantic import Field


class ChatRequest(BaseModel):
    question: str
    k: int = Field(default=4, ge=1, le=20)


class SearchRequest(BaseModel):
    question: str
    k: int = Field(default=4, ge=1, le=20)
