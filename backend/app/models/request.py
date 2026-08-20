from typing import Literal

from pydantic import BaseModel, Field, field_validator

from app.core.config import settings


class ConversationMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=8000)


class RetrievalRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    k: int = Field(
        default=settings.DEFAULT_RETRIEVAL_K,
        ge=1,
        le=settings.MAX_RETRIEVAL_K,
    )
    document_ids: list[str] = Field(default_factory=list, max_length=20)

    @field_validator("question")
    @classmethod
    def normalize_question(cls, value: str) -> str:
        normalized = " ".join(value.split())
        if not normalized:
            raise ValueError("Question cannot be empty")
        return normalized

    @field_validator("document_ids")
    @classmethod
    def validate_document_ids(cls, values: list[str]) -> list[str]:
        unique: list[str] = []
        for value in values:
            normalized = value.strip().lower()
            if len(normalized) != 32 or any(character not in "0123456789abcdef" for character in normalized):
                raise ValueError("Each document ID must be a 32-character hexadecimal value")
            if normalized not in unique:
                unique.append(normalized)
        return unique


class ChatRequest(RetrievalRequest):
    mode: Literal["quick", "balanced", "deep"] = "balanced"
    history: list[ConversationMessage] = Field(default_factory=list, max_length=12)


class SearchRequest(RetrievalRequest):
    hybrid: bool = True
