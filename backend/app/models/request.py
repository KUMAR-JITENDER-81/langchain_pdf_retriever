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
    task: Literal["answer", "summary", "compare", "extract", "quiz", "translate"] = "answer"
    response_language: str = Field(default="English", min_length=2, max_length=40)
    history: list[ConversationMessage] = Field(default_factory=list, max_length=12)

    @field_validator("response_language")
    @classmethod
    def clean_response_language(cls, value: str) -> str:
        normalized = " ".join(value.split())
        if not normalized:
            raise ValueError("Response language cannot be empty")
        return normalized


class SearchRequest(RetrievalRequest):
    hybrid: bool = True


class FeedbackRequest(BaseModel):
    answer_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    rating: Literal["helpful", "not_helpful"]
    reasons: list[
        Literal[
            "incorrect",
            "missing_information",
            "wrong_source",
            "ocr_problem",
            "too_slow",
            "hard_to_read",
        ]
    ] = Field(default_factory=list, max_length=6)
    comment: str = Field(default="", max_length=1000)

    @field_validator("reasons")
    @classmethod
    def unique_reasons(cls, values: list[str]) -> list[str]:
        return list(dict.fromkeys(values))

    @field_validator("comment")
    @classmethod
    def clean_comment(cls, value: str) -> str:
        return " ".join(value.split())


class BulkIndexRequest(BaseModel):
    document_ids: list[str] = Field(min_length=1, max_length=100)
    force: bool = False

    @field_validator("document_ids")
    @classmethod
    def validate_bulk_document_ids(cls, values: list[str]) -> list[str]:
        unique: list[str] = []
        for value in values:
            normalized = value.strip().lower()
            if len(normalized) != 32 or any(
                character not in "0123456789abcdef" for character in normalized
            ):
                raise ValueError("Each document ID must be a 32-character hexadecimal value")
            if normalized not in unique:
                unique.append(normalized)
        return unique
