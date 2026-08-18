from langchain_openai import OpenAIEmbeddings

from app.core.config import settings


def get_embedding_model() -> OpenAIEmbeddings:
    """Create the configured embedding model."""
    return OpenAIEmbeddings(
        model=settings.EMBEDDING_MODEL,
        api_key=settings.OPENAI_API_KEY,
    )
