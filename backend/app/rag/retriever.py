from app.services.vector_service import get_vector_store


def search_documents(question: str, k: int = 4) -> dict[str, object]:
    """Return the closest indexed chunks for a user question."""
    if not question.strip():
        raise ValueError("Question cannot be empty")
    if k < 1 or k > 20:
        raise ValueError("Result count must be between 1 and 20")

    vector_store = get_vector_store()
    matches = vector_store.similarity_search_with_score(question, k=k)

    results = [
        {
            "text": document.page_content,
            "metadata": document.metadata,
            "distance": score,
        }
        for document, score in matches
    ]

    return {
        "question": question,
        "result_count": len(results),
        "results": results,
    }
