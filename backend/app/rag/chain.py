from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from app.core.config import settings
from app.rag.retriever import search_documents


SYSTEM_PROMPT = """You answer questions using only the provided document context.
If the context does not contain the answer, say that the answer was not found
in the uploaded documents. Do not invent facts. Cite supporting pages using
the format [Page N]."""


def answer_question(question: str, k: int = 4) -> dict[str, object]:
    """Retrieve relevant chunks and generate a grounded answer."""
    search_results = search_documents(question, k=k)
    results = search_results["results"]

    if not results:
        return {
            "answer": "I could not find relevant information in the uploaded documents.",
            "sources": [],
        }

    context = "\n\n".join(
        f"[Page {result['metadata'].get('page', 'unknown')}]\n{result['text']}"
        for result in results
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", "Document context:\n{context}\n\nQuestion: {question}"),
        ]
    )
    llm = ChatOpenAI(
        model=settings.MODEL_NAME,
        temperature=settings.TEMPERATURE,
        api_key=settings.OPENAI_API_KEY,
    )
    response = llm.invoke(
        prompt.format_messages(context=context, question=question)
    )

    return {
        "answer": str(response.content),
        "sources": [
            {
                "page": result["metadata"].get("page"),
                "document_id": result["metadata"].get("document_id"),
                "distance": result["distance"],
            }
            for result in results
        ],
    }
