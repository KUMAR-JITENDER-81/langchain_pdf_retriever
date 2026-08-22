import hashlib

from fastapi.testclient import TestClient
from langchain_core.embeddings import Embeddings
from langchain_core.messages import AIMessage, AIMessageChunk
import pymupdf

from app.core.config import settings
from app.main import app
from app.rag import chain
from app.services import vector_service


class DeterministicEmbeddings(Embeddings):
    def _vector(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        return [byte / 255 for byte in digest[:16]]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)


class StaticChatModel:
    def invoke(self, messages):
        return AIMessage(content="The project codename is ORCHID-4829 [Source 1].")

    def stream(self, messages):
        yield AIMessageChunk(content="The project codename is ")
        yield AIMessageChunk(content="ORCHID-4829 [Source 1].")


def make_pdf(text: str) -> bytes:
    document = pymupdf.open()
    page = document.new_page()
    page.insert_textbox(page.rect + (50, 50, -50, -50), text, fontsize=12)
    payload = document.tobytes()
    document.close()
    return payload


def configure_local_test_models(monkeypatch):
    monkeypatch.setattr(settings, "EMBEDDING_PROVIDER", "ollama")
    monkeypatch.setattr(settings, "OLLAMA_EMBEDDING_MODEL", "test-embedding")
    monkeypatch.setattr(settings, "MIN_RETRIEVAL_RELEVANCE", 0.0)
    monkeypatch.setattr(vector_service, "get_embedding_model", lambda: DeterministicEmbeddings())
    vector_service.clear_vector_store_cache()


def upload_and_index(client: TestClient, filename: str, text: str) -> str:
    upload = client.post(
        "/upload/",
        files={"file": (filename, make_pdf(text), "application/pdf")},
    )
    document_id = upload.json()["data"]["document_id"]
    indexed = client.post(f"/documents/{document_id}/index?wait=true")
    assert indexed.json()["data"]["job"]["status"] == "ready"
    return document_id


def test_search_is_scoped_to_selected_document(monkeypatch):
    configure_local_test_models(monkeypatch)
    client = TestClient(app)
    first_id = upload_and_index(
        client,
        "orchid.pdf",
        "The internal project codename is ORCHID-4829. The launch owner is Maya. " * 5,
    )
    upload_and_index(
        client,
        "falcon.pdf",
        "The separate project codename is FALCON-9931. The launch owner is Ravi. " * 5,
    )

    response = client.post(
        "/documents/search",
        json={
            "question": "What is the ORCHID-4829 project codename?",
            "document_ids": [first_id],
            "k": 5,
            "hybrid": True,
        },
    )
    results = response.json()["data"]["results"]

    assert response.status_code == 200
    assert results
    assert all(result["metadata"]["document_id"] == first_id for result in results)
    assert results[0]["metadata"]["filename"] == "orchid.pdf"


def test_chat_returns_rich_sources_without_external_generation(monkeypatch):
    configure_local_test_models(monkeypatch)
    monkeypatch.setattr(chain, "get_chat_model", lambda *args, **kwargs: StaticChatModel())
    client = TestClient(app)
    document_id = upload_and_index(
        client,
        "orchid.pdf",
        "The internal project codename is ORCHID-4829. The launch owner is Maya. " * 5,
    )

    response = client.post(
        "/chat/",
        json={
            "question": "What is the project codename?",
            "document_ids": [document_id],
            "mode": "balanced",
            "k": 5,
        },
    )
    data = response.json()["data"]

    assert response.status_code == 200
    assert "[Source 1]" in data["answer"]
    assert data["sources"][0]["filename"] == "orchid.pdf"
    assert data["sources"][0]["page"] == 1
    assert data["mode"] == "balanced"
    assert len(data["answer_id"]) == 32
    assert data["diagnostics"]["candidate_count"] >= 1
    assert data["diagnostics"]["total_ms"] >= data["diagnostics"]["retrieval_ms"]


def test_chat_works_with_builtin_local_answer_provider(monkeypatch):
    configure_local_test_models(monkeypatch)
    monkeypatch.setattr(settings, "GENERATION_PROVIDER", "local")
    monkeypatch.setattr(settings, "LOCAL_ANSWER_MODEL", "extractive-v1")
    client = TestClient(app)
    document_id = upload_and_index(
        client,
        "local-answer.pdf",
        "Object-oriented programming organizes software around objects and classes. "
        "Encapsulation keeps state together with the methods that manage it. " * 5,
    )

    response = client.post(
        "/chat/",
        json={
            "question": "What is object-oriented programming about?",
            "document_ids": [document_id],
            "mode": "balanced",
            "k": 5,
        },
    )
    data = response.json()["data"]

    assert response.status_code == 200
    assert "object" in data["answer"].lower()
    assert "[Source 1]" in data["answer"]
    assert any("entirely locally" in warning for warning in data["warnings"])


def test_chat_stream_emits_sources_tokens_and_done(monkeypatch):
    configure_local_test_models(monkeypatch)
    monkeypatch.setattr(chain, "get_chat_model", lambda *args, **kwargs: StaticChatModel())
    client = TestClient(app)
    document_id = upload_and_index(
        client,
        "stream.pdf",
        "The internal project codename is ORCHID-4829. The launch owner is Maya. " * 5,
    )

    response = client.post(
        "/chat/stream",
        json={
            "question": "What is the project codename?",
            "document_ids": [document_id],
            "mode": "quick",
            "k": 4,
        },
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert "event: sources" in response.text
    assert "event: token" in response.text
    assert "ORCHID-4829" in response.text
    assert "event: done" in response.text
    assert '"answer_id"' in response.text
    assert '"diagnostics"' in response.text


def test_repeated_question_uses_document_versioned_answer_cache(monkeypatch):
    configure_local_test_models(monkeypatch)
    monkeypatch.setattr(settings, "GENERATION_PROVIDER", "local")
    monkeypatch.setattr(settings, "QUICK_MODE_LOCAL", True)
    monkeypatch.setattr(settings, "ANSWER_CACHE_ENABLED", True)
    client = TestClient(app)
    document_id = upload_and_index(
        client,
        "cache.pdf",
        "The launch code is ORCHID-4829 and the launch owner is Maya. " * 6,
    )
    payload = {
        "question": "What is the launch code?",
        "document_ids": [document_id],
        "mode": "quick",
        "k": 4,
    }

    first = client.post("/chat/", json=payload).json()["data"]
    second = client.post("/chat/", json=payload).json()["data"]

    assert first["diagnostics"]["cache_hit"] is False
    assert second["diagnostics"]["cache_hit"] is True
    assert second["answer"] == first["answer"]
    assert second["answer_id"] != first["answer_id"]
