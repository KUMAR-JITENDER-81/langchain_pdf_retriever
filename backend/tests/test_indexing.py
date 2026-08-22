import hashlib
from threading import Event

from fastapi.testclient import TestClient
import pymupdf
from langchain_core.embeddings import Embeddings

from app.core.config import settings
from app.core.errors import ProviderQuotaError
from app.main import app
from app.services import indexing_service, vector_service


class DeterministicEmbeddings(Embeddings):
    def _vector(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        return [byte / 255 for byte in digest[:16]]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)


def make_native_pdf(text: str) -> bytes:
    document = pymupdf.open()
    page = document.new_page()
    page.insert_textbox(page.rect + (50, 50, -50, -50), text, fontsize=12)
    payload = document.tobytes()
    document.close()
    return payload


def test_background_index_job_reaches_ready(monkeypatch):
    monkeypatch.setattr(settings, "EMBEDDING_PROVIDER", "ollama")
    monkeypatch.setattr(settings, "OLLAMA_EMBEDDING_MODEL", "test-embedding")
    monkeypatch.setattr(vector_service, "get_embedding_model", lambda: DeterministicEmbeddings())
    vector_service.clear_vector_store_cache()

    client = TestClient(app)
    upload = client.post(
        "/upload/",
        files={
            "file": (
                "guide.pdf",
                make_native_pdf(
                    "Background indexing stores this document in deterministic vectors. " * 8
                ),
                "application/pdf",
            )
        },
    )
    document_id = upload.json()["data"]["document_id"]

    response = client.post(f"/documents/{document_id}/index?wait=true")
    job = response.json()["data"]["job"]
    status = client.get(f"/documents/{document_id}/status").json()["data"]

    assert response.status_code == 202
    assert job["status"] == "ready"
    assert status["document"]["status"] == "ready"
    assert status["document"]["chunk_count"] > 0
    assert status["latest_job"]["job_id"] == job["job_id"]


def test_background_job_exposes_provider_quota_error(monkeypatch):
    def fail_index(*args, **kwargs):
        raise ProviderQuotaError("Embedding quota is unavailable")

    monkeypatch.setattr(indexing_service, "index_document", fail_index)
    client = TestClient(app)
    upload = client.post(
        "/upload/",
        files={
            "file": (
                "quota.pdf",
                make_native_pdf("A document used to verify structured provider errors." * 3),
                "application/pdf",
            )
        },
    )
    document_id = upload.json()["data"]["document_id"]

    response = client.post(f"/documents/{document_id}/index?wait=true")
    job = response.json()["data"]["job"]

    assert job["status"] == "failed"
    assert job["error_code"] == "provider_quota_exceeded"
    assert "quota" in job["error_message"].lower()


def test_running_index_job_can_be_cancelled(monkeypatch):
    started = Event()
    continue_work = Event()

    def slow_index(document_id, force=False, progress_callback=None):
        started.set()
        assert continue_work.wait(timeout=3)
        if progress_callback:
            progress_callback("extracting_text", 0.5)
        return {
            "chunk_count": 1,
            "embedding_provider": "local",
            "embedding_model": "test",
            "collection": "test",
            "index_fingerprint": "test",
        }

    monkeypatch.setattr(indexing_service, "index_document", slow_index)
    client = TestClient(app)
    upload = client.post(
        "/upload/",
        files={
            "file": (
                "cancel.pdf",
                make_native_pdf("A document whose indexing job will be cancelled." * 3),
                "application/pdf",
            )
        },
    )
    document_id = upload.json()["data"]["document_id"]
    queued = client.post(f"/documents/{document_id}/index").json()["data"]["job"]
    assert started.wait(timeout=3)

    cancellation = client.post(f"/documents/{document_id}/index/cancel")
    continue_work.set()
    final_job = indexing_service.wait_for_index_job(queued["job_id"], timeout=3)

    assert cancellation.status_code == 200
    assert final_job["status"] == "cancelled"
    status = client.get(f"/documents/{document_id}/status").json()["data"]["document"]
    assert status["status"] == "uploaded"
