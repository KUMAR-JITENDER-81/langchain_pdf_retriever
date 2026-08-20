from io import BytesIO

from fastapi.testclient import TestClient
from pypdf import PdfWriter

from app.core.config import settings
from app.main import app


def make_pdf(page_count: int = 1) -> bytes:
    output = BytesIO()
    writer = PdfWriter()
    for _ in range(page_count):
        writer.add_blank_page(width=300, height=300)
    writer.write(output)
    return output.getvalue()


def test_document_lifecycle(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "UPLOAD_DIR", str(tmp_path))
    client = TestClient(app)

    upload_response = client.post(
        "/upload/",
        files={"file": ("sample.pdf", make_pdf(), "application/pdf")},
    )

    assert upload_response.status_code == 200
    document = upload_response.json()["data"]
    document_id = document["document_id"]

    text_response = client.get(f"/documents/{document_id}/text")
    assert text_response.status_code == 200
    assert text_response.json()["data"]["page_count"] == 1

    list_response = client.get("/documents/")
    assert list_response.status_code == 200
    assert list_response.json()["data"]["documents"][0]["document_id"] == document_id

    delete_response = client.delete(f"/documents/{document_id}")
    assert delete_response.status_code == 200
    assert not (tmp_path / f"{document_id}.pdf").exists()


def test_upload_rejects_non_pdf(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "UPLOAD_DIR", str(tmp_path))
    client = TestClient(app)

    response = client.post(
        "/upload/",
        files={"file": ("notes.txt", b"not a PDF", "text/plain")},
    )

    assert response.status_code == 400
    assert "PDF" in response.json()["detail"]


def test_api_token_is_required_when_configured(monkeypatch):
    monkeypatch.setattr(settings, "API_AUTH_TOKEN", "test-token")
    client = TestClient(app)

    unauthorized = client.get("/documents/")
    authorized = client.get(
        "/documents/",
        headers={"Authorization": "Bearer test-token"},
    )

    assert unauthorized.status_code == 401
    assert authorized.status_code == 200


def test_duplicate_upload_reuses_existing_document():
    client = TestClient(app)
    payload = make_pdf()

    first = client.post(
        "/upload/",
        files={"file": ("first.pdf", payload, "application/pdf")},
    )
    second = client.post(
        "/upload/",
        files={"file": ("same-content.pdf", payload, "application/pdf")},
    )

    assert first.status_code == 200
    assert second.status_code == 200
    assert second.json()["data"]["duplicate"] is True
    assert second.json()["data"]["document_id"] == first.json()["data"]["document_id"]
    assert len(client.get("/documents/").json()["data"]["documents"]) == 1


def test_upload_size_limit_returns_413(monkeypatch):
    monkeypatch.setattr(settings, "MAX_UPLOAD_SIZE_MB", 0.00001)
    client = TestClient(app)

    response = client.post(
        "/upload/",
        files={"file": ("large.pdf", make_pdf(), "application/pdf")},
    )

    assert response.status_code == 413
    assert response.json()["error"]["code"] == "upload_too_large"


def test_page_limit_returns_413(monkeypatch):
    monkeypatch.setattr(settings, "MAX_PDF_PAGES", 1)
    client = TestClient(app)

    response = client.post(
        "/upload/",
        files={"file": ("two-pages.pdf", make_pdf(2), "application/pdf")},
    )

    assert response.status_code == 413
    assert response.json()["error"]["code"] == "document_limit_exceeded"


def test_health_has_request_and_security_headers():
    client = TestClient(app)
    response = client.get("/health", headers={"X-Request-ID": "test-request-123"})

    assert response.status_code == 200
    assert response.headers["x-request-id"] == "test-request-123"
    assert response.headers["x-content-type-options"] == "nosniff"
