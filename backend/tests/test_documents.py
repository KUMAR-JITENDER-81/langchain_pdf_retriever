from io import BytesIO

from fastapi.testclient import TestClient
from pypdf import PdfWriter

from app.core.config import settings
from app.main import app


def make_pdf() -> bytes:
    output = BytesIO()
    writer = PdfWriter()
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
