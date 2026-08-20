from io import BytesIO

from fastapi.testclient import TestClient
import pymupdf

from app.core.config import settings
from app.main import app
from app.services import ocr_service
from app.services.ocr_service import OCRResult, tesseract_available


def make_scanned_pdf(text: str) -> bytes:
    source = pymupdf.open()
    source_page = source.new_page(width=800, height=240)
    source_page.insert_text((50, 125), text, fontsize=30)
    image = source_page.get_pixmap(dpi=200, alpha=False).tobytes("png")

    scanned = pymupdf.open()
    scanned_page = scanned.new_page(width=800, height=240)
    scanned_page.insert_image(scanned_page.rect, stream=image)
    payload = scanned.tobytes()
    source.close()
    scanned.close()
    return payload


def test_scanned_pdf_uses_tesseract_when_available(monkeypatch):
    if not tesseract_available():
        return

    monkeypatch.setattr(settings, "OCR_ENABLED", True)
    monkeypatch.setattr(settings, "OCR_PROVIDER", "tesseract")
    client = TestClient(app)

    upload = client.post(
        "/upload/",
        files={
            "file": (
                "scan.pdf",
                BytesIO(make_scanned_pdf("Invoice number 4829 total amount 1250")),
                "application/pdf",
            )
        },
    )
    document_id = upload.json()["data"]["document_id"]
    extracted = client.get(f"/documents/{document_id}/text")

    assert extracted.status_code == 200
    data = extracted.json()["data"]
    assert data["ocr_page_count"] == 1
    assert "4829" in data["text"]
    assert data["page_details"][0]["method"] == "tesseract"


def test_native_pdf_does_not_call_ocr(monkeypatch):
    monkeypatch.setattr(settings, "OCR_ENABLED", True)
    monkeypatch.setattr(settings, "OCR_PROVIDER", "auto")

    document = pymupdf.open()
    page = document.new_page()
    page.insert_text(
        (72, 100),
        "This native PDF contains enough searchable text that OCR should not be required.",
        fontsize=12,
    )
    payload = document.tobytes()
    document.close()

    client = TestClient(app)
    upload = client.post(
        "/upload/",
        files={"file": ("native.pdf", payload, "application/pdf")},
    )
    document_id = upload.json()["data"]["document_id"]
    extracted = client.get(f"/documents/{document_id}/text").json()["data"]

    assert extracted["native_page_count"] == 1
    assert extracted["ocr_pages_attempted"] == 0
    assert extracted["page_details"][0]["method"] == "native"


def test_handwriting_fallback_is_cached(monkeypatch):
    calls = {"count": 0}

    def fake_openai_ocr(page, page_number):
        calls["count"] += 1
        return OCRResult(
            text="Handwritten meeting note: approve project Cedar.",
            method="openai",
            confidence=0.82,
            handwritten=True,
        )

    monkeypatch.setattr(settings, "OCR_ENABLED", True)
    monkeypatch.setattr(settings, "OCR_PROVIDER", "openai")
    monkeypatch.setattr(ocr_service, "_run_openai_ocr", fake_openai_ocr)
    client = TestClient(app)
    upload = client.post(
        "/upload/",
        files={
            "file": (
                "handwriting.pdf",
                make_scanned_pdf("visual placeholder"),
                "application/pdf",
            )
        },
    )
    document_id = upload.json()["data"]["document_id"]

    first = client.get(f"/documents/{document_id}/text").json()["data"]
    second = client.get(f"/documents/{document_id}/text").json()["data"]

    assert first["handwritten_page_count"] == 1
    assert first["page_details"][0]["method"] == "openai"
    assert second["text"] == first["text"]
    assert calls["count"] == 1
