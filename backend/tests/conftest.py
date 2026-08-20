import pytest

from app.core.config import settings


@pytest.fixture(autouse=True)
def isolate_runtime_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "UPLOAD_DIR", str(tmp_path / "uploads"))
    monkeypatch.setattr(settings, "CHROMA_DIR", str(tmp_path / "chroma"))
    monkeypatch.setattr(settings, "METADATA_DB", str(tmp_path / "data" / "documents.sqlite3"))
    monkeypatch.setattr(
        settings,
        "EXTRACTION_CACHE_DIR",
        str(tmp_path / "data" / "extractions"),
    )
    monkeypatch.setattr(settings, "OCR_ENABLED", False)
    monkeypatch.setattr(settings, "OPENAI_OCR_FALLBACK", False)
    monkeypatch.setattr(settings, "RATE_LIMIT_ENABLED", False)
