from __future__ import annotations

from dataclasses import dataclass, field
import base64
import json
import math
import os
from pathlib import Path
import re
import shutil
from typing import Any
import unicodedata

import pymupdf
from openai import APIConnectionError, APIStatusError, OpenAI, RateLimitError

from app.core.config import settings
from app.core.errors import ConfigurationError, ProviderQuotaError, ProviderUnavailableError


@dataclass(slots=True)
class OCRResult:
    text: str
    method: str
    confidence: float | None = None
    handwritten: bool | None = None
    warnings: list[str] = field(default_factory=list)


def normalize_text(text: str) -> str:
    """Normalize OCR/native text without destroying paragraph boundaries."""
    normalized = unicodedata.normalize("NFKC", text or "").replace("\x00", "")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in normalized.splitlines()]
    output: list[str] = []
    previous_blank = False
    for line in lines:
        is_blank = not line
        if is_blank and previous_blank:
            continue
        output.append(line)
        previous_blank = is_blank
    return "\n".join(output).strip()


def meaningful_character_count(text: str) -> int:
    return sum(character.isalnum() for character in text)


def page_image_coverage(page: pymupdf.Page) -> float:
    """Estimate how much of a page is occupied by raster images."""
    page_area = max(float(page.rect.width * page.rect.height), 1.0)
    covered_area = 0.0
    try:
        for image in page.get_image_info():
            rectangle = pymupdf.Rect(image.get("bbox", page.rect)) & page.rect
            if not rectangle.is_empty:
                covered_area += float(rectangle.width * rectangle.height)
    except (RuntimeError, ValueError):
        return 0.0
    return min(covered_area / page_area, 1.0)


def needs_ocr(page: pymupdf.Page, native_text: str) -> tuple[bool, str | None]:
    if not settings.OCR_ENABLED or settings.OCR_PROVIDER.lower() == "disabled":
        return False, None

    normalized = normalize_text(native_text)
    meaningful = meaningful_character_count(normalized)
    if meaningful < settings.OCR_MIN_TEXT_CHARACTERS:
        return True, "insufficient_native_text"

    replacement_ratio = normalized.count("\ufffd") / max(len(normalized), 1)
    if replacement_ratio > 0.05:
        return True, "damaged_text_encoding"

    if page_image_coverage(page) >= 0.6 and meaningful < 200:
        return True, "image_dominant_page"

    return False, None


def run_ocr(page: pymupdf.Page, page_number: int) -> OCRResult:
    provider = settings.OCR_PROVIDER.strip().lower()
    if not settings.OCR_ENABLED or provider == "disabled":
        return OCRResult("", "disabled", warnings=["OCR is disabled"])

    if provider not in {"auto", "tesseract", "openai"}:
        raise ConfigurationError(
            "OCR_PROVIDER must be one of: auto, tesseract, openai, disabled"
        )

    warnings: list[str] = []
    if provider in {"auto", "tesseract"}:
        try:
            local_result = _run_tesseract_ocr(page)
            if meaningful_character_count(local_result.text) >= 5:
                return local_result
            warnings.extend(local_result.warnings)
        except (ConfigurationError, ProviderUnavailableError) as exc:
            if provider == "tesseract":
                raise
            warnings.append(str(exc))

    if provider == "openai" or (provider == "auto" and settings.OPENAI_OCR_FALLBACK):
        result = _run_openai_ocr(page, page_number)
        result.warnings = warnings + result.warnings
        return result

    warnings.append(
        "No OCR provider was available; install Tesseract or enable OPENAI_OCR_FALLBACK"
    )
    return OCRResult("", "unavailable", warnings=warnings)


def _run_tesseract_ocr(page: pymupdf.Page) -> OCRResult:
    tessdata_path = resolve_tessdata_path()
    if tessdata_path is None:
        raise ConfigurationError(
            "Tesseract language data was not found; configure TESSERACT_DATA_PATH"
        )

    try:
        text_page = page.get_textpage_ocr(
            language=settings.OCR_LANGUAGES,
            dpi=settings.OCR_DPI,
            full=True,
            tessdata=str(tessdata_path),
        )
        text = normalize_text(page.get_text("text", textpage=text_page, sort=True))
    except RuntimeError as exc:
        raise ProviderUnavailableError(
            "Tesseract could not OCR this page",
            code="tesseract_failed",
        ) from exc

    warnings = [] if text else ["Tesseract did not recognize text on this page"]
    return OCRResult(text=text, method="tesseract", warnings=warnings)


def resolve_tessdata_path() -> Path | None:
    candidates: list[Path] = []
    if settings.TESSERACT_DATA_PATH:
        candidates.append(Path(settings.TESSERACT_DATA_PATH))
    if os.environ.get("TESSDATA_PREFIX"):
        candidates.append(Path(os.environ["TESSDATA_PREFIX"]))

    executable = shutil.which("tesseract")
    if executable:
        candidates.append(Path(executable).parent / "tessdata")

    candidates.extend(
        [
            Path("C:/Program Files/Tesseract-OCR/tessdata"),
            Path("C:/Program Files (x86)/Tesseract-OCR/tessdata"),
            Path("/usr/share/tesseract-ocr/5/tessdata"),
            Path("/usr/share/tesseract-ocr/4.00/tessdata"),
            Path("/usr/share/tessdata"),
        ]
    )

    primary_language = settings.OCR_LANGUAGES.split("+")[0]
    for candidate in candidates:
        if candidate.is_dir() and (candidate / f"{primary_language}.traineddata").is_file():
            return candidate
    return None


def tesseract_available() -> bool:
    return resolve_tessdata_path() is not None


def _render_page_as_data_url(page: pymupdf.Page) -> str:
    dpi = max(72, settings.OCR_DPI)
    width = float(page.rect.width) / 72 * dpi
    height = float(page.rect.height) / 72 * dpi
    pixel_count = max(width * height, 1)
    if pixel_count > settings.OCR_MAX_IMAGE_PIXELS:
        dpi = max(
            72,
            int(dpi * math.sqrt(settings.OCR_MAX_IMAGE_PIXELS / pixel_count)),
        )

    pixmap = page.get_pixmap(dpi=dpi, colorspace=pymupdf.csRGB, alpha=False)
    encoded = base64.b64encode(pixmap.tobytes("png")).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _run_openai_ocr(page: pymupdf.Page, page_number: int) -> OCRResult:
    if not settings.OPENAI_API_KEY:
        raise ConfigurationError(
            "OPENAI_API_KEY is required when OpenAI handwriting OCR is enabled"
        )

    client = OpenAI(api_key=settings.OPENAI_API_KEY, max_retries=2, timeout=90)
    schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "text": {"type": "string"},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "handwritten": {"type": "boolean"},
            "warnings": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["text", "confidence", "handwritten", "warnings"],
        "additionalProperties": False,
    }
    instructions = (
        "Transcribe every visible word on this PDF page exactly. Preserve reading order "
        "and line breaks. Do not summarize, correct, complete, or infer missing content. "
        "Use [illegible] where text cannot be read. Return confidence as a conservative "
        "0-to-1 estimate and identify whether meaningful handwriting is present."
    )

    try:
        response = client.responses.create(
            model=settings.OPENAI_OCR_MODEL,
            input=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": f"OCR PDF page {page_number}. {instructions}",
                        },
                        {
                            "type": "input_image",
                            "image_url": _render_page_as_data_url(page),
                            "detail": "high",
                        },
                    ],
                }
            ],
            max_output_tokens=settings.OPENAI_OCR_MAX_OUTPUT_TOKENS,
            text={
                "format": {
                    "type": "json_schema",
                    "name": "ocr_page",
                    "strict": True,
                    "schema": schema,
                }
            },
            store=False,
        )
        payload = json.loads(response.output_text)
    except RateLimitError as exc:
        error_code = _openai_error_code(exc)
        if error_code in {"insufficient_quota", "billing_hard_limit_reached"}:
            raise ProviderQuotaError(
                "OpenAI OCR quota is unavailable; check the API project's billing and limits"
            ) from exc
        raise ProviderUnavailableError(
            "OpenAI OCR is rate limited; retry later",
            code="openai_rate_limited",
        ) from exc
    except APIConnectionError as exc:
        raise ProviderUnavailableError(
            "Could not connect to OpenAI OCR",
            code="openai_connection_error",
        ) from exc
    except APIStatusError as exc:
        raise ProviderUnavailableError(
            f"OpenAI OCR failed with status {exc.status_code}",
            code="openai_ocr_failed",
        ) from exc
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise ProviderUnavailableError(
            "OpenAI OCR returned an invalid structured response",
            code="openai_ocr_invalid_response",
        ) from exc

    confidence = max(0.0, min(float(payload["confidence"]), 1.0))
    return OCRResult(
        text=normalize_text(str(payload["text"])),
        method="openai",
        confidence=confidence,
        handwritten=bool(payload["handwritten"]),
        warnings=[str(item) for item in payload.get("warnings", [])],
    )


def _openai_error_code(exc: RateLimitError) -> str | None:
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        nested = body.get("error")
        if isinstance(nested, dict):
            return nested.get("code")
        code = body.get("code")
        return str(code) if code else None
    return None
