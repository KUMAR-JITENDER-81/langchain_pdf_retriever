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
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pymupdf

from app.core.config import settings
from app.core.errors import ConfigurationError, ProviderUnavailableError


MOJIBAKE_MARKERS = ("â€", "â€™", "â€œ", "â€˜", "â€¦", "â", "Ã", "Â", "ðŸ", "\ufffd")
SAFE_PUNCTUATION = set(".,:;!?%$€£¥+-=/()[]{}_'\"@#&*<>|\\")
COMMON_PROSE_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "because", "by", "can", "for",
    "from", "has", "have", "in", "is", "it", "of", "on", "or", "that", "the",
    "their", "this", "to", "used", "using", "was", "which", "with", "you",
}


@dataclass(slots=True)
class OCRResult:
    text: str
    method: str
    confidence: float | None = None
    handwritten: bool | None = None
    warnings: list[str] = field(default_factory=list)


def normalize_text(text: str) -> str:
    """Normalize OCR/native text without destroying paragraph boundaries."""
    normalized = repair_mojibake(text or "")
    normalized = unicodedata.normalize("NFKC", normalized).replace("\x00", "")
    normalized = normalized.replace("\u00ad", "").replace("\u200b", "")
    normalized = re.sub(r"(?<=\w)-[ \t]*\n[ \t]*(?=\w)", "", normalized)
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


def repair_mojibake(text: str) -> str:
    """Repair common UTF-8 text that was accidentally decoded as Windows-1252."""
    current = text
    for _ in range(3):
        previous = current
        for encoding in ("cp1252", "latin1"):
            try:
                candidate = current.encode(encoding).decode("utf-8")
            except (UnicodeEncodeError, UnicodeDecodeError):
                continue
            if _mojibake_score(candidate) < _mojibake_score(current):
                current = candidate
        if current == previous:
            break
    return current


def meaningful_character_count(text: str) -> int:
    return sum(character.isalnum() for character in text)


def ocr_text_quality(text: str) -> float:
    """Estimate whether OCR output is readable enough to avoid vision fallback."""
    normalized = normalize_text(text)
    if meaningful_character_count(normalized) < 5:
        return 0.0
    characters = [character for character in normalized if not character.isspace()]
    alphanumeric_ratio = sum(character.isalnum() for character in characters) / max(
        len(characters), 1
    )
    words = re.findall(r"\w+", normalized, re.UNICODE)
    useful_words = [word for word in words if len(word) >= 2]
    word_ratio = len(useful_words) / max(len(words), 1)
    substantial_words = [word for word in words if len(word) >= 3]
    substantial_ratio = len(substantial_words) / max(len(words), 1)
    prose_ratio = min(
        sum(word.casefold() in COMMON_PROSE_WORDS for word in words)
        / max(len(words) * 0.16, 1),
        1.0,
    )
    alphabetic_words = [word for word in words if word.isalpha() and len(word) >= 4]
    mixed_case_ratio = sum(
        any(character.isupper() for character in word[1:])
        and not word.isupper()
        for word in alphabetic_words
    ) / max(len(alphabetic_words), 1)
    suspicious_symbols = sum(
        not character.isalnum()
        and not character.isspace()
        and character not in SAFE_PUNCTUATION
        for character in normalized
    ) / max(len(normalized), 1)
    short_line_ratio = sum(
        len(line.strip()) < 4 for line in normalized.splitlines() if line.strip()
    ) / max(sum(bool(line.strip()) for line in normalized.splitlines()), 1)
    replacement_penalty = normalized.count("\ufffd") / max(len(normalized), 1)
    mojibake_penalty = _mojibake_score(normalized) / max(len(normalized), 1)
    control_penalty = sum(
        unicodedata.category(character).startswith("C")
        for character in normalized
        if character not in "\n\t"
    ) / max(len(normalized), 1)
    score = (
        alphanumeric_ratio * 0.34
        + word_ratio * 0.18
        + substantial_ratio * 0.13
        + prose_ratio * 0.20
        + (1 - mixed_case_ratio) * 0.08
        + (1 - short_line_ratio) * 0.07
    )
    score -= min(
        0.7,
        replacement_penalty * 5
        + control_penalty * 5
        + suspicious_symbols * 1.6
        + mojibake_penalty * 5,
    )
    score -= min(0.24, mixed_case_ratio * 0.35)
    return round(max(0.0, min(score, 1.0)), 3)


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

    image_coverage = page_image_coverage(page)
    if (
        ocr_text_quality(normalized) < settings.OCR_NATIVE_QUALITY_THRESHOLD
        and (image_coverage >= 0.2 or meaningful < 400)
    ):
        return True, "low_native_text_quality"

    if image_coverage >= 0.6 and meaningful < 200:
        return True, "image_dominant_page"

    return False, None


def run_ocr(
    page: pymupdf.Page,
    page_number: int,
    *,
    allow_vision: bool = True,
) -> OCRResult:
    provider = settings.OCR_PROVIDER.strip().lower()
    if not settings.OCR_ENABLED or provider == "disabled":
        return OCRResult("", "disabled", warnings=["OCR is disabled"])

    if provider not in {"auto", "tesseract", "ollama"}:
        raise ConfigurationError(
            "OCR_PROVIDER must be one of: auto, tesseract, ollama, disabled"
        )

    warnings: list[str] = []
    tesseract_result: OCRResult | None = None
    if provider in {"auto", "tesseract"}:
        try:
            tesseract_result = _run_tesseract_ocr(page)
            if provider == "tesseract":
                return tesseract_result
            if (
                meaningful_character_count(tesseract_result.text) >= 5
                and float(tesseract_result.confidence or 0.0)
                >= settings.OCR_VISION_QUALITY_THRESHOLD
            ):
                return tesseract_result
            warnings.extend(tesseract_result.warnings)
            warnings.append(
                "Tesseract output was uncertain; trying the local vision model"
            )
        except (ConfigurationError, ProviderUnavailableError) as exc:
            if provider == "tesseract":
                raise
            warnings.append(exc.message)

    should_try_vision = provider == "ollama" or (
        provider == "auto" and settings.OLLAMA_OCR_FALLBACK and allow_vision
    )
    if should_try_vision:
        try:
            result = _run_ollama_ocr(page, page_number)
            result.warnings = warnings + result.warnings
            if meaningful_character_count(result.text) >= 5:
                return result
            warnings.extend(result.warnings)
        except (ConfigurationError, ProviderUnavailableError) as exc:
            if provider == "ollama":
                raise
            warnings.append(exc.message)
    elif provider == "auto" and settings.OLLAMA_OCR_FALLBACK and not allow_vision:
        warnings.append(
            "Vision OCR was skipped because this document reached its local vision-page limit"
        )

    if tesseract_result and meaningful_character_count(tesseract_result.text) >= 5:
        tesseract_result.warnings = list(dict.fromkeys(warnings + tesseract_result.warnings))
        return tesseract_result

    warnings.append(
        "No local OCR provider produced readable text; start Ollama or upload a clearer scan"
    )
    return OCRResult("", "unavailable", warnings=list(dict.fromkeys(warnings)))


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

    confidence = ocr_text_quality(text)
    warnings = [] if text else ["Tesseract did not recognize text on this page"]
    return OCRResult(
        text=text,
        method="tesseract",
        confidence=confidence,
        warnings=warnings,
    )


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


def _render_page_as_base64(page: pymupdf.Page) -> str:
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
    return base64.b64encode(pixmap.tobytes("png")).decode("ascii")


def _run_ollama_ocr(page: pymupdf.Page, page_number: int) -> OCRResult:
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
        f"OCR PDF page {page_number}. Transcribe every visible word exactly in reading "
        "order. Preserve useful line breaks. Do not summarize, correct, complete, or infer "
        "missing content. Use [illegible] where text cannot be read. Return a conservative "
        "confidence from 0 to 1 and whether meaningful handwriting is present. Return only "
        f"JSON matching this schema: {json.dumps(schema, separators=(',', ':'))}"
    )
    payload = {
        "model": settings.OLLAMA_OCR_MODEL,
        "messages": [
            {
                "role": "user",
                "content": instructions,
                "images": [_render_page_as_base64(page)],
            }
        ],
        "stream": False,
        "format": schema,
        "think": False,
        "keep_alive": settings.OLLAMA_KEEP_ALIVE,
        "options": {
            "temperature": 0,
            "num_predict": settings.OLLAMA_OCR_MAX_OUTPUT_TOKENS,
            "num_ctx": settings.OLLAMA_NUM_CTX,
            "seed": 42,
        },
    }
    request = Request(
        f"{settings.OLLAMA_BASE_URL.rstrip('/')}/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )

    try:
        with urlopen(request, timeout=settings.OLLAMA_OCR_TIMEOUT_SECONDS) as response:
            response_payload = json.loads(response.read().decode("utf-8"))
        content = str(response_payload["message"]["content"]).strip()
        if content.startswith("```"):
            content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content).strip()
        result = json.loads(content)
    except HTTPError as exc:
        if exc.code == 404:
            raise ConfigurationError(
                f"Ollama vision model '{settings.OLLAMA_OCR_MODEL}' is not installed"
            ) from exc
        raise ProviderUnavailableError(
            f"Ollama vision OCR returned status {exc.code}",
            code="ollama_ocr_failed",
        ) from exc
    except (URLError, TimeoutError, OSError) as exc:
        raise ProviderUnavailableError(
            "Could not reach the local Ollama vision OCR service",
            code="ollama_unavailable",
        ) from exc
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise ProviderUnavailableError(
            "The local vision model returned invalid OCR data",
            code="ollama_ocr_invalid_response",
        ) from exc

    confidence = max(0.0, min(float(result["confidence"]), 1.0))
    return OCRResult(
        text=normalize_text(str(result["text"])),
        method="ollama-vision",
        confidence=confidence,
        handwritten=bool(result["handwritten"]),
        warnings=[str(item) for item in result.get("warnings", [])],
    )


def _mojibake_score(text: str) -> int:
    return sum(text.count(marker) for marker in MOJIBAKE_MARKERS)
