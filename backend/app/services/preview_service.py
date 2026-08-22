from __future__ import annotations

import math
from typing import Any

import pymupdf

from app.core.config import settings
from app.core.errors import AppError, InvalidPDFError
from app.services.pdf_service import get_pdf_path


def render_page_preview(
    document_id: str,
    page_number: int,
    *,
    bbox: list[float] | None = None,
) -> tuple[bytes, dict[str, Any]]:
    """Render one PDF page and optionally paint a non-destructive evidence highlight."""
    try:
        pdf = pymupdf.open(get_pdf_path(document_id))
    except (pymupdf.FileDataError, RuntimeError, ValueError) as exc:
        raise InvalidPDFError() from exc

    try:
        if page_number < 1 or page_number > pdf.page_count:
            raise AppError(
                f"Page must be between 1 and {pdf.page_count}",
                code="invalid_page_number",
                status_code=422,
            )
        page = pdf.load_page(page_number - 1)
        highlight = _clamped_bbox(page, bbox)
        if highlight is not None:
            page.draw_rect(
                highlight,
                color=(0.12, 0.72, 0.5),
                fill=(0.35, 0.95, 0.73),
                width=2.2,
                fill_opacity=0.20,
                overlay=True,
            )

        dpi = max(72, min(int(settings.PDF_PREVIEW_DPI), 220))
        pixel_count = (
            float(page.rect.width) / 72 * dpi * float(page.rect.height) / 72 * dpi
        )
        if pixel_count > settings.PDF_PREVIEW_MAX_PIXELS:
            dpi = max(
                72,
                int(dpi * math.sqrt(settings.PDF_PREVIEW_MAX_PIXELS / pixel_count)),
            )
        pixmap = page.get_pixmap(dpi=dpi, colorspace=pymupdf.csRGB, alpha=False)
        return pixmap.tobytes("png"), {
            "page": page_number,
            "page_count": pdf.page_count,
            "dpi": dpi,
            "highlighted": highlight is not None,
            "width": pixmap.width,
            "height": pixmap.height,
        }
    finally:
        pdf.close()


def _clamped_bbox(
    page: pymupdf.Page,
    bbox: list[float] | None,
) -> pymupdf.Rect | None:
    if bbox is None or len(bbox) != 4:
        return None
    try:
        rectangle = pymupdf.Rect(*(float(value) for value in bbox)) & page.rect
    except (TypeError, ValueError):
        return None
    if rectangle.is_empty or rectangle.width < 1 or rectangle.height < 1:
        return None
    padding = 3
    expanded = pymupdf.Rect(
        rectangle.x0 - padding,
        rectangle.y0 - padding,
        rectangle.x1 + padding,
        rectangle.y1 + padding,
    ) & page.rect
    return expanded if not expanded.is_empty else rectangle
