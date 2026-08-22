from app.rag import splitter


def test_tables_become_dedicated_chunks_with_bounding_boxes(monkeypatch):
    table_markdown = "[Table]\n| Item | Amount |\n| --- | --- |\n| Hosting | 120 |"
    monkeypatch.setattr(
        splitter,
        "extract_pdf_text",
        lambda *args, **kwargs: {
            "pages": [f"Quarterly costs are listed below.\n\n{table_markdown}"],
            "page_details": [
                {
                    "heading": "Quarterly costs",
                    "method": "native",
                    "ocr_confidence": None,
                    "handwritten": False,
                    "text_quality": 0.9,
                    "image_coverage": 0.0,
                    "rotation": 0,
                    "blocks": [
                        {"bbox": [50, 50, 300, 80], "text": "Quarterly costs are listed below."}
                    ],
                    "tables": [
                        {"bbox": [50, 100, 400, 220], "markdown": table_markdown}
                    ],
                }
            ],
            "source_sha256": "hash",
            "extraction_fingerprint": "fingerprint",
        },
    )

    result = splitter.split_pdf_text("d" * 32, chunk_size=100, chunk_overlap=10)

    assert result["table_count"] == 1
    assert len(result["chunks"]) == 2
    text_chunk = next(item for item in result["chunks"] if item["content_type"] == "text")
    table_chunk = next(item for item in result["chunks"] if item["content_type"] == "table")
    assert text_chunk["bbox"] == [50.0, 50.0, 300.0, 80.0]
    assert table_chunk["bbox"] == [50.0, 100.0, 400.0, 220.0]
    assert table_chunk["table_index"] == 1
    assert "Hosting" in table_chunk["text"]
