# langchain_pdf_retriever

## Current API

Start the backend from the `backend` directory:

```bash
uvicorn app.main:app --reload
```

Upload a PDF with:

```bash
curl -X POST http://127.0.0.1:8000/upload/ \
  -F "file=@/path/to/document.pdf"
```

The endpoint validates the `.pdf` extension, stores the file in `UPLOAD_DIR`,
and returns a generated document ID. Retrieval, chunking, and embeddings are
the next implementation steps.

Extract text from an uploaded document with:

```bash
curl http://127.0.0.1:8000/documents/<document_id>/text
```

The extraction endpoint returns the page count, page text, and combined text.

Split an uploaded document into overlapping chunks with:

```bash
curl http://127.0.0.1:8000/documents/<document_id>/chunks
```

The chunking endpoint uses a 1,000-character chunk size and 200-character
overlap by default, while preserving the source page for every chunk.
