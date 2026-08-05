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
and returns a generated document ID. PDF text extraction and retrieval are
the next implementation steps.
