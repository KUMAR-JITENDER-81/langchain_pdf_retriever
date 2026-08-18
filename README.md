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

Create embeddings and store the chunks in Chroma with:

```bash
curl -X POST http://127.0.0.1:8000/documents/<document_id>/index
```

This uses `EMBEDDING_MODEL` and `CHROMA_DIR` from `backend/.env`.

Copy `.env.example` to `backend/.env` and fill in the API key before indexing
documents or using chat. Uploads are limited to 10 MB by default.

Search the indexed chunks with:

```bash
curl -X POST http://127.0.0.1:8000/documents/search \
  -H "Content-Type: application/json" \
  -d '{"question":"What is this document about?","k":4}'
```

The response contains the closest chunks, their page metadata, and Chroma's
distance score. Lower distance means a closer vector match.

Ask a question using the RAG chat endpoint:

```bash
curl -X POST http://127.0.0.1:8000/chat/ \
  -H "Content-Type: application/json" \
  -d '{"question":"What is this document about?","k":4}'
```

The chat endpoint retrieves relevant chunks, sends them to the configured
model, and returns an answer with source page references.

List stored documents:

```bash
curl http://127.0.0.1:8000/documents/
```

Delete a document and its indexed vectors:

```bash
curl -X DELETE http://127.0.0.1:8000/documents/<document_id>
```

Run the automated tests from the `backend` directory:

```bash
python -m pytest tests -q
```

Run the browser frontend in a second terminal:

```bash
python -m http.server 5173 --directory frontend
```

Open `http://127.0.0.1:5173` after starting the backend.

## Docker

Create `backend/.env` from `.env.example`, then start the backend with:

```bash
docker compose up --build
```

Uploaded PDFs and Chroma data are stored in named Docker volumes.

For a deployed API, set `API_AUTH_TOKEN` in `backend/.env`. When it is set,
all application endpoints require `Authorization: Bearer <token>`.
