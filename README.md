# langchain_pdf_retriever

A production-oriented FastAPI and LangChain application for uploading PDFs, extracting
native or OCR text, indexing it in Chroma, and streaming grounded answers with page-level
sources. It supports ordinary text PDFs, scanned documents, tables, and an optional
vision fallback for handwritten pages.

## What is implemented

- Validated streaming uploads with configurable byte, page, document, and extraction limits
- SHA-256 duplicate detection and preserved original filenames
- Persistent SQLite document metadata, background jobs, progress, and safe error codes
- Selective PyMuPDF/Tesseract OCR with cached extraction results
- Optional OpenAI vision transcription fallback for handwriting
- Table extraction to Markdown and token-aware, page-preserving chunks
- OpenAI or Ollama embeddings in provider/model-specific Chroma collections
- Batched and fingerprinted indexing that skips unchanged documents
- Document-scoped hybrid vector + BM25 retrieval, relevance filtering, and deduplication
- Quick, Balanced, and Deep answer modes with conversation-aware retrieval
- OpenAI or Ollama answer generation with source labels and OCR warnings
- Server-Sent Events streaming, cancellation, source snippets, and PDF page opening
- Responsive browser UI with upload/index progress and multi-document selection
- Optional bearer authentication, constant-time token comparison, rate limits, request IDs,
  security headers, rotating logs, Docker services, and automated tests

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the complete data flow and scale boundary.

## Local setup

Requirements:

- Python 3.11
- Tesseract 5 for local OCR
- An OpenAI API key, or a running Ollama installation for local models

From the repository root on Windows PowerShell:

```powershell
py -3.11 -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
Copy-Item .env.example backend\.env
```

Edit `backend/.env`, then start the API:

```powershell
Set-Location backend
..\venv\Scripts\python.exe -m uvicorn app.main:app --reload
```

In a second terminal, start the frontend:

```powershell
python -m http.server 5173 --directory frontend
```

Open `http://127.0.0.1:5173`. API documentation is available at
`http://127.0.0.1:8000/docs`, and health/configuration status is at `/health`.

## Provider configuration

### OpenAI

The default configuration uses OpenAI for embeddings and answers:

```dotenv
OPENAI_API_KEY=your-project-api-key
EMBEDDING_PROVIDER=openai
EMBEDDING_MODEL=text-embedding-3-small
GENERATION_PROVIDER=openai
MODEL_NAME=gpt-4o-mini
```

API billing and model access are separate from a ChatGPT subscription. If indexing reports
`provider_quota_exceeded`, check the API project's billing balance, usage, and limits.
Temporary rate limits are retried automatically; exhausted quota requires an account change.

### Fully local with Ollama

Install Ollama, pull the configured models, and use:

```dotenv
EMBEDDING_PROVIDER=ollama
OLLAMA_EMBEDDING_MODEL=nomic-embed-text
GENERATION_PROVIDER=ollama
OLLAMA_CHAT_MODEL=llama3.2:3b
OLLAMA_BASE_URL=http://127.0.0.1:11434
```

Changing the embedding provider or model requires re-indexing existing documents because
vector dimensions and semantics can differ. The API detects incompatible indexes and returns
a clear error instead of mixing them.

## OCR and handwriting

The default `OCR_PROVIDER=auto` flow attempts native text first and uses Tesseract only for
pages that need OCR. On Windows, the application checks common Tesseract installation paths;
set `TESSERACT_DATA_PATH` explicitly if language data is elsewhere.

For difficult handwriting, enable the image-capable fallback:

```dotenv
OCR_ENABLED=true
OCR_PROVIDER=auto
OPENAI_OCR_FALLBACK=true
OPENAI_OCR_MODEL=gpt-4o-mini
OCR_LANGUAGES=eng
OCR_DPI=300
```

You can also use `OCR_PROVIDER=openai` to route every OCR-required page directly to vision.
This can improve handwriting recognition but adds cost, network latency, and a data-sharing
consideration. Keep it disabled when PDFs must remain local. OCR cannot guarantee perfect
transcription; review low-confidence passages against the linked source page.

Additional Tesseract languages use `+`, for example `eng+hin`, and require the matching
`.traineddata` files.

## Upload and performance defaults

The default limits are 25 MB and 300 pages. These are application defaults rather than PDF
format limits. For larger documents, also consider OCR page count, rendered image pixels,
extracted character count, disk quota, worker CPU, and reverse-proxy request limits.

Relevant settings include:

```dotenv
MAX_UPLOAD_SIZE_MB=25
MAX_PDF_PAGES=300
MAX_TOTAL_DOCUMENTS=1000
MAX_PAGE_CHARACTERS=250000
MAX_EXTRACTED_CHARACTERS=5000000
OCR_MAX_PAGES=300
OCR_MAX_IMAGE_PIXELS=20000000
MAX_CONCURRENT_INDEX_JOBS=2
EMBEDDING_BATCH_SIZE=64
```

Uploads are copied in bounded chunks. OCR and embeddings run after upload in a background
job, so the browser receives progress instead of waiting on one long HTTP request. Native
extraction, OCR results, model clients, and unchanged indexes are reused where safe.

## Answer modes

- **Quick:** at most four passages and a short output.
- **Balanced:** hybrid retrieval and normal explanatory output.
- **Deep:** follow-up rewriting, additional search queries, more evidence, and a longer answer.

All modes stream text to the browser. Deep mode intentionally takes longer and makes more
model/retrieval calls. Retrieval remains limited to selected, ready documents.

## Main API endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/health` | Provider, OCR, and limit status without exposing secrets |
| `POST` | `/upload/` | Validate, deduplicate, and store a PDF |
| `GET` | `/documents/` | List documents and processing status |
| `GET` | `/documents/{id}/status` | Latest document and indexing-job state |
| `GET` | `/documents/{id}/text` | Native/OCR extraction details |
| `GET` | `/documents/{id}/chunks` | Token-aware chunks and metadata |
| `GET` | `/documents/{id}/file` | Open the original PDF securely |
| `POST` | `/documents/{id}/index` | Queue indexing; use `force=true` to rebuild |
| `GET` | `/documents/index-jobs/{job_id}` | Poll a specific job |
| `POST` | `/documents/search` | Scoped dense or hybrid retrieval |
| `POST` | `/chat/` | Non-streamed grounded answer |
| `POST` | `/chat/stream` | SSE answer stream used by the frontend |
| `DELETE` | `/documents/{id}` | Delete PDF, extraction cache, metadata, and vectors |

Example scoped chat request:

```json
{
  "question": "What are the main conclusions?",
  "document_ids": ["0123456789abcdef0123456789abcdef"],
  "mode": "balanced",
  "k": 5,
  "history": []
}
```

When `API_AUTH_TOKEN` is set, send `Authorization: Bearer <token>` to application endpoints.
Do not embed a privileged production token in a public frontend.

## Tests

```powershell
Set-Location backend
..\venv\Scripts\python.exe -m pytest tests -q
```

The suite covers upload validation, size/page limits, duplicate handling, authentication,
native extraction, real Tesseract OCR, mocked handwriting fallback and caching, persistent
jobs, quota errors, document-scoped hybrid retrieval, grounded sources, and SSE streaming.

## Docker

Create `backend/.env`, then run:

```powershell
docker compose up --build
```

The frontend is exposed on port `5173` and the API on `8000`. Named volumes preserve PDFs,
metadata/extraction caches, vectors, and logs. The backend image includes English Tesseract
data and runs as a non-root user.

## Production boundary

This repository is ready for a private, single-instance deployment. Before exposing it as a
multi-user service, add identity-provider authentication, tenant ownership on every record,
object storage, malware scanning, a durable external job queue, centralized rate limiting,
encrypted backups, and monitoring for latency, OCR confidence, retrieval recall, cost, and
provider failures. See [SECURITY.md](SECURITY.md).
