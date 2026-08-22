# langchain_pdf_retriever

A production-oriented FastAPI and LangChain application for uploading PDFs, extracting
native or OCR text, indexing it in Chroma, and streaming grounded answers with page-level
sources. It supports ordinary text PDFs, scanned documents, tables, and an optional
vision fallback for handwritten pages.

## What is implemented

- Validated streaming uploads with configurable byte, page, document, and extraction limits
- SHA-256 duplicate detection and preserved original filenames
- Persistent SQLite metadata, recoverable/cancellable background jobs, bulk refresh, and safe errors
- Selective PyMuPDF/Tesseract OCR with cached extraction results
- Optional local Ollama vision transcription fallback for handwriting
- Dedicated table chunks that preserve Markdown rows, page numbers, and table bounding boxes
- Free local ONNX MiniLM or Ollama embeddings in provider/model-specific collections
- Batched and fingerprinted indexing that skips unchanged documents
- Document-scoped hybrid vector + BM25 retrieval followed by a local ONNX CrossEncoder reranker
- Quick, Balanced, and Deep depth modes plus Q&A, summary, compare, extract, quiz, and translate tasks
- Local Ollama answer generation with a built-in evidence-only fallback
- Automatic fallback between installed Ollama text models and non-blocking startup warmup
- Live Ollama/model readiness in the UI, with a manual warmup control
- SSE streaming, answer cancellation, clickable citations, highlighted page previews, and full-PDF opening
- Document-version-aware answer caching for safe instant repeats
- Local answer IDs, timings, citation checks, feedback, quality dashboard, and regression benchmarks
- Responsive browser UI with upload/index progress and multi-document selection
- Optional bearer authentication, constant-time token comparison, rate limits, request IDs,
  security headers, rotating logs, Docker services, and automated tests

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the complete data flow and scale boundary.

## Local setup

Requirements:

- Python 3.11
- Tesseract 5 for local OCR
- Ollama for high-quality local answers and difficult handwriting OCR

From the repository root on Windows PowerShell:

```powershell
py -3.11 -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
Copy-Item .env.example backend\.env
winget install --exact --id Ollama.Ollama
ollama pull qwen3:0.6b
ollama pull qwen3:1.7b
ollama pull qwen3-vl:4b-instruct
```

Edit `backend/.env`, then start the complete application:

```powershell
.\venv\Scripts\python.exe app.py
```

The launcher checks Ollama, starts any missing backend/frontend services, opens the app,
and prints actionable startup errors. Use `python app.py --check` to check service status or
`python app.py --no-browser` to start without opening a browser.

For manual development, start the backend:

```powershell
Set-Location backend
..\venv\Scripts\python.exe -m uvicorn app.main:app --reload
```

Then start the frontend from the repository root in a second terminal:

```powershell
.\venv\Scripts\python.exe -m http.server 5173 --bind 127.0.0.1 --directory frontend
```

Open `http://127.0.0.1:5173`. API documentation is available at
`http://127.0.0.1:8000/docs`, and health/configuration status is at `/health`.

## Free local provider configuration

No API key, subscription, or per-request payment is used. The default setup uses Chroma's
local ONNX `all-MiniLM-L6-v2` model for semantic embeddings and Ollama for answers:

```dotenv
EMBEDDING_PROVIDER=local
LOCAL_EMBEDDING_MODEL=all-MiniLM-L6-v2
GENERATION_PROVIDER=ollama
OLLAMA_FAST_MODEL=qwen3:0.6b
OLLAMA_CHAT_MODEL=qwen3:1.7b
OLLAMA_OCR_MODEL=qwen3-vl:4b-instruct
OLLAMA_BASE_URL=http://127.0.0.1:11434
OLLAMA_AUTO_MODEL_FALLBACK=true
OLLAMA_WARMUP_ON_START=true
OLLAMA_WARMUP_TIMEOUT_SECONDS=60
OLLAMA_BALANCED_TIMEOUT_SECONDS=30
OLLAMA_DEEP_TIMEOUT_SECONDS=75
OLLAMA_QUEUE_TIMEOUT_SECONDS=2
OLLAMA_MAX_CONCURRENT_GENERATIONS=1
BALANCED_CONTEXT_CHARACTERS=5600
DEEP_CONTEXT_CHARACTERS=6800
MAX_HISTORY_CHARACTERS=1600
LOCAL_ANSWER_FALLBACK=true
QUICK_MODE_LOCAL=true
BALANCED_MODE_LOCAL=false
```

MiniLM runs inside the Python process and downloads its model files once on first use. If
Ollama is stopped or its model is unavailable, the application automatically produces a
grounded extractive answer from retrieved source sentences instead of failing. Existing
documents that failed only because of OpenAI quota are automatically queued for local
re-indexing after the backend restarts.

Retrieval also uses `cross-encoder/ms-marco-MiniLM-L6-v2` as a second-stage CPU reranker.
Its Apache-licensed ONNX file downloads once to `backend/data/models/`; if it cannot load,
hybrid ranking continues automatically. Use `POST /reranker/warmup` to install and warm it
before the first question.

The backend preloads the Balanced model in the background at startup, and the browser shows
which text and vision models are installed. `POST /ollama/warmup` can be used to preload the
Balanced model again after Ollama has released it from memory. If either configured text
model is missing, the application can temporarily use the other installed text model while
keeping citations and the evidence-only fallback active.

Balanced mode has a bounded CPU time budget, and overlapping Ollama requests fail over to
the instant cited-evidence engine instead of queuing for a minute. Retrieved overview chunks
are presented in document order, visible headings provide a conservative document-type hint,
and conversation/evidence text is capped so long chats cannot overflow the model context.

You can optionally use `EMBEDDING_PROVIDER=ollama` with
`OLLAMA_EMBEDDING_MODEL=nomic-embed-text`. Changing embedding models requires re-indexing
because vector dimensions and semantics differ.

## OCR and handwriting

The default `OCR_PROVIDER=auto` flow attempts native text first and uses Tesseract only for
pages that need OCR. On Windows, the application checks common Tesseract installation paths;
set `TESSERACT_DATA_PATH` explicitly if language data is elsewhere.

For difficult handwriting, a free image-capable Ollama model is used:

```dotenv
OCR_ENABLED=true
OCR_PROVIDER=auto
OLLAMA_OCR_FALLBACK=true
OLLAMA_OCR_MODEL=qwen3-vl:4b-instruct
OCR_LANGUAGES=eng
OCR_DPI=300
```

Use `OCR_PROVIDER=ollama` to force every OCR-required page through local vision. In `auto`
mode, fast Tesseract runs first and Ollama is used only when its output looks uncertain.
Auto mode limits expensive vision retries per document so large scans remain manageable.
Everything stays on the computer, but vision OCR is slower than Tesseract. OCR cannot
guarantee perfect transcription; review low-confidence passages against the linked page.

Additional Tesseract languages use `+`, for example `eng+hin`, and require the matching
`.traineddata` files.

## Upload and performance defaults

The default limits are 100 MB and 500 pages. These are application defaults rather than PDF
format limits. For larger documents, also consider OCR page count, rendered image pixels,
extracted character count, disk quota, worker CPU, and reverse-proxy request limits.

Relevant settings include:

```dotenv
MAX_UPLOAD_SIZE_MB=100
MAX_PDF_PAGES=500
MAX_TOTAL_DOCUMENTS=1000
MAX_PAGE_CHARACTERS=250000
MAX_EXTRACTED_CHARACTERS=10000000
OCR_MAX_PAGES=500
OCR_MAX_IMAGE_PIXELS=20000000
OCR_VISION_MAX_PAGES_PER_DOCUMENT=12
MAX_CONCURRENT_INDEX_JOBS=1
EMBEDDING_BATCH_SIZE=64
```

Uploads are copied in bounded chunks. OCR and embeddings run after upload in a background
job, so the browser receives progress instead of waiting on one long HTTP request. Native
extraction, OCR results, model clients, and unchanged indexes are reused where safe.
Active indexing can be cancelled between processing stages, interrupted jobs recover after
restart, and the UI can refresh all ready documents in one request.

## Answer modes

- **Quick:** instant extractive answer from cited passages; no model wait.
- **Balanced:** concise local-AI synthesis with Qwen3 0.6B, normally the best speed/quality choice.
- **Deep:** broader retrieval and local synthesis with Qwen3 1.7B; slower but more detailed.

All modes stream text to the browser. Deep mode intentionally takes longer and makes more
model/retrieval calls. Retrieval remains limited to selected, ready documents.

Task mode is independent of answer depth: use `summary`, `compare`, `extract`, `quiz`, or
`translate` when a specific output shape is required. Advanced tasks use Ollama even with
Quick depth when it is available, while the evidence fallback remains active.

## Main API endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/health` | Provider, OCR, and limit status without exposing secrets |
| `POST` | `/ollama/warmup` | Preload the Balanced Ollama model to reduce first-answer delay |
| `POST` | `/reranker/warmup` | Install/load the local semantic reranker |
| `POST` | `/upload/` | Validate, deduplicate, and store a PDF |
| `GET` | `/documents/` | List documents and processing status |
| `GET` | `/documents/{id}/status` | Latest document and indexing-job state |
| `GET` | `/documents/{id}/text` | Native/OCR extraction details |
| `GET` | `/documents/{id}/chunks` | Token-aware chunks and metadata |
| `GET` | `/documents/{id}/file` | Open the original PDF securely |
| `GET` | `/documents/{id}/pages/{page}/preview` | Render a page with an optional evidence highlight |
| `POST` | `/documents/{id}/index` | Queue indexing; use `force=true` to rebuild |
| `POST` | `/documents/{id}/index/cancel` | Cancel active indexing safely |
| `POST` | `/documents/index` | Bulk queue or refresh documents |
| `GET` | `/documents/index-jobs/{job_id}` | Poll a specific job |
| `POST` | `/documents/search` | Scoped dense or hybrid retrieval |
| `POST` | `/chat/` | Non-streamed grounded answer |
| `POST` | `/chat/stream` | SSE answer stream used by the frontend |
| `POST` | `/quality/feedback` | Store local helpful/not-helpful feedback |
| `GET` | `/quality/summary` | Aggregated latency, grounding, and feedback metrics |
| `GET` | `/quality/runs` | Recent local answer diagnostics |
| `DELETE` | `/documents/{id}` | Delete PDF, extraction cache, metadata, and vectors |

Example scoped chat request:

```json
{
  "question": "What are the main conclusions?",
  "document_ids": ["0123456789abcdef0123456789abcdef"],
  "mode": "balanced",
  "task": "summary",
  "response_language": "English",
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
native extraction, real Tesseract OCR, mocked local-vision handwriting fallback and caching,
persistent/cancellable jobs, semantic reranking, table chunks, page previews, answer caching,
quality feedback, evaluation metrics, grounded sources, and SSE streaming.

To run a repeatable quality benchmark, copy and edit the example dataset, then run:

```powershell
Set-Location backend
Copy-Item evaluation/cases.example.json evaluation/cases.json
..\venv\Scripts\python.exe scripts/evaluate.py evaluation/cases.json --output evaluation/latest.report.json --fail-under 0.8
```

Cases can select PDFs by filename or document ID and check expected answer terms, source
pages, citation coverage, latency, and an overall minimum score. Evaluation bypasses the
answer cache by default.

## Docker

Create `backend/.env`, then run:

```powershell
docker compose up --build
```

The frontend is exposed on port `5173`, the API on `8000`, and Ollama on `11434`. Compose
downloads both text models and the Qwen vision model, and keeps Ollama, MiniLM, PDFs, extraction caches,
vectors, metadata, and logs in named volumes. The backend includes English Tesseract data
and runs as a non-root user.

## Production boundary

This repository is ready for a private, single-instance deployment. Before exposing it as a
multi-user service, add identity-provider authentication, tenant ownership on every record,
object storage, malware scanning, a durable external job queue, centralized rate limiting,
encrypted backups, and monitoring for latency, OCR confidence, retrieval recall, cost, and
provider failures. See [SECURITY.md](SECURITY.md).
