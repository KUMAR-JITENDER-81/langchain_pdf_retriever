# Architecture

## Ingestion flow

```text
Browser upload
  -> extension, MIME, PDF-header, parser, byte and page validation
  -> SHA-256 deduplication and local PDF storage
  -> SQLite document record
  -> background indexing job
      -> native text and layout extraction with PyMuPDF
      -> selective OCR for image-only or damaged pages
      -> optional table-to-Markdown extraction
      -> token-aware page chunks with source metadata
      -> batched embeddings
      -> provider/model-specific Chroma collection
  -> ready / failed status with progress and a safe error code
```

Extraction results are cached by the PDF hash and OCR configuration. A repeated index
request is skipped when its extraction, chunking, provider, and model fingerprint has
not changed. Changing an OCR or chunk setting invalidates the relevant cache safely.

## Question flow

```text
Question + selected document IDs + answer mode
  -> validate that selected documents are ready and embedding-compatible
  -> dense vector retrieval
  -> local BM25 keyword retrieval
  -> score fusion, relevance threshold, and near-duplicate removal
  -> source-labelled context
  -> OpenAI or Ollama generation
  -> SSE text stream
  -> answer, warnings, snippets, and clickable PDF pages
```

Quick mode uses fewer passages and a shorter output. Balanced mode is the normal path.
Deep mode can contextualize follow-up questions, generate additional search queries,
retrieve more evidence, and produce a longer synthesis.

## Storage

| Location | Purpose |
|---|---|
| `backend/uploads/` | Original PDFs named by generated document ID |
| `backend/data/documents.sqlite3` | Names, hashes, status, jobs, OCR and indexing metadata |
| `backend/data/extractions/` | Cached native/OCR extraction JSON |
| `backend/chroma_db/` | Persistent vector collections |
| `backend/logs/` | Rotating application logs |

These directories are runtime state and are intentionally excluded from Git.

## OCR strategy

Native text is always attempted first. OCR is selected for pages with too little useful
text, damaged Unicode, or image-dominant content. Tesseract is the local, private path.
`OPENAI_OCR_FALLBACK=true` enables a stricter vision transcription fallback that is more
useful for handwriting but sends page images to the configured external provider.

OCR is probabilistic. The application stores the method and confidence when available,
shows warnings for low-confidence sources, and asks the model not to infer illegible text.

## Scale boundary

The current worker pool and rate limiter live inside one API process. This is deliberate
for a small deployment. Before running multiple API replicas, replace them with a durable
queue (for example Redis-backed workers), centralized rate limiting, object storage, and
a tenant-aware relational database.
