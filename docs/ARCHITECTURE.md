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
      -> table-to-Markdown extraction with dedicated table chunks
      -> token-aware page chunks with OCR quality and evidence bounding boxes
      -> batched local ONNX MiniLM embeddings
      -> provider/model-specific Chroma collection
  -> ready / failed / cancelled status with progress and a safe error code
```

Extraction results are cached by the PDF hash and OCR configuration. A repeated index
request is skipped when its extraction, chunking, provider, and model fingerprint has
not changed. Safe text-cleanup upgrades can migrate an older cache without rerunning OCR;
an explicit force rebuild reruns extraction and OCR.

## Question flow

```text
Question + selected document IDs + depth/task mode
  -> document-version-aware answer cache lookup
  -> validate that selected documents are ready and embedding-compatible
  -> dense vector retrieval
  -> local BM25 keyword retrieval
  -> readable representative-page sampling for overview questions
  -> score fusion and OCR-quality weighting
  -> local CrossEncoder reranking, MMR diversity, and near-duplicate removal
  -> source-labelled context
  -> local Ollama generation, with evidence-only extractive fallback
  -> SSE text stream
  -> answer diagnostics and local feedback record
  -> answer, warnings, snippets, clickable citations, and highlighted page previews
```

Quick mode uses an instant cited extractive engine. Balanced mode uses Qwen3 0.6B for a
responsive local synthesis. Deep mode uses Qwen3 1.7B, contextualizes follow-up questions,
derives additional local search queries, retrieves more evidence, and produces a longer
synthesis. Every model path falls back to the extractive engine if Ollama is unavailable.
Exact repeated questions can reuse a cached answer only while every selected document index,
task, model, prompt, and conversation scope remains unchanged.

## Storage

| Location | Purpose |
|---|---|
| `backend/uploads/` | Original PDFs named by generated document ID |
| `backend/data/documents.sqlite3` | Names, hashes, status, jobs, OCR and indexing metadata |
| `backend/data/extractions/` | Cached native/OCR extraction JSON |
| `backend/data/models/` | Downloaded local ONNX reranker files |
| `backend/chroma_db/` | Persistent vector collections |
| `backend/logs/` | Rotating application logs |

These directories are runtime state and are intentionally excluded from Git.

## OCR strategy

Native text is always attempted first. OCR is selected for pages with too little useful
text, damaged Unicode, or image-dominant content. Tesseract is the fast local path.
`OLLAMA_OCR_FALLBACK=true` enables local Qwen vision transcription when Tesseract output
looks uncertain, which is more useful for handwriting without sending page images away.
Mojibake repair, readability scoring, and a per-document vision fallback limit keep noisy
or large scans usable without allowing CPU-only OCR work to grow without bound.

OCR is probabilistic. The application stores the method and confidence when available,
shows warnings for low-confidence sources, counts low-quality/handwritten pages, and asks the
model not to infer illegible text. Native blocks and detected tables retain bounding boxes;
OCR-only evidence falls back to a whole-page preview when exact coordinates are unavailable.

## Scale boundary

The current worker pool and rate limiter live inside one API process. This is deliberate
for a small deployment. Before running multiple API replicas, replace them with a durable
queue (for example Redis-backed workers), centralized rate limiting, object storage, and
a tenant-aware relational database.
