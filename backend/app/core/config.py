from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Free, local providers. No API key or paid service is required.
    GENERATION_PROVIDER: str = "ollama"
    OLLAMA_FAST_MODEL: str = "qwen3:0.6b"
    OLLAMA_CHAT_MODEL: str = "qwen3:1.7b"
    LOCAL_ANSWER_MODEL: str = "extractive-v1"
    LOCAL_ANSWER_FALLBACK: bool = True
    QUICK_MODE_LOCAL: bool = True
    BALANCED_MODE_LOCAL: bool = False
    EMBEDDING_PROVIDER: str = "local"
    LOCAL_EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    OLLAMA_BASE_URL: str = "http://127.0.0.1:11434"
    OLLAMA_EMBEDDING_MODEL: str = "nomic-embed-text"
    OLLAMA_KEEP_ALIVE: str = "10m"
    OLLAMA_NUM_CTX: int = 3072
    OLLAMA_AUTO_MODEL_FALLBACK: bool = True
    OLLAMA_WARMUP_ON_START: bool = True
    OLLAMA_WARMUP_TIMEOUT_SECONDS: float = 60.0
    OLLAMA_BALANCED_TIMEOUT_SECONDS: float = 30.0
    OLLAMA_DEEP_TIMEOUT_SECONDS: float = 75.0
    OLLAMA_QUEUE_TIMEOUT_SECONDS: float = 2.0
    OLLAMA_MAX_CONCURRENT_GENERATIONS: int = 1

    # Generation
    TEMPERATURE: float = 0.0
    MAX_ANSWER_TOKENS: int = 300
    GENERATION_TIMEOUT_SECONDS: float = 75.0
    BALANCED_CONTEXT_CHARACTERS: int = 5600
    DEEP_CONTEXT_CHARACTERS: int = 6800
    MAX_HISTORY_CHARACTERS: int = 1600

    # Storage
    UPLOAD_DIR: str = "uploads"
    CHROMA_DIR: str = "chroma_db"
    METADATA_DB: str = "data/documents.sqlite3"
    EXTRACTION_CACHE_DIR: str = "data/extractions"
    LOG_DIR: str = "logs"
    LOG_LEVEL: str = "INFO"

    # Upload and document resource limits
    MAX_UPLOAD_SIZE_MB: float = 100.0
    MAX_PDF_PAGES: int = 500
    MAX_TOTAL_DOCUMENTS: int = 1000
    MAX_PAGE_CHARACTERS: int = 250_000
    MAX_EXTRACTED_CHARACTERS: int = 10_000_000
    PDF_PREVIEW_DPI: int = 144
    PDF_PREVIEW_MAX_PIXELS: int = 8_000_000

    # OCR. Native text is always attempted before OCR.
    OCR_ENABLED: bool = True
    OCR_PROVIDER: str = "auto"
    OCR_LANGUAGES: str = "eng"
    TESSERACT_DATA_PATH: str = ""
    OCR_DPI: int = 300
    OCR_MIN_TEXT_CHARACTERS: int = 40
    OCR_MAX_PAGES: int = 500
    OCR_MAX_IMAGE_PIXELS: int = 20_000_000
    OLLAMA_OCR_FALLBACK: bool = True
    OLLAMA_OCR_MODEL: str = "qwen3-vl:4b-instruct"
    OLLAMA_OCR_MAX_OUTPUT_TOKENS: int = 4000
    OLLAMA_OCR_TIMEOUT_SECONDS: float = 240.0
    OCR_NATIVE_QUALITY_THRESHOLD: float = 0.36
    OCR_VISION_QUALITY_THRESHOLD: float = 0.52
    OCR_VISION_MAX_PAGES_PER_DOCUMENT: int = 12
    EXTRACT_TABLES: bool = True

    # Indexing and retrieval
    MAX_CONCURRENT_INDEX_JOBS: int = 1
    AUTO_RETRY_LEGACY_PROVIDER_FAILURES: bool = True
    AUTO_RETRY_INTERRUPTED_JOBS: bool = True
    EMBEDDING_BATCH_SIZE: int = 64
    CHUNK_SIZE_TOKENS: int = 450
    CHUNK_OVERLAP_TOKENS: int = 75
    DEFAULT_RETRIEVAL_K: int = 6
    MAX_RETRIEVAL_K: int = 20
    RETRIEVAL_CANDIDATE_MULTIPLIER: int = 5
    OVERVIEW_RETRIEVAL_K: int = 8
    MAX_VECTOR_DISTANCE: float = 1.35
    MIN_RETRIEVAL_RELEVANCE: float = 0.12
    HYBRID_SEARCH_ENABLED: bool = True
    HYBRID_MAX_CORPUS_CHUNKS: int = 5000
    RERANKER_ENABLED: bool = True
    RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L6-v2"
    RERANKER_ONNX_FILE: str = "onnx/model.onnx"
    RERANKER_CACHE_DIR: str = "data/models"
    RERANKER_AUTO_DOWNLOAD: bool = True
    RERANKER_MAX_CANDIDATES: int = 24
    RERANKER_MAX_LENGTH: int = 384
    RERANKER_BATCH_SIZE: int = 8
    RERANKER_WEIGHT: float = 0.65

    # Local quality tracking. Answer text and feedback stay in the metadata database.
    QUALITY_TRACKING_ENABLED: bool = True
    QUALITY_RECENT_RUN_LIMIT: int = 200
    ANSWER_CACHE_ENABLED: bool = True
    ANSWER_CACHE_TTL_HOURS: int = 168
    ANSWER_CACHE_MAX_ENTRIES: int = 500

    # Browser/API access
    FRONTEND_ORIGINS: str = "http://127.0.0.1:5173,http://localhost:5173"
    API_AUTH_TOKEN: str = ""
    RATE_LIMIT_ENABLED: bool = True
    CHAT_RATE_LIMIT_PER_MINUTE: int = 30
    UPLOAD_RATE_LIMIT_PER_MINUTE: int = 10
    INDEX_RATE_LIMIT_PER_MINUTE: int = 20
    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
        case_sensitive=True,
    )


settings = Settings()
