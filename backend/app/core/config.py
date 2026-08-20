from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Providers
    OPENAI_API_KEY: str = ""
    GENERATION_PROVIDER: str = "openai"
    MODEL_NAME: str = "gpt-4o-mini"
    OLLAMA_CHAT_MODEL: str = "llama3.2:3b"
    EMBEDDING_PROVIDER: str = "openai"
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    OLLAMA_BASE_URL: str = "http://127.0.0.1:11434"
    OLLAMA_EMBEDDING_MODEL: str = "nomic-embed-text"

    # Generation
    TEMPERATURE: float = 0.0
    MAX_ANSWER_TOKENS: int = 900
    GENERATION_TIMEOUT_SECONDS: float = 90.0

    # Storage
    UPLOAD_DIR: str = "uploads"
    CHROMA_DIR: str = "chroma_db"
    METADATA_DB: str = "data/documents.sqlite3"
    EXTRACTION_CACHE_DIR: str = "data/extractions"
    LOG_DIR: str = "logs"
    LOG_LEVEL: str = "INFO"

    # Upload and document resource limits
    MAX_UPLOAD_SIZE_MB: float = 25.0
    MAX_PDF_PAGES: int = 300
    MAX_TOTAL_DOCUMENTS: int = 1000
    MAX_PAGE_CHARACTERS: int = 250_000
    MAX_EXTRACTED_CHARACTERS: int = 5_000_000

    # OCR. Native text is always attempted before OCR.
    OCR_ENABLED: bool = True
    OCR_PROVIDER: str = "auto"
    OCR_LANGUAGES: str = "eng"
    TESSERACT_DATA_PATH: str = ""
    OCR_DPI: int = 300
    OCR_MIN_TEXT_CHARACTERS: int = 40
    OCR_MAX_PAGES: int = 300
    OCR_MAX_IMAGE_PIXELS: int = 20_000_000
    OPENAI_OCR_FALLBACK: bool = False
    OPENAI_OCR_MODEL: str = "gpt-4o-mini"
    OPENAI_OCR_MAX_OUTPUT_TOKENS: int = 4000
    EXTRACT_TABLES: bool = True

    # Indexing and retrieval
    MAX_CONCURRENT_INDEX_JOBS: int = 2
    EMBEDDING_BATCH_SIZE: int = 64
    CHUNK_SIZE_TOKENS: int = 450
    CHUNK_OVERLAP_TOKENS: int = 75
    DEFAULT_RETRIEVAL_K: int = 5
    MAX_RETRIEVAL_K: int = 20
    RETRIEVAL_CANDIDATE_MULTIPLIER: int = 3
    MAX_VECTOR_DISTANCE: float = 1.35
    MIN_RETRIEVAL_RELEVANCE: float = 0.12
    HYBRID_SEARCH_ENABLED: bool = True
    HYBRID_MAX_CORPUS_CHUNKS: int = 5000

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
