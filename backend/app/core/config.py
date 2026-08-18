from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    OPENAI_API_KEY: str
    MODEL_NAME: str
    EMBEDDING_MODEL: str
    TEMPERATURE: float
    UPLOAD_DIR: str
    CHROMA_DIR: str
    LOG_LEVEL: str
    MAX_UPLOAD_SIZE_MB: float = 10.0
    FRONTEND_ORIGINS: str = "http://127.0.0.1:5173,http://localhost:5173"
    API_AUTH_TOKEN: str = ""
    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore"
    )


settings = Settings()
