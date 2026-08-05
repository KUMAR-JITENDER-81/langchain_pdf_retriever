from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    OPENAI_API_KEY: str
    MODEL_NAME: str
    EMBEDDING_MODEL: str
    TEMPERATURE: float
    UPLOAD_DIR: str
    CHROMA_DIR: str
    LOG_LEVEL: str
    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore"
    )


settings = Settings()
