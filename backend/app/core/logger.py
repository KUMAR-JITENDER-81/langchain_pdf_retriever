import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from app.core.config import settings


log_directory = Path(settings.LOG_DIR)
log_directory.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        RotatingFileHandler(
            log_directory / "app.log",
            maxBytes=5 * 1024 * 1024,
            backupCount=3,
            encoding="utf-8",
        ),
        logging.StreamHandler(),
    ],
)

logger = logging.getLogger("langchain_pdf_retriever")
