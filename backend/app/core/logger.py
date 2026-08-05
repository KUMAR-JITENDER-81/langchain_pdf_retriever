import logging
import os
# Create logs folder if it doesn't exist
os.makedirs("logs", exist_ok=True)
# Logger configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler("logs/app.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("financial_rag")