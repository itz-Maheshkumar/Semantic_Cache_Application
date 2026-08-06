"""
config.py — Centralized configuration for the Semantic Cache application.

All settings are read from environment variables (with sane defaults).
Copy `.env.example` to `.env` and fill in your values.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file if it exists (dev convenience)
load_dotenv()

# ── Base paths ──────────────────────────────────────────────────────────────
BASE_DIR: Path = Path(__file__).resolve().parent.parent
LOGS_DIR: Path = BASE_DIR / "logs"
DATA_DIR: Path = BASE_DIR / "data"

# Ensure directories exist at import time
LOGS_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

# ── Embedding ────────────────────────────────────────────────────────────────
EMBEDDING_MODEL: str = os.getenv(
    "EMBEDDING_MODEL", "all-MiniLM-L6-v2"
)

# ── Semantic Cache ───────────────────────────────────────────────────────────
# Cosine similarity threshold: queries above this score are treated as a hit.
SIMILARITY_THRESHOLD: float = float(
    os.getenv("SIMILARITY_THRESHOLD", "0.85")
)

# Paths for persisted FAISS index and metadata sidecar
FAISS_INDEX_PATH: Path = DATA_DIR / os.getenv(
    "FAISS_INDEX_FILE", "cache.faiss"
)
CACHE_METADATA_PATH: Path = DATA_DIR / os.getenv(
    "CACHE_METADATA_FILE", "cache_metadata.json"
)

# SQLite database for request logs
LOG_DB_PATH: Path = DATA_DIR / os.getenv(
    "LOG_DB_FILE", "cache_logs.db"
)

# ── OpenAI / LLM ────────────────────────────────────────────────────────────
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# Maximum number of retry attempts on transient API errors
LLM_MAX_RETRIES: int = int(os.getenv("LLM_MAX_RETRIES", "3"))

# ── Logging ──────────────────────────────────────────────────────────────────
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_FILE: Path = LOGS_DIR / "app.log"
LOG_MAX_BYTES: int = int(os.getenv("LOG_MAX_BYTES", str(5 * 1024 * 1024)))  # 5 MB
LOG_BACKUP_COUNT: int = int(os.getenv("LOG_BACKUP_COUNT", "3"))
