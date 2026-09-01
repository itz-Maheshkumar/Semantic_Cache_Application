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

# ── Hybrid Search (Vector + BM25) ─────────────────────────────────────────────
# When enabled, cache lookups blend FAISS cosine similarity with BM25 keyword
# scoring instead of relying on vector similarity alone — this lets exact
# terms, acronyms, and IDs that the embedding model tends to blur together
# still surface a strong match. Off by default so plain vector-only lookups
# (and SIMILARITY_THRESHOLD above) keep behaving exactly as before unless a
# caller opts in.
ENABLE_HYBRID_SEARCH: bool = os.getenv("ENABLE_HYBRID_SEARCH", "false").strip().lower() in (
    "1", "true", "yes", "on"
)

# Relative weight given to each signal when blending:
#     hybrid_score = VECTOR_WEIGHT * cosine_similarity + BM25_WEIGHT * bm25_norm
# Both scores are on a 0..1 scale (cosine similarity naturally; BM25 via
# min-max normalization — see src/hybrid_search.py), so keeping the weights
# summed to 1.0 keeps hybrid_score on a comparable 0..1 scale too.
VECTOR_WEIGHT: float = float(os.getenv("VECTOR_WEIGHT", "0.6"))
BM25_WEIGHT: float = float(os.getenv("BM25_WEIGHT", "0.4"))

# How many top candidates each retriever (FAISS, BM25) contributes to the
# merged candidate pool before ranking. Bounds the cost of the BM25 scoring
# pass (a pure-Python, whole-corpus scan) on every lookup.
HYBRID_TOP_K: int = int(os.getenv("HYBRID_TOP_K", "10"))

# Minimum blended hybrid score (0.0 - 1.0) required to accept a cache hit in
# hybrid mode. Kept separate from SIMILARITY_THRESHOLD because it is measured
# on a different scale (a weighted mix of cosine similarity and normalized
# BM25, not pure cosine similarity).
HYBRID_SIMILARITY_THRESHOLD: float = float(os.getenv("HYBRID_SIMILARITY_THRESHOLD", "0.55"))

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
