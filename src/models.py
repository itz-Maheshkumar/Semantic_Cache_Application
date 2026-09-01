"""
models.py — Data structures for Semantic Cache results and logging.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional


@dataclass
class CacheResult:
    """
    Represents the outcome of a semantic cache lookup on a cache hit.

    `similarity_score` is the score the hit was actually judged against: pure
    cosine similarity in vector-only mode, or the blended hybrid score when
    SemanticCache is running in hybrid (vector + BM25) mode. `vector_score`
    and `bm25_score` surface the two component signals individually — they
    are only populated on hits produced by hybrid search, and stay None
    otherwise, so vector-only callers see the exact same shape as before.
    """
    query: str
    response: str
    similarity_score: float
    cached_at: str
    matched_query: Optional[str] = None
    vector_score: Optional[float] = None
    bm25_score: Optional[float] = None

    def to_dict(self) -> dict:
        """Convert result to a standard dictionary."""
        return {
            "query": self.query,
            "response": self.response,
            "similarity_score": self.similarity_score,
            "cached_at": self.cached_at,
            "matched_query": self.matched_query,
            "vector_score": self.vector_score,
            "bm25_score": self.bm25_score,
        }


@dataclass
class RequestLog:
    """
    Represents a single recorded query request through the pipeline.
    """
    query: str
    response: str
    is_hit: bool
    similarity_score: float
    latency_ms: float
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    matched_query: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert log entry to a standard dictionary."""
        return {
            "query": self.query,
            "response": self.response,
            "is_hit": self.is_hit,
            "similarity_score": self.similarity_score,
            "latency_ms": self.latency_ms,
            "timestamp": self.timestamp,
            "matched_query": self.matched_query,
        }
