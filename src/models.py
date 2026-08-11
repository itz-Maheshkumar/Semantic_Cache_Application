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
    """
    query: str
    response: str
    similarity_score: float
    cached_at: str
    matched_query: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert result to a standard dictionary."""
        return {
            "query": self.query,
            "response": self.response,
            "similarity_score": self.similarity_score,
            "cached_at": self.cached_at,
            "matched_query": self.matched_query,
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
