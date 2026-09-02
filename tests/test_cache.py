"""
test_cache.py — Unit tests for the Cache Engine (src/cache.py).
"""

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

from src.cache import SemanticCache
from src.embedder import Embedder
from src.models import CacheResult


@pytest.fixture(scope="module")
def embedder():
    """Module-level fixture to share a single Embedder instance across tests."""
    return Embedder()


@pytest.fixture
def temp_cache_paths(tmp_path: Path):
    """Fixture providing isolated temporary paths for index and metadata files."""
    index_path = tmp_path / "test_cache.faiss"
    metadata_path = tmp_path / "test_metadata.json"
    return index_path, metadata_path


@pytest.fixture
def cache(embedder: Embedder, temp_cache_paths):
    """Fixture providing a fresh, empty SemanticCache instance for each test."""
    index_path, metadata_path = temp_cache_paths
    return SemanticCache(
        embedder=embedder,
        threshold=0.85,
        index_path=index_path,
        metadata_path=metadata_path,
        auto_load=False,
    )


def test_cache_initial_empty(cache: SemanticCache):
    """Test that a newly initialized cache is empty and returns None on lookup."""
    assert cache.is_empty
    assert cache.size == 0
    assert cache.get("What is artificial intelligence?") is None


def test_cache_exact_hit(cache: SemanticCache):
    """Test that querying the exact same stored string produces a cache HIT with ~1.0 similarity."""
    query = "What is the capital of France?"
    response = "The capital of France is Paris."

    cache.put(query, response, auto_save=False)

    assert not cache.is_empty
    assert cache.size == 1

    result = cache.get(query)
    assert result is not None
    assert isinstance(result, CacheResult)
    assert result.query == query
    assert result.response == response
    assert result.similarity_score >= 0.99
    assert result.matched_query == query


def test_cache_semantic_hit(cache: SemanticCache):
    """Test that a semantically similar query (paraphrase) triggers a cache HIT."""
    stored_query = "How do I reset my account password?"
    stored_response = "Go to Settings -> Account -> Password Reset."

    cache.put(stored_query, stored_response, auto_save=False)

    similar_query = "I forgot my password, how can I change it?"
    result = cache.get(similar_query)

    assert result is not None
    assert result.response == stored_response
    assert result.similarity_score >= cache.threshold
    assert result.matched_query == stored_query


def test_cache_dissimilar_miss(cache: SemanticCache):
    """Test that an unrelated query triggers a cache MISS (returns None)."""
    stored_query = "How do I reset my account password?"
    stored_response = "Go to Settings -> Account -> Password Reset."

    cache.put(stored_query, stored_response, auto_save=False)

    unrelated_query = "What is the distance from Earth to the Moon?"
    result = cache.get(unrelated_query)

    assert result is None


def test_cache_threshold_boundary(embedder: Embedder, temp_cache_paths):
    """Test custom similarity threshold behavior."""
    index_path, metadata_path = temp_cache_paths
    strict_cache = SemanticCache(
        embedder=embedder,
        threshold=0.98,  # Very strict threshold
        index_path=index_path,
        metadata_path=metadata_path,
        auto_load=False,
    )

    strict_cache.put("How do I reset my password?", "Go to settings.", auto_save=False)

    # A paraphrased query will be > 0.85 but < 0.98, so strict cache should miss
    similar_query = "I forgot my password, how can I change it?"
    assert strict_cache.get(similar_query) is None


def test_cache_persistence_save_and_load(embedder: Embedder, temp_cache_paths):
    """Test persisting cache to disk and loading it back into a new cache instance."""
    index_path, metadata_path = temp_cache_paths

    cache1 = SemanticCache(
        embedder=embedder,
        threshold=0.85,
        index_path=index_path,
        metadata_path=metadata_path,
        auto_load=False,
    )

    q1, r1 = "What is Python?", "Python is a programming language."
    q2, r2 = "What is FAISS?", "FAISS is a vector similarity search library."

    cache1.put(q1, r1, auto_save=True)
    cache1.put(q2, r2, auto_save=True)

    assert index_path.exists()
    assert metadata_path.exists()

    # Create a new cache instance pointing to the same files
    cache2 = SemanticCache(
        embedder=embedder,
        threshold=0.85,
        index_path=index_path,
        metadata_path=metadata_path,
        auto_load=True,
    )

    assert cache2.size == 2
    res1 = cache2.get("What is Python programming language?")
    res2 = cache2.get("Can you explain what FAISS is?")

    assert res1 is not None and res1.response == r1
    assert res2 is not None and res2.response == r2


def test_cache_clear(cache: SemanticCache, temp_cache_paths):
    """Test clearing cache memory and deleting disk files."""
    index_path, metadata_path = temp_cache_paths

    cache.put("Test query", "Test response", auto_save=True)
    assert cache.size == 1

    cache.clear(delete_files=True)
    assert cache.is_empty
    assert cache.size == 0
    assert not index_path.exists()
    assert not metadata_path.exists()


# ── Cache Eviction: TTL (Time-To-Live) + LRU (Least-Recently-Used) ─────────────
#
# These use a small deterministic FakeEmbedder (orthonormal basis vectors, one
# per distinct query text) instead of the real sentence-transformers model
# used above — eviction tests care about *which* entries survive and in what
# order, not approximate cosine similarity, and this keeps them fast and
# independent of the real model's exact behavior. Where a test needs to
# simulate the passage of time, it backdates `cached_at`/`last_accessed_at`
# directly on `cache.metadata` rather than sleeping.

class FakeEmbedder:
    """
    Deterministic stand-in for src.embedder.Embedder: hands back a unique
    orthonormal basis vector per distinct text (first call for a given
    string claims the next unused dimension). Identical text always embeds
    identically (cosine similarity 1.0); any two distinct texts are exactly
    orthogonal (cosine similarity 0.0).
    """

    def __init__(self, dim: int = 32):
        self._dim = dim
        self._vectors = {}
        self._next = 0

    def embed(self, text: str) -> np.ndarray:
        if text not in self._vectors:
            if self._next >= self._dim:
                raise ValueError("FakeEmbedder ran out of basis dimensions; increase dim.")
            vector = np.zeros(self._dim, dtype=np.float32)
            vector[self._next] = 1.0
            self._vectors[text] = vector
            self._next += 1
        return self._vectors[text]

    @property
    def embedding_dim(self) -> int:
        return self._dim


@pytest.fixture
def fake_embedder():
    return FakeEmbedder()


def backdate(entry: dict, seconds_ago: float, field: str = "cached_at") -> None:
    """Rewrite a metadata entry's timestamp field to `seconds_ago` in the past."""
    past = datetime.now(timezone.utc) - timedelta(seconds=seconds_ago)
    entry[field] = past.isoformat()


def test_ttl_disabled_by_default(fake_embedder, temp_cache_paths):
    index_path, metadata_path = temp_cache_paths
    cache = SemanticCache(
        embedder=fake_embedder, index_path=index_path, metadata_path=metadata_path, auto_load=False,
    )
    assert cache.ttl_seconds == 0

    cache.put("old query", "old response", auto_save=False)
    backdate(cache.metadata[0], seconds_ago=999_999)

    assert cache.prune_expired() == 0
    assert cache.size == 1


def test_ttl_prunes_only_entries_older_than_the_limit(fake_embedder, temp_cache_paths):
    index_path, metadata_path = temp_cache_paths
    cache = SemanticCache(
        embedder=fake_embedder, ttl_seconds=60,
        index_path=index_path, metadata_path=metadata_path, auto_load=False,
    )

    cache.put("stale query", "stale response", auto_save=False)
    cache.put("fresh query", "fresh response", auto_save=False)
    backdate(cache.metadata[0], seconds_ago=3600)  # 1 hour old, TTL is 60s

    removed = cache.prune_expired()

    assert removed == 1
    assert cache.size == 1
    assert cache.metadata[0]["query"] == "fresh query"
    assert cache.index.ntotal == 1  # FAISS index stayed in sync with metadata

    assert cache.get("stale query") is None
    result = cache.get("fresh query")
    assert result is not None
    assert result.response == "fresh response"


def test_ttl_prune_preserves_order_and_bm25_alignment(fake_embedder, temp_cache_paths):
    """Evicting a *middle* entry must not misalign the surviving entries
    between the FAISS index, metadata list, and BM25 index."""
    index_path, metadata_path = temp_cache_paths
    cache = SemanticCache(
        embedder=fake_embedder, ttl_seconds=60,
        index_path=index_path, metadata_path=metadata_path, auto_load=False,
    )

    cache.put("first query", "first response", auto_save=False)
    cache.put("middle query", "middle response", auto_save=False)
    cache.put("last query", "last response", auto_save=False)
    backdate(cache.metadata[1], seconds_ago=3600)  # expire the middle one only

    removed = cache.prune_expired()

    assert removed == 1
    assert [e["query"] for e in cache.metadata] == ["first query", "last query"]
    assert cache.get("first query").response == "first response"
    assert cache.get("last query").response == "last response"
    assert cache.get("middle query") is None


def test_prune_expired_noop_on_empty_cache(fake_embedder, temp_cache_paths):
    index_path, metadata_path = temp_cache_paths
    cache = SemanticCache(
        embedder=fake_embedder, ttl_seconds=60,
        index_path=index_path, metadata_path=metadata_path, auto_load=False,
    )
    assert cache.prune_expired() == 0


def test_put_runs_ttl_pruning_automatically(fake_embedder, temp_cache_paths):
    index_path, metadata_path = temp_cache_paths
    cache = SemanticCache(
        embedder=fake_embedder, ttl_seconds=60,
        index_path=index_path, metadata_path=metadata_path, auto_load=False,
    )

    cache.put("old query", "old response", auto_save=False)
    backdate(cache.metadata[0], seconds_ago=3600)

    # put() itself should prune the now-expired entry — no manual
    # prune_expired() call needed.
    cache.put("new query", "new response", auto_save=False)

    assert cache.size == 1
    assert cache.metadata[0]["query"] == "new query"


def test_lru_disabled_by_default(fake_embedder, temp_cache_paths):
    index_path, metadata_path = temp_cache_paths
    cache = SemanticCache(
        embedder=fake_embedder, index_path=index_path, metadata_path=metadata_path, auto_load=False,
    )
    assert cache.max_size == 0

    for i in range(5):
        cache.put(f"query {i}", f"response {i}", auto_save=False)

    assert cache.enforce_capacity() == 0
    assert cache.size == 5


def test_lru_evicts_oldest_insertion_when_never_re_accessed(fake_embedder, temp_cache_paths):
    index_path, metadata_path = temp_cache_paths
    cache = SemanticCache(
        embedder=fake_embedder, max_size=2,
        index_path=index_path, metadata_path=metadata_path, auto_load=False,
    )

    cache.put("query A", "response A", auto_save=False)
    cache.put("query B", "response B", auto_save=False)
    cache.put("query C", "response C", auto_save=False)  # put() enforces capacity automatically

    assert cache.size == 2
    assert [e["query"] for e in cache.metadata] == ["query B", "query C"]
    assert cache.get("query A") is None
    assert cache.get("query B") is not None
    assert cache.get("query C") is not None


def test_lru_a_cache_hit_protects_an_entry_from_eviction(fake_embedder, temp_cache_paths):
    """The whole point of LRU over plain insertion-order pruning: touching an
    old entry via a cache HIT should save it from being the next evicted."""
    index_path, metadata_path = temp_cache_paths
    cache = SemanticCache(
        embedder=fake_embedder, max_size=2,
        index_path=index_path, metadata_path=metadata_path, auto_load=False,
    )

    cache.put("query A", "response A", auto_save=False)
    cache.put("query B", "response B", auto_save=False)

    # Touch A — it's now the most recently *used* entry, even though B was
    # inserted more recently.
    assert cache.get("query A") is not None

    cache.put("query C", "response C", auto_save=False)  # size would be 3 > max_size=2

    assert cache.size == 2
    # B, never touched after insertion, should be the one evicted — not A.
    assert cache.get("query B") is None
    assert cache.get("query A") is not None
    assert cache.get("query C") is not None


def test_enforce_capacity_noop_when_within_limit(fake_embedder, temp_cache_paths):
    index_path, metadata_path = temp_cache_paths
    cache = SemanticCache(
        embedder=fake_embedder, max_size=10,
        index_path=index_path, metadata_path=metadata_path, auto_load=False,
    )
    cache.put("query A", "response A", auto_save=False)
    assert cache.enforce_capacity() == 0
    assert cache.size == 1


def test_eviction_on_load_prunes_and_persists(fake_embedder, temp_cache_paths):
    """A cache reloaded from disk should have TTL/LRU applied immediately —
    e.g. after lowering CACHE_TTL_SECONDS or CACHE_MAX_SIZE between runs —
    and the cleanup should be written back to disk."""
    index_path, metadata_path = temp_cache_paths

    writer = SemanticCache(
        embedder=fake_embedder, index_path=index_path, metadata_path=metadata_path, auto_load=False,
    )
    writer.put("stale query", "stale response", auto_save=False)
    writer.put("fresh query", "fresh response", auto_save=False)
    backdate(writer.metadata[0], seconds_ago=3600)
    writer.save()

    reader = SemanticCache(
        embedder=fake_embedder, ttl_seconds=60,
        index_path=index_path, metadata_path=metadata_path, auto_load=True,
    )
    assert reader.size == 1
    assert reader.metadata[0]["query"] == "fresh query"

    with open(metadata_path, "r", encoding="utf-8") as f:
        persisted = json.load(f)
    assert len(persisted) == 1
    assert persisted[0]["query"] == "fresh query"


def test_load_backfills_missing_last_accessed_at(fake_embedder, temp_cache_paths):
    """Metadata saved before eviction support existed won't have
    `last_accessed_at` — load() must backfill it (from `cached_at`) rather
    than KeyError the first time LRU tries to sort by it."""
    index_path, metadata_path = temp_cache_paths

    writer = SemanticCache(
        embedder=fake_embedder, index_path=index_path, metadata_path=metadata_path, auto_load=False,
    )
    writer.put("legacy query", "legacy response", auto_save=False)
    del writer.metadata[0]["last_accessed_at"]  # simulate a pre-eviction-feature save
    writer.save()

    reader = SemanticCache(
        embedder=fake_embedder, max_size=5,
        index_path=index_path, metadata_path=metadata_path, auto_load=True,
    )
    assert "last_accessed_at" in reader.metadata[0]
    assert reader.enforce_capacity() == 0  # doesn't raise, and nothing to evict
