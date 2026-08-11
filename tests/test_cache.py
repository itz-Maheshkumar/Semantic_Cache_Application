"""
test_cache.py — Unit tests for the Cache Engine (src/cache.py).
"""

import os
from pathlib import Path
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
