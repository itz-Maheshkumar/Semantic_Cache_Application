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
from src.modality import Modality
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


# ── Multi-Modal Caching: IMAGE modality ─────────────────────────────────────────
#
# Mirrors the FakeEmbedder approach above: a deterministic stand-in for
# src.image_embedder.ImageEmbedder, so these tests don't need a real CLIP
# model download and care only about SemanticCache's modality-dispatch
# behavior (auto-labeling, hybrid always forced off, BM25 skipped) rather
# than actual image content.

class FakeImageEmbedder:
    """
    Deterministic stand-in for src.image_embedder.ImageEmbedder. Same
    orthonormal-basis-vector trick as FakeEmbedder, keyed by the raw image
    bytes a test passes in (a real ImageEmbedder also accepts file paths and
    PIL.Image, but bytes are the simplest hashable stand-in for "an image"
    here — dict-keying only needs to distinguish one fake image from another).
    """

    def __init__(self, dim: int = 16):
        self._dim = dim
        self._vectors = {}
        self._next = 0

    def embed(self, image: bytes) -> np.ndarray:
        if image not in self._vectors:
            if self._next >= self._dim:
                raise ValueError("FakeImageEmbedder ran out of basis dimensions; increase dim.")
            vector = np.zeros(self._dim, dtype=np.float32)
            vector[self._next] = 1.0
            self._vectors[image] = vector
            self._next += 1
        return self._vectors[image]

    @property
    def embedding_dim(self) -> int:
        return self._dim


@pytest.fixture
def fake_image_embedder():
    return FakeImageEmbedder()


@pytest.fixture
def image_cache(fake_image_embedder, temp_cache_paths):
    """Fresh, empty IMAGE-modality SemanticCache for each test."""
    index_path, metadata_path = temp_cache_paths
    return SemanticCache(
        embedder=fake_image_embedder,
        modality=Modality.IMAGE,
        index_path=index_path,
        metadata_path=metadata_path,
        auto_load=False,
    )


def test_image_cache_hybrid_is_always_forced_off(fake_image_embedder, temp_cache_paths):
    """Hybrid/BM25 is keyword search over text and has no meaning for an
    image; it must stay off for an IMAGE cache even if a caller (or a live
    dashboard toggle) explicitly asks for it."""
    index_path, metadata_path = temp_cache_paths
    cache = SemanticCache(
        embedder=fake_image_embedder,
        modality=Modality.IMAGE,
        hybrid_enabled=True,
        index_path=index_path,
        metadata_path=metadata_path,
        auto_load=False,
    )
    assert cache.hybrid_enabled is False


def test_image_cache_put_and_get_auto_label(image_cache: SemanticCache):
    """A put()/get() with no explicit query_label should auto-derive a
    stable, displayable label from the image's content hash."""
    photo = b"fake-jpeg-bytes-of-a-cat"
    image_cache.put(photo, "A photo of a cat.", auto_save=False)

    stored_label = image_cache.metadata[0]["query"]
    assert stored_label.startswith("image:")
    assert stored_label == image_cache.display_query(photo, None)

    result = image_cache.get(photo)
    assert result is not None
    assert result.response == "A photo of a cat."
    assert result.matched_query == stored_label
    # No explicit label given at get()-time either, so the same auto-label
    # is used for the incoming query's display too.
    assert result.query == stored_label


def test_image_cache_put_and_get_explicit_label(image_cache: SemanticCache):
    """An explicit query_label should be stored/echoed verbatim instead of
    the auto-generated hash label."""
    photo = b"fake-png-bytes-of-a-dog"
    image_cache.put(photo, "A photo of a dog.", auto_save=False, query_label="dog.png")

    assert image_cache.metadata[0]["query"] == "dog.png"

    result = image_cache.get(photo, query_label="dog.png")
    assert result is not None
    assert result.query == "dog.png"
    assert result.matched_query == "dog.png"


def test_image_cache_auto_label_is_stable_for_identical_content(image_cache: SemanticCache):
    """Two lookups of the exact same bytes should get the exact same
    auto-generated label (useful for logs/dedup), and different content
    should get a different label."""
    photo = b"identical-bytes"
    other_photo = b"different-bytes"

    assert image_cache.display_query(photo, None) == image_cache.display_query(photo, None)
    assert image_cache.display_query(photo, None) != image_cache.display_query(other_photo, None)


def test_display_query_ignores_label_for_text_modality(cache: SemanticCache):
    """query_label is a non-text-modality concept; a TEXT cache should
    always echo the query string itself regardless of what's passed."""
    assert cache.display_query("hello world", "some label") == "hello world"


def test_image_cache_semantic_hit_by_vector_similarity(fake_image_embedder, temp_cache_paths):
    """Vector-only lookup still works for IMAGE modality exactly like TEXT:
    a query embedding to the same basis vector as a stored one is a HIT."""
    index_path, metadata_path = temp_cache_paths
    cache = SemanticCache(
        embedder=fake_image_embedder, modality=Modality.IMAGE, threshold=0.85,
        index_path=index_path, metadata_path=metadata_path, auto_load=False,
    )
    photo = b"a-cached-photo"
    cache.put(photo, "cached description", auto_save=False)

    # FakeImageEmbedder gives identical bytes an identical vector, so
    # looking the same bytes up again is a near-1.0 cosine-similarity hit.
    result = cache.get(photo)
    assert result is not None
    assert result.response == "cached description"

    unrelated_photo = b"a-totally-different-photo"
    assert cache.get(unrelated_photo) is None


def test_rebuild_bm25_is_a_noop_for_image_modality(image_cache: SemanticCache):
    """BM25 has no meaning for images; the keyword index should stay empty
    no matter how many images are put() into an IMAGE cache."""
    image_cache.put(b"photo one", "response one", auto_save=False)
    image_cache.put(b"photo two", "response two", auto_save=False)

    assert image_cache.bm25_index.is_empty


def test_image_cache_eviction_still_works(fake_image_embedder, temp_cache_paths):
    """TTL/LRU eviction operates on timestamps/vectors, not on the query's
    type, so it should apply to an IMAGE cache exactly like a TEXT one."""
    index_path, metadata_path = temp_cache_paths
    cache = SemanticCache(
        embedder=fake_image_embedder, modality=Modality.IMAGE, max_size=1,
        index_path=index_path, metadata_path=metadata_path, auto_load=False,
    )
    cache.put(b"photo A", "response A", auto_save=False)
    cache.put(b"photo B", "response B", auto_save=False)  # put() enforces capacity

    assert cache.size == 1
    assert cache.get(b"photo A") is None
    assert cache.get(b"photo B") is not None
