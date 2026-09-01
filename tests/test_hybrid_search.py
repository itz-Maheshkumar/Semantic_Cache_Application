"""
test_hybrid_search.py — Tests for BM25 keyword search and vector+BM25 score
fusion (src/bm25_index.py, src/hybrid_search.py), plus SemanticCache's
opt-in hybrid retrieval mode (src/cache.py).

The pure scoring/fusion tests need no embedding model at all. The
SemanticCache-level tests use a small deterministic FakeEmbedder (defined
below) instead of the real sentence-transformers model, so hybrid-mode
behavior — including the "BM25 rescues a borderline vector match" case that
is the whole point of hybrid search — can be verified with exact, known
scores rather than model-dependent approximations.
"""

import math
from pathlib import Path

import numpy as np
import pytest

from src.bm25_index import BM25Index, tokenize
from src.cache import SemanticCache
from src.hybrid_search import blend, normalize_bm25_scores


# ── tokenize() ────────────────────────────────────────────────────────────────

def test_tokenize_lowercases_and_splits_on_non_alphanumeric():
    assert tokenize("Reset My Password!") == ["reset", "password"]


def test_tokenize_strips_stopwords():
    tokens = tokenize("What is the capital of France?")
    assert "capital" in tokens
    assert "france" in tokens
    assert "what" not in tokens
    assert "is" not in tokens
    assert "the" not in tokens
    assert "of" not in tokens


def test_tokenize_empty_string_returns_empty_list():
    assert tokenize("") == []
    assert tokenize("   ") == []


# ── BM25Index ─────────────────────────────────────────────────────────────────

def test_bm25_index_empty_before_rebuild():
    idx = BM25Index()
    assert idx.is_empty
    assert idx.search("anything", top_k=5) == []


def test_bm25_index_rebuild_with_empty_list_resets_to_empty():
    idx = BM25Index()
    idx.rebuild(["some query", "another query"])
    assert not idx.is_empty

    idx.rebuild([])
    assert idx.is_empty


def test_bm25_index_finds_exact_keyword_match():
    idx = BM25Index()
    idx.rebuild([
        "How do I reset my account password?",
        "What is the weather like today?",
        "How do I cancel my subscription?",
    ])

    results = idx.search("password reset help", top_k=3)
    assert results  # at least one scored candidate
    top_index, top_score = results[0]
    assert top_index == 0  # the password document should rank first
    assert top_score > 0


def test_bm25_index_unrelated_query_scores_zero():
    idx = BM25Index()
    idx.rebuild(["How do I reset my account password?"])

    # No shared content tokens at all with the indexed document.
    results = idx.search("weather forecast tomorrow", top_k=1)
    assert results == [(0, 0.0)]


def test_bm25_index_respects_top_k():
    idx = BM25Index()
    idx.rebuild([f"document number {i} about cats" for i in range(10)])
    results = idx.search("cats", top_k=3)
    assert len(results) == 3


# ── normalize_bm25_scores() ────────────────────────────────────────────────────

def test_normalize_bm25_scores_empty_input():
    assert normalize_bm25_scores({}) == {}


def test_normalize_bm25_scores_min_max_scaling():
    raw = {0: 1.0, 1: 3.0, 2: 5.0}
    normalized = normalize_bm25_scores(raw)
    assert normalized[0] == pytest.approx(0.0)
    assert normalized[1] == pytest.approx(0.5)
    assert normalized[2] == pytest.approx(1.0)


def test_normalize_bm25_scores_single_candidate_positive_saturates_to_one():
    assert normalize_bm25_scores({0: 2.5}) == {0: 1.0}


def test_normalize_bm25_scores_single_candidate_zero_stays_zero():
    assert normalize_bm25_scores({0: 0.0}) == {0: 0.0}


def test_normalize_bm25_scores_all_equal_nonzero_saturate_to_one():
    normalized = normalize_bm25_scores({0: 4.0, 1: 4.0, 2: 4.0})
    assert normalized == {0: 1.0, 1: 1.0, 2: 1.0}


# ── blend() ────────────────────────────────────────────────────────────────────

def test_blend_weighted_combination_is_exact():
    vector_scores = {0: 0.9, 1: 0.5}
    bm25_scores = {1: 5.0, 2: 2.0}  # candidate 1 shared by both retrievers

    candidates = blend(vector_scores, bm25_scores, vector_weight=0.6, bm25_weight=0.4)
    by_index = {c.index: c for c in candidates}

    # normalize_bm25_scores({1: 5.0, 2: 2.0}) -> {1: 1.0, 2: 0.0}
    assert by_index[0].hybrid_score == pytest.approx(0.6 * 0.9 + 0.4 * 0.0)
    assert by_index[1].hybrid_score == pytest.approx(0.6 * 0.5 + 0.4 * 1.0)
    assert by_index[2].hybrid_score == pytest.approx(0.6 * 0.0 + 0.4 * 0.0)


def test_blend_sorts_descending_by_hybrid_score():
    vector_scores = {0: 0.9, 1: 0.4, 2: 0.1}
    bm25_scores = {0: 1.0, 1: 8.0, 2: 0.0}

    candidates = blend(vector_scores, bm25_scores, vector_weight=0.5, bm25_weight=0.5)

    scores = [c.hybrid_score for c in candidates]
    assert scores == sorted(scores, reverse=True)
    # Candidate 1 (moderate vector score, but the single strongest BM25 match)
    # should outrank candidate 0 (higher vector score, weakest BM25 match).
    assert candidates[0].index == 1


def test_blend_candidate_found_by_only_one_retriever_is_not_dropped():
    vector_scores = {0: 0.8}
    bm25_scores = {1: 3.0}  # a different candidate, found only via keywords

    candidates = blend(vector_scores, bm25_scores, vector_weight=0.6, bm25_weight=0.4)
    indices = {c.index for c in candidates}
    assert indices == {0, 1}


def test_blend_empty_inputs_returns_empty_list():
    assert blend({}, {}, 0.6, 0.4) == []


# ── SemanticCache hybrid mode (deterministic FakeEmbedder) ─────────────────────

class FakeEmbedder:
    """
    Deterministic stand-in for src.embedder.Embedder.

    Real cosine similarity is hard to predict without running the actual
    sentence-transformers model (unavailable offline). This fake instead maps
    each expected input string to a hand-picked, pre-normalized 2D vector, so
    tests can assert exact hybrid scores instead of hoping a real model
    produces a similarity in some plausible range.

    Two unit vectors at angle theta apart have cosine similarity cos(theta),
    which is what `angle()` below is for.
    """

    def __init__(self, vectors: dict):
        self._vectors = vectors

    def embed(self, text: str) -> np.ndarray:
        if text not in self._vectors:
            raise KeyError(f"FakeEmbedder has no vector configured for: {text!r}")
        return self._vectors[text].astype(np.float32)

    @property
    def embedding_dim(self) -> int:
        return 2


def unit_vector(theta_rad: float) -> np.ndarray:
    """A 2D unit vector at angle `theta_rad` from [1, 0]."""
    return np.array([math.cos(theta_rad), math.sin(theta_rad)], dtype=np.float32)


def angle_for_cosine(cosine: float) -> float:
    """The angle (radians) between two unit vectors with the given cosine similarity."""
    return math.acos(cosine)


@pytest.fixture
def hybrid_cache_paths(tmp_path: Path):
    return tmp_path / "hybrid.faiss", tmp_path / "hybrid_metadata.json"


def test_hybrid_disabled_by_default_matches_original_vector_only_shape(hybrid_cache_paths):
    """With no hybrid arguments passed, SemanticCache behaves exactly as it
    did before this feature existed: pure vector similarity, and CacheResult
    carries no component scores."""
    index_path, metadata_path = hybrid_cache_paths
    stored, query = "stored query", "stored query"  # identical -> perfect match
    embedder = FakeEmbedder({stored: unit_vector(0.0), query: unit_vector(0.0)})

    cache = SemanticCache(
        embedder=embedder, threshold=0.85,
        index_path=index_path, metadata_path=metadata_path, auto_load=False,
    )
    assert cache.hybrid_enabled is False

    cache.put(stored, "the answer", auto_save=False)
    result = cache.get(query)

    assert result is not None
    assert result.similarity_score == pytest.approx(1.0)
    assert result.vector_score is None
    assert result.bm25_score is None


def test_hybrid_rescues_a_borderline_vector_match_via_keyword_overlap(hybrid_cache_paths):
    """The central claim of hybrid search: a query that shares a distinctive
    keyword (here, an order ID) with a cached query, but whose embedding
    isn't similar enough to pass on vector similarity alone, should still hit
    once BM25 is blended in.

    BM25's IDF is meaningless with only one document in the corpus (every
    term looks equally "common" when there's nothing to contrast it with),
    so this uses three cached entries — one relevant, two unrelated fillers —
    which is enough for "zx9981" to stand out as a distinctive term.
    """
    index_path, metadata_path = hybrid_cache_paths

    doc_target = "What is the order status for id ZX9981?"
    doc_noise1 = "How do I reset my account password?"
    doc_noise2 = "What is the refund policy for shipping delays?"
    query = "checking up on my order ZX9981 today"

    # Vector similarity to doc_target is exactly 0.5 — comfortably below a
    # vector-only SIMILARITY_THRESHOLD (0.85 default) — and the noise docs
    # sit even farther away, so doc_target is still the best vector match,
    # just not a good enough one on vector similarity alone.
    gap = angle_for_cosine(0.5)
    embedder = FakeEmbedder({
        doc_target: unit_vector(0.0),
        doc_noise1: unit_vector(math.radians(180)),
        doc_noise2: unit_vector(math.radians(250)),
        query: unit_vector(gap),
    })
    docs = [doc_target, doc_noise1, doc_noise2]

    vector_only_cache = SemanticCache(
        embedder=embedder, threshold=0.85, hybrid_enabled=False,
        index_path=index_path, metadata_path=metadata_path, auto_load=False,
    )
    for doc in docs:
        vector_only_cache.put(doc, f"response to: {doc}", auto_save=False)
    assert vector_only_cache.get(query) is None  # vector-only: MISS

    hybrid_cache = SemanticCache(
        embedder=embedder, hybrid_enabled=True, hybrid_threshold=0.55,
        vector_weight=0.6, bm25_weight=0.4,
        index_path=index_path, metadata_path=metadata_path.with_name("hybrid2.json"),
        auto_load=False,
    )
    for doc in docs:
        hybrid_cache.put(doc, f"response to: {doc}", auto_save=False)
    result = hybrid_cache.get(query)

    assert result is not None  # hybrid: HIT, thanks to the shared "zx9981" token
    assert result.matched_query == doc_target
    assert result.vector_score == pytest.approx(0.5, abs=1e-4)
    assert result.bm25_score > 0.0
    # hybrid_score should be exactly the weighted blend of the two components.
    assert result.similarity_score == pytest.approx(
        0.6 * result.vector_score + 0.4 * result.bm25_score, abs=1e-4
    )


def test_hybrid_still_misses_when_neither_signal_is_strong(hybrid_cache_paths):
    """No shared keywords and low vector similarity -> MISS even in hybrid mode."""
    index_path, metadata_path = hybrid_cache_paths

    stored = "How do I reset my account password?"
    query = "recommend a good science fiction movie"

    embedder = FakeEmbedder({stored: unit_vector(0.0), query: unit_vector(math.pi / 2)})

    cache = SemanticCache(
        embedder=embedder, hybrid_enabled=True, hybrid_threshold=0.55,
        index_path=index_path, metadata_path=metadata_path, auto_load=False,
    )
    cache.put(stored, "Go to settings.", auto_save=False)

    assert cache.get(query) is None


def test_hybrid_picks_the_better_of_two_candidates(hybrid_cache_paths):
    """With multiple cached entries, hybrid search should select the one with
    the higher blended score, not just whichever was inserted first."""
    index_path, metadata_path = hybrid_cache_paths

    doc_a = "How do I reset my password?"
    doc_b = "What is the refund policy for order ZX9981?"
    doc_c = "What is the weather forecast for tomorrow?"
    query = "checking the refund policy for my order ZX9981"

    embedder = FakeEmbedder({
        doc_a: unit_vector(math.radians(150)),   # unrelated to the query
        doc_b: unit_vector(angle_for_cosine(0.6)),
        doc_c: unit_vector(math.radians(-150)),  # unrelated to the query
        query: unit_vector(0.0),
    })

    cache = SemanticCache(
        embedder=embedder, hybrid_enabled=True, hybrid_threshold=0.55,
        index_path=index_path, metadata_path=metadata_path, auto_load=False,
    )
    cache.put(doc_a, "Go to settings.", auto_save=False)
    cache.put(doc_b, "Refunds are processed within 5 business days.", auto_save=False)
    cache.put(doc_c, "Expect sunshine tomorrow.", auto_save=False)

    result = cache.get(query)
    assert result is not None
    assert result.matched_query == doc_b


def test_clear_resets_bm25_index_alongside_vector_index(hybrid_cache_paths):
    index_path, metadata_path = hybrid_cache_paths
    embedder = FakeEmbedder({"q": unit_vector(0.0)})

    cache = SemanticCache(
        embedder=embedder, hybrid_enabled=True,
        index_path=index_path, metadata_path=metadata_path, auto_load=False,
    )
    cache.put("q", "a", auto_save=False)
    assert not cache.bm25_index.is_empty

    cache.clear(delete_files=False)
    assert cache.bm25_index.is_empty
    assert cache.is_empty


def test_hybrid_survives_save_and_load_roundtrip(hybrid_cache_paths):
    """BM25 isn't persisted directly — it's derived from `metadata` — so a
    freshly loaded cache must rebuild it before its first hybrid lookup."""
    index_path, metadata_path = hybrid_cache_paths
    stored, query = "order status for ZX9981", "order status for ZX9981"
    embedder = FakeEmbedder({stored: unit_vector(0.0), query: unit_vector(0.0)})

    cache1 = SemanticCache(
        embedder=embedder, hybrid_enabled=True,
        index_path=index_path, metadata_path=metadata_path, auto_load=False,
    )
    cache1.put(stored, "Shipped.", auto_save=True)

    cache2 = SemanticCache(
        embedder=embedder, hybrid_enabled=True,
        index_path=index_path, metadata_path=metadata_path, auto_load=True,
    )
    assert not cache2.bm25_index.is_empty
    result = cache2.get(query)
    assert result is not None
    assert result.response == "Shipped."


def test_hybrid_pipeline_passthrough(hybrid_cache_paths):
    """CachePipeline.process_query should surface the component vector/BM25
    scores for a hybrid hit, and report them as None on a miss."""
    from unittest.mock import MagicMock

    from src.llm import LLMClient
    from src.pipeline import CachePipeline
    from src.request_logger import RequestLogger

    index_path, metadata_path = hybrid_cache_paths
    stored, query = "reset my password please", "reset my password please"
    embedder = FakeEmbedder({stored: unit_vector(0.0), query: unit_vector(0.0)})

    cache = SemanticCache(
        embedder=embedder, hybrid_enabled=True,
        index_path=index_path, metadata_path=metadata_path, auto_load=False,
    )
    cache.put(stored, "Go to settings.", auto_save=False)

    mock_llm = MagicMock(spec=LLMClient)
    request_logger = RequestLogger(db_path=metadata_path.with_name("logs.db"))
    pipeline = CachePipeline(cache=cache, llm_client=mock_llm, request_logger=request_logger)

    result = pipeline.process_query(query)
    assert result["is_hit"] is True
    assert result["vector_score"] == pytest.approx(1.0)
    assert result["bm25_score"] is not None
