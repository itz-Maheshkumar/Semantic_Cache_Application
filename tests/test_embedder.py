"""
test_embedder.py — Unit tests for the Embedding Engine (src/embedder.py).
"""

import numpy as np
import pytest
from src.embedder import Embedder


@pytest.fixture(scope="module")
def embedder():
    """Module-level fixture to instantiate the Embedder model once across all tests."""
    return Embedder()


def test_embed_single_shape_and_type(embedder: Embedder):
    """Test that embedding a single text string produces a 1D float32 array with expected dimension."""
    text = "What is semantic caching?"
    vector = embedder.embed(text)

    assert isinstance(vector, np.ndarray)
    assert vector.ndim == 1
    assert vector.dtype == np.float32
    assert vector.shape[0] == embedder.embedding_dim
    assert vector.shape[0] == 384  # default all-MiniLM-L6-v2 dimension


def test_embed_vector_normalization(embedder: Embedder):
    """Test that generated vectors are L2 normalized (unit length, ||v|| == 1.0)."""
    text = "How does vector search work in FAISS?"
    vector = embedder.embed(text)
    norm = np.linalg.norm(vector)
    
    assert np.isclose(norm, 1.0, atol=1e-5)


def test_embed_batch(embedder: Embedder):
    """Test batch embedding generation."""
    texts = [
        "What is machine learning?",
        "Explain deep learning.",
        "How to train a neural network?"
    ]
    vectors = embedder.embed_batch(texts)

    assert isinstance(vectors, np.ndarray)
    assert vectors.ndim == 2
    assert vectors.shape == (3, embedder.embedding_dim)
    assert vectors.dtype == np.float32

    # Verify each vector in batch is normalized
    for vec in vectors:
        assert np.isclose(np.linalg.norm(vec), 1.0, atol=1e-5)


def test_semantic_similarity(embedder: Embedder):
    """
    Test semantic similarity behavior:
    Semantically similar queries should yield high dot product (cosine similarity),
    while unrelated queries yield low dot product.
    """
    q_base = "How do I reset my password?"
    q_similar = "I forgot my password, how can I restore it?"
    q_unrelated = "What is the capital of France?"

    v_base = embedder.embed(q_base)
    v_similar = embedder.embed(q_similar)
    v_unrelated = embedder.embed(q_unrelated)

    # Since vectors are L2-normalized, dot product == cosine similarity
    sim_similar = np.dot(v_base, v_similar)
    sim_unrelated = np.dot(v_base, v_unrelated)

    assert sim_similar > 0.70, f"Expected high similarity for paraphrased query, got {sim_similar:.4f}"
    assert sim_unrelated < 0.40, f"Expected low similarity for unrelated query, got {sim_unrelated:.4f}"


def test_empty_text_raises_value_error(embedder: Embedder):
    """Test that passing empty or whitespace-only strings raises ValueError."""
    with pytest.raises(ValueError):
        embedder.embed("")

    with pytest.raises(ValueError):
        embedder.embed("   ")

    with pytest.raises(ValueError):
        embedder.embed_batch(["valid query", ""])
