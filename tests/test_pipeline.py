"""
test_pipeline.py — Unit and integration tests for Module 4 (LLM Interface, Request Logger & Pipeline).
"""

from pathlib import Path
from unittest.mock import MagicMock
import pytest

from src.cache import SemanticCache
from src.embedder import Embedder
from src.llm import LLMClient
from src.models import RequestLog
from src.pipeline import CachePipeline
from src.request_logger import RequestLogger


@pytest.fixture(scope="module")
def embedder():
    """Shared Embedder instance for unit tests."""
    return Embedder()


@pytest.fixture
def temp_env(tmp_path: Path):
    """Fixture providing isolated temporary paths for SQLite DB, index, and metadata."""
    db_path = tmp_path / "test_logs.db"
    index_path = tmp_path / "test_cache.faiss"
    metadata_path = tmp_path / "test_metadata.json"
    return db_path, index_path, metadata_path


@pytest.fixture
def mock_llm():
    """Mock LLMClient that returns predictable responses without calling real APIs."""
    client = MagicMock(spec=LLMClient)
    client.generate.side_effect = lambda query, system_prompt=None: f"Mock response to: '{query}'"
    return client


def test_request_logger_basic(temp_env):
    """Test RequestLogger database insertion, retrieval, stats, and clear."""
    db_path, _, _ = temp_env
    logger = RequestLogger(db_path=db_path)

    assert len(logger.get_all()) == 0

    log1 = RequestLog(
        query="What is AI?",
        response="Artificial Intelligence.",
        is_hit=False,
        similarity_score=0.0,
        latency_ms=1200.5,
    )
    log2 = RequestLog(
        query="Tell me about AI?",
        response="Artificial Intelligence.",
        is_hit=True,
        similarity_score=0.91,
        latency_ms=4.2,
        matched_query="What is AI?",
    )

    logger.log(log1)
    logger.log(log2)

    logs = logger.get_all()
    assert len(logs) == 2
    assert logs[0].query == "Tell me about AI?"  # Most recent first
    assert logs[0].is_hit is True
    assert logs[1].is_hit is False

    stats = logger.get_stats()
    assert stats["total_requests"] == 2
    assert stats["total_hits"] == 1
    assert stats["total_misses"] == 1
    assert stats["hit_rate_pct"] == 50.0

    logger.clear()
    assert len(logger.get_all()) == 0


def test_pipeline_hit_miss_flow(embedder: Embedder, mock_llm: MagicMock, temp_env):
    """Test full CachePipeline flow: Miss -> Cache -> Hit -> Stats."""
    db_path, index_path, metadata_path = temp_env

    cache = SemanticCache(
        embedder=embedder,
        threshold=0.85,
        index_path=index_path,
        metadata_path=metadata_path,
        auto_load=False,
    )
    request_logger = RequestLogger(db_path=db_path)

    pipeline = CachePipeline(
        cache=cache,
        llm_client=mock_llm,
        request_logger=request_logger,
    )

    # 1. First query — should be a MISS
    q1 = "How do I reset my user password?"
    res1 = pipeline.process_query(q1)

    assert res1["is_hit"] is False
    assert res1["query"] == q1
    assert "Mock response to:" in res1["response"]
    assert mock_llm.generate.call_count == 1

    # 2. Second query (paraphrased) — should be a HIT
    q2 = "How can I reset my password?"
    res2 = pipeline.process_query(q2)

    assert res2["is_hit"] is True
    assert res2["similarity_score"] >= 0.85
    assert res2["response"] == res1["response"]
    # LLM should NOT have been called a second time
    assert mock_llm.generate.call_count == 1
    assert res2["latency_ms"] < 200.0  # Fast cached response

    # 3. Third query (unrelated) — should be a MISS
    q3 = "What is the capital of France?"
    res3 = pipeline.process_query(q3)

    assert res3["is_hit"] is False
    assert mock_llm.generate.call_count == 2

    # 4. Verify request logger metrics
    stats = request_logger.get_stats()
    assert stats["total_requests"] == 3
    assert stats["total_hits"] == 1
    assert stats["total_misses"] == 2
    assert round(stats["hit_rate_pct"], 1) == 33.3


def test_pipeline_empty_query_raises(temp_env):
    """Test that submitting empty or whitespace queries raises ValueError."""
    db_path, index_path, metadata_path = temp_env
    pipeline = CachePipeline(
        cache=MagicMock(spec=SemanticCache),
        llm_client=MagicMock(spec=LLMClient),
        request_logger=RequestLogger(db_path=db_path),
    )

    with pytest.raises(ValueError):
        pipeline.process_query("")

    with pytest.raises(ValueError):
        pipeline.process_query("   ")
