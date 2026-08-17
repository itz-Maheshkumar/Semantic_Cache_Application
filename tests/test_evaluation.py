"""
test_evaluation.py — Benchmark evaluation test suite for Semantic Cache hit rate and latency.
"""

import time
from pathlib import Path
from unittest.mock import MagicMock
import pytest

from src.cache import SemanticCache
from src.embedder import Embedder
from src.llm import LLMClient
from src.pipeline import CachePipeline
from src.request_logger import RequestLogger


# Test dataset of query clusters (Base Query + Paraphrased Variants)
EVALUATION_DATASET = [
    {
        "base": "How do I reset my user password?",
        "paraphrases": [
            "How can I reset my password?",
            "How do I reset my account password?",
            "What are the steps to reset my password?",
        ],
        "unrelated": "What is the capital of Japan?",
    },
    {
        "base": "What is the difference between supervised and unsupervised learning?",
        "paraphrases": [
            "How does supervised learning differ from unsupervised learning?",
            "Compare supervised vs unsupervised learning.",
        ],
        "unrelated": "How do I bake a chocolate cake?",
    },
    {
        "base": "How do I calculate cosine similarity in Python?",
        "paraphrases": [
            "How to compute cosine similarity using Python?",
            "Calculate cosine similarity between two vectors in Python.",
        ],
        "unrelated": "What is the speed of light in a vacuum?",
    },
]


@pytest.fixture(scope="module")
def embedder():
    """Shared Embedder instance."""
    return Embedder()


@pytest.fixture
def temp_env(tmp_path: Path):
    """Fixture providing isolated temporary paths."""
    db_path = tmp_path / "eval_logs.db"
    index_path = tmp_path / "eval_cache.faiss"
    metadata_path = tmp_path / "eval_metadata.json"
    return db_path, index_path, metadata_path


def test_benchmark_hit_rate_and_latency(embedder: Embedder, temp_env):
    """
    Benchmark test to verify:
    1. Hit rate on paraphrased queries is >= 80%.
    2. Cached query latency is significantly lower than LLM generation latency.
    3. Unrelated queries result in cache misses.
    """
    db_path, index_path, metadata_path = temp_env

    # Mock LLM client with simulated ~50ms API delay
    mock_llm = MagicMock(spec=LLMClient)

    def simulated_llm_generate(query, system_prompt=None):
        time.sleep(0.05)  # Simulate 50ms LLM network call
        return f"Response to: '{query}'"

    mock_llm.generate.side_effect = simulated_llm_generate

    cache = SemanticCache(
        embedder=embedder,
        threshold=0.85,
        index_path=index_path,
        metadata_path=metadata_path,
        auto_load=False,
    )
    logger = RequestLogger(db_path=db_path)
    pipeline = CachePipeline(cache=cache, llm_client=mock_llm, request_logger=logger)

    total_paraphrased_queries = 0
    paraphrase_hits = 0

    hit_latencies = []
    miss_latencies = []

    # Step 1: Prime cache with base queries
    for cluster in EVALUATION_DATASET:
        base_query = cluster["base"]
        res = pipeline.process_query(base_query)
        assert res["is_hit"] is False
        miss_latencies.append(res["latency_ms"])

    # Step 2: Query paraphrased variants (expecting HITs)
    for cluster in EVALUATION_DATASET:
        for para in cluster["paraphrases"]:
            total_paraphrased_queries += 1
            res = pipeline.process_query(para)
            if res["is_hit"]:
                paraphrase_hits += 1
                hit_latencies.append(res["latency_ms"])
            else:
                miss_latencies.append(res["latency_ms"])

    # Step 3: Query unrelated queries (expecting MISSes)
    for cluster in EVALUATION_DATASET:
        unrelated = cluster["unrelated"]
        res = pipeline.process_query(unrelated)
        assert res["is_hit"] is False
        miss_latencies.append(res["latency_ms"])

    # Calculate metrics
    hit_rate = (paraphrase_hits / total_paraphrased_queries) * 100.0
    avg_hit_latency = sum(hit_latencies) / len(hit_latencies) if hit_latencies else 0.0
    avg_miss_latency = sum(miss_latencies) / len(miss_latencies) if miss_latencies else 0.0

    print(f"\n--- Evaluation Results ---")
    print(f"Paraphrase Hit Rate: {hit_rate:.1f}% ({paraphrase_hits}/{total_paraphrased_queries})")
    print(f"Avg Hit Latency: {avg_hit_latency:.2f} ms")
    print(f"Avg Miss Latency: {avg_miss_latency:.2f} ms")

    # Assertions
    assert hit_rate >= 80.0, f"Hit rate of {hit_rate:.1f}% is below target 80.0%"
    assert avg_hit_latency < avg_miss_latency, "Cached response latency should be lower than LLM latency"
