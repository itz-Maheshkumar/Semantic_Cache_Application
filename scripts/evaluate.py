"""
evaluate.py — Standalone CLI evaluation & benchmarking script for Semantic Cache.

Usage:
    python scripts/evaluate.py [--threshold 0.85] [--mock]
"""

import argparse
import time
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np

# Ensure parent directory is in python path
import sys
from pathlib import Path

# Ensure UTF-8 output encoding for Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.cache import SemanticCache
from src.config import SIMILARITY_THRESHOLD
from src.embedder import Embedder
from src.llm import LLMClient
from src.pipeline import CachePipeline
from src.request_logger import RequestLogger

# Comprehensive evaluation dataset
BENCHMARK_DATASET = [
    {
        "category": "Account & Auth",
        "base": "How do I reset my account password?",
        "paraphrases": [
            "How can I reset my password?",
            "What is the procedure to reset my account password?",
            "How do I change my forgotten password?",
        ],
    },
    {
        "category": "Machine Learning",
        "base": "What is the difference between classification and regression?",
        "paraphrases": [
            "How does classification differ from regression in ML?",
            "Explain classification vs regression tasks.",
            "Compare classification and regression algorithms.",
        ],
    },
    {
        "category": "Vector Databases",
        "base": "How does FAISS vector indexing work?",
        "paraphrases": [
            "Explain how FAISS handles vector similarity indexing.",
            "How does FAISS store and search high-dimensional vectors?",
            "What is the underlying mechanism of FAISS vector search?",
        ],
    },
    {
        "category": "Python Programming",
        "base": "How do I read a JSON file in Python?",
        "paraphrases": [
            "What is the code to parse a JSON file using Python?",
            "How to open and load JSON data in Python?",
            "Python script to read JSON file into a dictionary.",
        ],
    },
]


def run_benchmark(threshold: float = 0.85, use_mock: bool = True):
    """Run full benchmark suite and print performance summary."""
    print("=" * 70)
    print("⚡ SEMANTIC CACHE BENCHMARK EVALUATION")
    print("=" * 70)
    print(f"Similarity Threshold: {threshold}")
    print(f"LLM Execution Mode:  {'Mock Mode (Simulated 100ms API latency)' if use_mock else 'Live OpenAI API'}")
    print("-" * 70)

    # Isolated temporary benchmark cache & log DB
    eval_dir = Path(__file__).resolve().parent.parent / "data" / "benchmark_run"
    eval_dir.mkdir(parents=True, exist_ok=True)

    index_path = eval_dir / "bench_cache.faiss"
    metadata_path = eval_dir / "bench_metadata.json"
    db_path = eval_dir / "bench_logs.db"

    embedder = Embedder()
    cache = SemanticCache(
        embedder=embedder,
        threshold=threshold,
        index_path=index_path,
        metadata_path=metadata_path,
        auto_load=False,
    )
    cache.clear(delete_files=True)

    logger = RequestLogger(db_path=db_path)
    logger.clear()

    if use_mock:
        llm_client = MagicMock(spec=LLMClient)
        def mock_generate(q, system_prompt=None):
            time.sleep(0.10)  # Simulate 100ms API call
            return f"Generated response for: '{q}'"
        llm_client.generate.side_effect = mock_generate
    else:
        llm_client = LLMClient()

    pipeline = CachePipeline(cache=cache, llm_client=llm_client, request_logger=logger)

    print("\n1. Priming cache with base queries...")
    for item in BENCHMARK_DATASET:
        base_q = item["base"]
        res = pipeline.process_query(base_q)
        print(f"   [MISS] Primed: '{base_q[:45]}...' ({res['latency_ms']:.2f} ms)")

    print(f"\n2. Testing {sum(len(x['paraphrases']) for x in BENCHMARK_DATASET)} paraphrased query variants...")
    hits = 0
    total_paraphrases = 0

    for item in BENCHMARK_DATASET:
        category = item["category"]
        print(f"\n   Category: {category}")
        for para in item["paraphrases"]:
            total_paraphrases += 1
            res = pipeline.process_query(para)
            status = "HIT ✅" if res["is_hit"] else "MISS ❌"
            if res["is_hit"]:
                hits += 1
            print(f"     [{status}] Similarity: {res['similarity_score']:.4f} | Latency: {res['latency_ms']:.2f} ms")
            print(f"            Query: '{para}'")

    print("\n" + "=" * 70)
    print("📊 EVALUATION RESULTS SUMMARY")
    print("=" * 70)

    stats = logger.get_stats()

    hit_rate = (hits / total_paraphrases) * 100.0 if total_paraphrases > 0 else 0.0
    speedup = (
        (stats["avg_miss_latency_ms"] / stats["avg_hit_latency_ms"])
        if stats["avg_hit_latency_ms"] > 0
        else 0.0
    )

    print(f"Total Base Queries Primed:       {len(BENCHMARK_DATASET)}")
    print(f"Total Paraphrased Variants:      {total_paraphrased_queries if 'total_paraphrased_queries' in locals() else total_paraphrases}")
    print(f"Cache Hits:                      {hits}")
    print(f"Paraphrase Hit Rate:             {hit_rate:.1f}%")
    print(f"Average Cache Hit Latency:       {stats['avg_hit_latency_ms']:.2f} ms")
    print(f"Average Cache Miss Latency:      {stats['avg_miss_latency_ms']:.2f} ms")
    print(f"Speedup Factor (Miss vs Hit):    {speedup:.1f}x Faster")
    print(f"Estimated Tokens Saved:          {stats['estimated_tokens_saved']}")
    print(f"Estimated Cost Savings (USD):    ${stats['estimated_cost_saved_usd']:.5f}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Semantic Cache Performance.")
    parser.add_argument(
        "--threshold",
        type=float,
        default=SIMILARITY_THRESHOLD,
        help="Similarity threshold for cache hits (default: 0.85)",
    )
    parser.add_argument(
        "--live-llm",
        action="store_true",
        help="Use real OpenAI API instead of simulated mock LLM",
    )
    args = parser.parse_args()

    run_benchmark(threshold=args.threshold, use_mock=not args.live_llm)
