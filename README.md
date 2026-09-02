# Semantic Caching for Large Language Model Applications

[![CI Pipeline](https://github.com/itz-Maheshkumar/Semantic_Cache_Application/actions/workflows/ci.yml/badge.svg)](https://github.com/itz-Maheshkumar/Semantic_Cache_Application/actions/workflows/ci.yml)
[![Python Version](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A semantic caching layer for Large Language Model (LLM) applications that identifies semantically similar user queries using dense vector embeddings and reuses previously generated responses. Reduces query response latency, minimizes API costs, and eliminates redundant LLM calls. Includes a real-time Streamlit analytics dashboard for performance monitoring.

**Project by:** Maheshkumar V (25MCM022) — II MSc Computer Science  
**Project Guide:** Mrs. T. Kousiga, Assistant Professor  
**Department:** Computer Science, PSG College of Arts & Science  

---

## 🎯 Problem Statement

LLM applications frequently receive repeated or semantically similar queries (e.g. *"How do I reset my password?"* vs. *"How can I change my forgotten password?"*). Standard exact-match string caching fails on paraphrased queries, forcing redundant calls to expensive LLM APIs. This increases response latency and inflates operational costs.

## 💡 System Architecture & Approach

```
                  ┌───────────────────────┐
                  │   Incoming Query      │
                  └───────────┬───────────┘
                              │
                              ▼
                  ┌───────────────────────┐
                  │   Embedding Engine    │ (Sentence Transformers)
                  └───────────┬───────────┘
                              │ Vector (384-d)
                              ▼
                  ┌───────────────────────┐
                  │    Semantic Cache     │ (FAISS Vector Index)
                  └───────────┬───────────┘
                              │
               ┌──────────────┴──────────────┐
  Cosine Sim >= 0.85?                      Cosine Sim < 0.85?
               │                                     │
               ▼ (Cache HIT)                         ▼ (Cache MISS)
   ┌───────────────────────┐             ┌───────────────────────┐
   │ Return Cached Response│             │   Call OpenAI API     │
   │      (< 10ms)         │             │    (~1000-2000ms)     │
   └───────────┬───────────┘             └───────────┬───────────┘
               │                                     │
               │                                     ▼
               │                         ┌───────────────────────┐
               │                         │ Store in FAISS Cache  │
               │                         └───────────┬───────────┘
               │                                     │
               └──────────────┬──────────────────────┘
                              │
                              ▼
                  ┌───────────────────────┐
                  │ Request Logger (SQLite)│ -> Streamlit Dashboard
                  └───────────────────────┘
```

1. **Embedding Generation**: Encodes user queries into dense 384-dimensional unit L2 vectors using `all-MiniLM-L6-v2`.
2. **FAISS Vector Search**: Searches a FAISS `IndexFlatIP` index for nearest neighbor vectors.
3. **Threshold Check**: If cosine similarity $\ge 0.85$, returns the cached response instantly (**Cache HIT**).
4. **LLM Invocation**: If similarity $< 0.85$, queries OpenAI API (`gpt-4o-mini`), stores the new pair in FAISS, and returns the response (**Cache MISS**).
5. **SQLite Logging & Metrics**: Logs every request (query, response, similarity score, latency, hit/miss) to SQLite.
6. **Analytics Dashboard**: Streamlit dashboard visualizes hit rates, latency comparisons, cost savings, and request logs.

---

## 🔀 Hybrid Search (Vector + BM25)

Pure vector similarity can miss queries that share an exact keyword, product code, or acronym with a cached query but happen to embed differently (e.g. a short query dominated by a rare identifier). Hybrid search addresses this by blending FAISS cosine similarity with **BM25** keyword search over the same cached queries, so lexical overlap can rescue a match that vector similarity alone would score too low.

```
                     ┌───────────────────────┐
                     │     Incoming Query    │
                     └───────────┬───────────┘
                     ┌───────────┴───────────┐
                     ▼                       ▼
         ┌───────────────────────┐ ┌───────────────────────┐
         │   FAISS Vector Search │ │    BM25 Keyword Search│
         │   (semantic)          │ │   (lexical, rank_bm25)│
         └───────────┬───────────┘ └───────────┬───────────┘
                     │  top-K candidates        │  top-K candidates
                     └───────────┬───────────────┘
                                 ▼
                 ┌─────────────────────────────────┐
                 │  Score Fusion (src/hybrid_search)│
                 │  hybrid = w_v·vector + w_b·bm25  │
                 └─────────────────┬─────────────────┘
                                   ▼
                   Best candidate ≥ HYBRID_SIMILARITY_THRESHOLD?
                          │                        │
                          ▼ (Cache HIT)             ▼ (Cache MISS)
              Return cached response         Call OpenAI API, cache the pair
```

1. **Independent retrieval**: each lookup queries FAISS (semantic) and a `BM25Okapi` index (lexical, `src/bm25_index.py`) over the same cached queries, each returning its own top-K candidates.
2. **Score fusion** (`src/hybrid_search.py`): BM25's unbounded raw scores are min-max normalized to 0–1 across the candidate pool, then combined as `hybrid_score = VECTOR_WEIGHT * cosine_similarity + BM25_WEIGHT * bm25_normalized`. A candidate found by only one retriever isn't dropped — it scores 0 on the signal that missed it and can still win on the other.
3. **Threshold check**: the top-ranked candidate is accepted as a **Cache HIT** if `hybrid_score >= HYBRID_SIMILARITY_THRESHOLD`.

Hybrid search is **off by default** so existing vector-only behavior (and `SIMILARITY_THRESHOLD`) is unaffected unless explicitly enabled.

| Setting | Default | Description |
|---|---|---|
| `ENABLE_HYBRID_SEARCH` | `false` | Turn hybrid retrieval on. |
| `VECTOR_WEIGHT` / `BM25_WEIGHT` | `0.6` / `0.4` | Blend weights for each signal. |
| `HYBRID_TOP_K` | `10` | Candidates each retriever contributes before ranking. |
| `HYBRID_SIMILARITY_THRESHOLD` | `0.55` | Minimum blended score to accept a hit. |

Set these in `.env` (see `.env.example`), or pass them directly to `SemanticCache(hybrid_enabled=True, ...)`. The Streamlit dashboard also has a **🔀 Hybrid Search** toggle in the sidebar, including live weight/threshold sliders and a per-query breakdown (vector score vs. BM25 score) in the Live Query Playground. Unit tests for the feature live in `tests/test_hybrid_search.py`.

---

## 🧹 Cache Eviction (TTL + LRU)

A semantic cache that only ever grows eventually holds stale answers (the underlying facts changed) and wastes memory/disk on entries nobody queries anymore. Two independent, optional pruning policies address that:

- **TTL (Time-To-Live)**: entries older than a configured age (from `cached_at`, i.e. absolute expiry from insertion — not from last use) are evicted.
- **LRU (Least-Recently-Used)**: once the cache holds more than `CACHE_MAX_SIZE` entries, the least-recently-*accessed* ones are evicted until it fits — a cache **HIT** on an entry marks it as just-used (`last_accessed_at`), so a frequently-reused older entry survives longer than a stale one that was only ever queried once.

Both are **off by default** (unbounded growth, exactly the original behavior) and run automatically on every `put()` (a cache miss) and on `load()` — never on `get()`, so cache-hit latency stays fast. Eviction rebuilds the FAISS index directly from its own stored vectors (`IndexFlat.reconstruct_n`) rather than re-embedding surviving queries, so pruning never needs to call the embedding model.

| Setting | Default | Description |
|---|---|---|
| `CACHE_TTL_SECONDS` | `0` (disabled) | Entries older than this many seconds are pruned. |
| `CACHE_MAX_SIZE` | `0` (disabled) | Cache is capped at this many entries; oldest-by-last-use evicted past it. |

Set these in `.env`, or pass `ttl_seconds=`/`max_size=` directly to `SemanticCache(...)`. `cache.prune_expired()` and `cache.enforce_capacity()` are also public methods you can call manually — the Streamlit dashboard's sidebar has **⏳ TTL Expiry** / **📌 LRU Capacity Cap** toggles and a **🧹 Prune Now** button that call them on demand. Unit tests live alongside the rest of the cache engine's tests in `tests/test_cache.py`.

---

## 🛠️ Tech Stack

- **Python 3.12** — Core application development
- **Sentence Transformers** — Semantic vector embedding generation (`all-MiniLM-L6-v2`)
- **FAISS (CPU)** — High-performance vector similarity search
- **rank_bm25** — BM25 keyword search, blended with vector similarity for hybrid retrieval
- **SQLite** — Persistent request logging and metric storage
- **Streamlit** — Real-time analytics dashboard & query playground
- **OpenAI API** — LLM response generation on cache miss
- **pytest & flake8** — Automated test suite and code quality linting

---

## 📂 Project Structure

```text
semantic-cache-project/
├── .github/
│   └── workflows/
│       └── ci.yml               # GitHub Actions CI pipeline
├── src/                         # Application source code
│   ├── __init__.py              # Package init
│   ├── config.py                # Configuration and environment loader
│   ├── logger.py                # Structured logging utility
│   ├── embedder.py              # Embedding Engine (Sentence Transformers)
│   ├── cache.py                 # Semantic Cache Engine (FAISS + Storage)
│   ├── bm25_index.py            # BM25 keyword search index (hybrid search)
│   ├── hybrid_search.py         # Vector + BM25 score fusion (hybrid search)
│   ├── llm.py                   # OpenAI API client wrapper with retries
│   ├── models.py                # Data models (CacheResult, RequestLog)
│   ├── request_logger.py        # SQLite logging & KPI aggregator
│   └── pipeline.py              # Main Cache Pipeline orchestrator
├── tests/                       # Automated unit & benchmark test suite
│   ├── test_embedder.py
│   ├── test_cache.py
│   ├── test_hybrid_search.py
│   ├── test_pipeline.py
│   └── test_evaluation.py
├── scripts/
│   └── evaluate.py              # Standalone CLI evaluation & benchmark script
├── app.py                       # Streamlit Analytics Dashboard
├── requirements.txt             # Python dependencies
├── .env.example                 # Environment variables template
├── README.md                    # Project documentation
└── LICENSE                      # MIT License
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.12+
- `pip` package manager

### Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone https://github.com/itz-Maheshkumar/Semantic_Cache_Application.git
   cd semantic-cache-project
   ```

2. **Create and activate a virtual environment:**
   ```powershell
   py -3.12 -m venv .venv
   .venv\Scripts\Activate.ps1
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Environment Variables:**
   Copy `.env.example` to `.env` and add your OpenAI API key (optional for mock testing):
   ```bash
   cp .env.example .env
   ```

5. **Run the Streamlit Dashboard:**
   ```bash
   streamlit run app.py
   ```

---

## 📊 Evaluation & Benchmarking

Run the automated evaluation benchmark script to measure cache hit rates, average latency, and estimated cost savings:

```bash
python scripts/evaluate.py
```

### Benchmark Results Overview

| Metric | Without Cache (Baseline) | With Semantic Cache | Improvement |
|---|---|---|---|
| **Average Query Latency** | ~1200 ms | **< 10 ms** (on hit) | **~120x Faster** |
| **Paraphrased Query Hit Rate** | 0% | **> 85%** | **+85% Efficiency** |
| **API Cost per 1k Hits** | $0.15 | **$0.00** | **100% Cost Reduction** |

### Running Unit Tests

Run the complete test suite using `pytest`:

```bash
pytest tests/ --verbose
```

---

## 📄 License

This project is licensed under the **MIT License**.

| Permissions | Conditions | Limitations |
|---|---|---|
| ✅ Commercial use | 📋 License and copyright notice must be included | ❌ No liability |
| ✅ Modification | | ❌ No warranty |
| ✅ Distribution | | |
| ✅ Private use | | |

Full license text is available in the [LICENSE](LICENSE) file.

Copyright (c) 2026 Maheshkumar V
