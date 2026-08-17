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

## 🛠️ Tech Stack

- **Python 3.12** — Core application development
- **Sentence Transformers** — Semantic vector embedding generation (`all-MiniLM-L6-v2`)
- **FAISS (CPU)** — High-performance vector similarity search
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
│   ├── llm.py                   # OpenAI API client wrapper with retries
│   ├── models.py                # Data models (CacheResult, RequestLog)
│   ├── request_logger.py        # SQLite logging & KPI aggregator
│   └── pipeline.py              # Main Cache Pipeline orchestrator
├── tests/                       # Automated unit & benchmark test suite
│   ├── test_embedder.py
│   ├── test_cache.py
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
