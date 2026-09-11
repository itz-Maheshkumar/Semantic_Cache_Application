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

## 🖼️🎤 Multi-Modal Caching (Image + Audio)

Caching isn't limited to typed text queries. Images get their own CLIP-based vector cache; audio is transcribed to text and reuses the existing text cache unchanged — two spoken queries that say the same thing become the same cache lookup problem this project already solves.

```
  IMAGE query                                    AUDIO query
       │                                              │
       ▼                                              ▼
┌─────────────────┐                        ┌───────────────────────┐
│  CLIP Embedder   │                        │  Whisper Transcriber  │
│(image_embedder.py)│                       │ (audio_transcriber.py)│
└────────┬─────────┘                        └───────────┬───────────┘
         │ Vector (512-d)                                │ Transcript (text)
         ▼                                                ▼
┌─────────────────┐                        ┌───────────────────────┐
│  IMAGE Cache     │                        │     TEXT Cache        │
│ (separate FAISS  │                        │  (same cache/index as │
│  index, vector-  │                        │   ordinary typed      │
│  only, no BM25)  │                        │   queries — see above)│
└────────┬─────────┘                        └───────────┬───────────┘
         │ MISS                                          │ MISS
         ▼                                                ▼
┌─────────────────┐                        ┌───────────────────────┐
│ generate_vision()│                        │      generate()       │
│  (vision LLM)    │                        │       (text LLM)      │
└──────────────────┘                        └────────────────────────┘
```

- **Image queries** (`SemanticCache(modality=Modality.IMAGE)`, `src/image_embedder.py`) are embedded with a CLIP model (`clip-ViT-B-32` by default) into their own 512-dimensional vector space and FAISS index — kept entirely separate from the 384-dimensional text index, since the two embedding spaces aren't comparable. Hybrid/BM25 search is keyword search over text and has no meaning for an image, so it's **always forced off** for an image cache, regardless of `ENABLE_HYBRID_SEARCH`. TTL/LRU eviction, save/load, and everything else in `SemanticCache` work identically across modalities, since that machinery only ever deals in vectors and timestamps, not the original query type.
- Because a raw image (bytes/`PIL.Image`) can't be stored in metadata's string-typed `query` field or displayed/logged directly, each image entry gets a short **label** — either one you supply (`query_label=`, e.g. an uploaded filename) or an auto-generated one derived from the image's content hash (`image:<12-hex-digest>`), so identical images always get the same label.
- On a MISS, `CachePipeline.process_image_query()` calls `LLMClient.generate_vision()` (`src/llm.py`) — a vision-capable OpenAI chat completion (`gpt-4o-mini` supports image input out of the box). A local file, raw bytes, or `PIL.Image` is base64-encoded into a `data:` URL; a plain `http(s)` URL is passed straight through.
- **Audio queries** are never embedded or cached directly. `CachePipeline.process_audio_query()` first transcribes the audio via `AudioTranscriber.transcribe()` (`src/audio_transcriber.py`, OpenAI Whisper), then runs the resulting transcript through the exact same text pipeline as a typed query — cache lookup, LLM fallback, caching, logging, all unchanged. "Audio" is recorded only as a `modality="audio"` label on the resulting request log, for dashboard/analytics purposes; the underlying cache entry is indistinguishable from one created by a typed query.
- Every request log (`RequestLog`, `src/models.py`) carries a `modality` field (`"text"` / `"image"` / `"audio"`) so dashboard/analytics views can tell the three apart. The SQLite table migrates itself (`ALTER TABLE ... ADD COLUMN`) the first time `RequestLogger` opens an older database file that predates this column — existing rows come back with `modality=None`, nothing is lost.

| Setting | Default | Description |
|---|---|---|
| `IMAGE_EMBEDDING_MODEL` | `clip-ViT-B-32` | CLIP model used to embed image queries. |
| `IMAGE_FAISS_INDEX_FILE` / `IMAGE_CACHE_METADATA_FILE` | `image_cache.faiss` / `image_cache_metadata.json` | Persisted files for the image cache (stored in `data/`, separate from the text cache's files). |
| `AUDIO_TRANSCRIPTION_MODEL` | `whisper-1` | OpenAI Whisper model used to transcribe audio queries. |

Both features are entirely additive: a `CachePipeline` built without `image_cache=`/`audio_transcriber=` (the default) behaves exactly as it did before this feature existed — `process_image_query()`/`process_audio_query()` simply raise a clear `RuntimeError` if called without the relevant dependency configured. The Streamlit dashboard's **🖼️🎤 Multi-Modal Playground** tab has image/audio uploaders (with a "mock" mode so it works without a real `OPENAI_API_KEY`) and lazily loads the CLIP model only on first use, so it never slows down or breaks the rest of the dashboard if the model can't be downloaded. Unit tests live alongside the rest of the cache/pipeline test suites in `tests/test_cache.py` and `tests/test_pipeline.py`.

---

## 🔌 LLM Provider: Mock vs. Live

Every OpenAI call in the app — text generation, vision, and Whisper transcription — can run in two modes:

- **Mock** — a fixed canned response, no network call, no API key needed. Lets you exercise the cache/eviction/hybrid-search mechanics for free.
- **Live** — a real `gpt-4o-mini` call (and real Whisper transcription for audio), so cache-miss responses reflect the actual model instead of a fixed string.

The Streamlit dashboard now picks a sane default automatically instead of always defaulting to mock: each **"Use Mock ..."** checkbox (Live Query Playground, and the Multi-Modal Playground's image/audio tabs) defaults to **checked** when no real `OPENAI_API_KEY` is configured, and **unchecked** the moment a real key is set in `.env` — nothing to remember to toggle between runs. The sidebar's **System Info** panel also shows the current mode at a glance:

```
LLM Provider: 🟢 Live — gpt-4o-mini
LLM Provider: 🟡 Not configured (Mock only)
```

The Live Query Playground also now catches a failed LLM call (e.g. an invalid or expired key) and shows a clean `st.error(...)` message instead of crashing the page.

> **Note:** mock vs. live only changes what a cache **miss** returns. The semantic-cache hit/miss decision itself is driven entirely by query-embedding similarity (`SIMILARITY_THRESHOLD` / `HYBRID_SIMILARITY_THRESHOLD`) and behaves identically either way — switching to a real key makes miss responses genuine, it doesn't change the hit rate.

---

## 🛠️ Tech Stack

- **Python 3.12** — Core application development
- **Sentence Transformers** — Semantic vector embedding generation (`all-MiniLM-L6-v2` for text, `clip-ViT-B-32` for images)
- **FAISS (CPU)** — High-performance vector similarity search
- **rank_bm25** — BM25 keyword search, blended with vector similarity for hybrid retrieval
- **Pillow** — Image loading/decoding for multi-modal (image) caching
- **SQLite** — Persistent request logging and metric storage
- **Streamlit** — Real-time analytics dashboard & query playground
- **OpenAI API** — LLM response generation on cache miss, plus vision (image) and Whisper (audio transcription) for multi-modal queries
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
│   ├── embedder.py              # Embedding Engine (Sentence Transformers, text)
│   ├── image_embedder.py        # Image Embedding Engine (CLIP, multi-modal caching)
│   ├── audio_transcriber.py     # OpenAI Whisper client (multi-modal caching)
│   ├── modality.py              # Modality enum (TEXT / IMAGE) shared across the cache
│   ├── cache.py                 # Semantic Cache Engine (FAISS + Storage)
│   ├── bm25_index.py            # BM25 keyword search index (hybrid search)
│   ├── hybrid_search.py         # Vector + BM25 score fusion (hybrid search)
│   ├── llm.py                   # OpenAI API client wrapper with retries (text + vision)
│   ├── models.py                # Data models (CacheResult, RequestLog)
│   ├── request_logger.py        # SQLite logging & KPI aggregator
│   └── pipeline.py              # Main Cache Pipeline orchestrator
├── tests/                       # Automated unit & benchmark test suite
│   ├── test_embedder.py
│   ├── test_cache.py            # incl. TTL/LRU eviction + multi-modal (IMAGE) tests
│   ├── test_hybrid_search.py
│   ├── test_pipeline.py         # incl. multi-modal (image/audio) pipeline + LLM vision tests
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
   Copy `.env.example` to `.env`:
   ```bash
   cp .env.example .env
   ```
   Add a real `OPENAI_API_KEY` to enable **live** responses — the dashboard's mock checkboxes automatically default to unchecked once a valid key is detected. Leave the placeholder value in place to keep running fully in **mock** mode (no key required).

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
