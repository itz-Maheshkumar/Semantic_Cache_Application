# Roadmap & Task Tracker

Planned improvements for the Semantic Cache project, ordered roughly by effort vs. payoff. Update the **Status** column as work progresses, and add a line to the [Progress Log](#progress-log) when a task changes state.

**Last updated:** 2026-11-15

## Status legend

| Status | Meaning |
|---|---|
| ⬜ Not started | Planned, no work done yet |
| 🟡 In progress | Actively being worked on |
| ✅ Done | Merged to `main` |
| ⏸️ On hold | Deferred or blocked (say why in Notes) |
| 💬 Discussion only | Not planned to build; know the tradeoff for viva/interviews |

## Summary

| Group | Tasks | Done | In progress | Not started |
|---|---|---|---|---|
| 1. Quick wins | 4 | 0 | 0 | 4 |
| 2. Engineering additions | 4 | 0 | 0 | 4 |
| 3. Stretch / discussion topics | 3 | 0 | 0 | 3 |

---

## 1. Quick wins

A few hours each, high resume/viva payoff.

| ID | Task | Status | Notes |
|---|---|---|---|
| QW-1 | Surface the near-miss score on a cache MISS | ⬜ Not started | `src/pipeline.py` hardcodes `similarity_score: 0.0` on a miss (text and image paths) |
| QW-2 | Publish `scripts/evaluate.py` benchmark results in the README | ⬜ Not started | README table currently shows approximate figures, not real script output |
| QW-3 | Add a "Limitations / Future Work" section to the README | ⬜ Not started | |
| QW-4 | Add a demo GIF/screenshot block at the top of the README | ⬜ Not started | Screenshots already exist in `docs/images/` but aren't referenced |

### QW-1 — Surface the near-miss score on a cache MISS

`process_query()` currently hardcodes `similarity_score: 0.0` on a miss, hiding how close the query actually came to a cached entry. Fixing it turns the dashboard into a real debugging tool and gives a concrete "observability" talking point.

- [ ] Return the best candidate's score from the cache lookup even when it's below threshold
- [ ] Pass it through `process_query()` and `process_image_query()` instead of `0.0`
- [ ] Log it in `RequestLog` and show it in the Live Query Playground on a miss
- [ ] Add tests covering the miss-with-score case

### QW-2 — Publish benchmark results in the README

The benchmarking script with paraphrase datasets already exists; its output just isn't shown anywhere. A table of "X% hit rate, Y ms average latency saved" is exactly the kind of quantified line recruiters scan for.

- [ ] Run `python scripts/evaluate.py` and capture the output
- [ ] Replace the approximate figures in the README's "Benchmark Results Overview" with the measured ones
- [ ] Note the dataset size, threshold and provider used for the run

### QW-3 — Add a "Limitations / Future Work" section

Academic reviewers and interviewers both look for whether you know your own project's weak points. The README has no such section yet.

- [ ] Write the section (the items in group 3 below are the starting material)
- [ ] Link it to this roadmap

### QW-4 — Demo GIF/screenshot block

Cheap, and makes the repo land in 5 seconds instead of requiring someone to run it.

- [ ] Pick 2–3 screenshots from `docs/images/` (or record a short GIF)
- [ ] Rename them to descriptive, space-free file names
- [ ] Embed them near the top of the README

---

## 2. Engineering additions

Resume-grade depth.

| ID | Task | Status | Notes |
|---|---|---|---|
| EN-1 | Fill the test coverage gaps | ⬜ Not started | No tests for `llm.py`, `request_logger.py`, `image_embedder.py`, `audio_transcriber.py`, `app.py` |
| EN-2 | Wrap the pipeline as a FastAPI service | ⬜ Not started | `POST /cache/query` in front of `CachePipeline` |
| EN-3 | Dockerize | ⬜ Not started | Dockerfile + volume for `data/` |
| EN-4 | Add a `namespace` field to cache entries | ⬜ Not started | Per-app / per-tenant isolation |

### EN-1 — Fill the test coverage gaps

`tests/` covers cache, embedder, evaluation, hybrid search and pipeline, but several modules have no test file of their own. `llm.py` is the easiest to cover and directly demonstrates the multi-provider work.

- [ ] `tests/test_llm.py` — mock `openai.OpenAI`; assert the right `base_url`/model/key is picked per `LLM_PROVIDER`, placeholder keys raise, retries back off
- [ ] `tests/test_request_logger.py` — logging, KPI aggregation, `modality` column migration
- [ ] `tests/test_image_embedder.py`
- [ ] `tests/test_audio_transcriber.py`
- [ ] Smoke test for `app.py` (e.g. Streamlit's `AppTest`)

### EN-2 — FastAPI layer

Right now the project only works as a Streamlit demo. A thin API in front of the same `CachePipeline` turns it into caching middleware a real app could call — a much stronger story than "I built a dashboard."

- [ ] `POST /cache/query` returning response, hit/miss, score and latency
- [ ] Health and stats endpoints
- [ ] Tests with FastAPI's `TestClient`
- [ ] Document usage in the README

### EN-3 — Dockerize

Makes the project trivially deployable (Render, HF Spaces, Railway) and answers the near-guaranteed interview question "how would you deploy this?"

- [ ] `Dockerfile` and `.dockerignore`
- [ ] Volume for `data/` so the cache survives restarts
- [ ] Run instructions in the README
- [ ] Deploy to one free host and link it

### EN-4 — Namespaces

All entries are currently flat/global. Per-app or per-tenant namespaces are a small code change that signals this was designed as a shared service, not a single-user toy.

- [ ] Add `namespace` to cache entry metadata (default namespace for existing entries)
- [ ] Filter lookups by namespace in `get()` / `put()`
- [ ] Expose it in the pipeline (and the API, once EN-2 exists)
- [ ] Tests for isolation between namespaces

---

## 3. Stretch / discussion topics

Worth being able to explain in an interview even if never built.

| ID | Topic | Status | Notes |
|---|---|---|---|
| ST-1 | Approximate FAISS indexes for scale | 💬 Discussion only | `IndexIVFFlat` / HNSW vs. current `IndexFlatIP` |
| ST-2 | Vector DB instead of file-based persistence | 💬 Discussion only | Qdrant, pgvector, Redis |
| ST-3 | Guardrails | ⬜ Not started | Query length cap, sanitizing before logging |

### ST-1 — Scaling the index

`IndexFlatIP` is brute-force O(n) per lookup. That's fine at the current scale, but the answer to "how would this scale to millions of entries?" is approximate indexes such as `IndexIVFFlat` or HNSW, which trade a little accuracy for a large speedup. Flat was chosen for simplicity, exact results and small-to-medium cache sizes.

### ST-2 — Persistence

The cache persists as a FAISS file plus a JSON sidecar. A real vector database (Qdrant, pgvector, Redis) would add concurrent access, filtering and durability, at the cost of an extra service to run. A natural next step rather than a current need.

### ST-3 — Guardrails

There are no input size limits and no prompt-injection awareness right now. Even capping query length and sanitizing text before logging is worth doing, or at least listing under "Future Work".

- [ ] Cap query length before embedding / LLM call
- [ ] Sanitize query text before logging
- [ ] Mention prompt-injection handling in "Limitations / Future Work"

---

## Progress log

| Date | Change |
|---|---|
| 2026-11-15 | Roadmap created. Baseline: Hugging Face and Groq LLM providers shipped (`87c3d6b`). |
