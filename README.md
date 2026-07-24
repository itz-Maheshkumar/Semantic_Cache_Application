# Semantic Caching for Large Language Model Applications

A semantic caching layer for LLM applications that identifies semantically similar user queries and reuses previously generated responses, reducing latency, API cost, and redundant LLM calls. Includes a Streamlit analytics dashboard for monitoring cache performance.

## Problem

LLM applications frequently receive repeated or semantically similar queries. Each query triggers a new LLM call, increasing latency and operational cost, and existing systems lack an intelligent mechanism to reuse previous responses.

## Approach

1. Convert incoming user queries into embeddings using Sentence Transformers.
2. Search a FAISS vector index for the closest previously-seen query.
3. If the similarity score passes a defined threshold, return the cached response (**cache hit**).
4. Otherwise, call the LLM, store the new query/response pair in the index, and return the fresh response (**cache miss**).
5. Log every request (hit/miss, similarity score, response time) for analytics.
6. Visualize hit rate, latency, and cost savings on a Streamlit dashboard.

## Tech Stack

- **Sentence Transformers** — semantic embedding generation
- **FAISS** — vector similarity search and cache retrieval
- **Python** — core implementation
- **Streamlit** — analytics dashboard
- **OpenAI API** (or alternative LLM provider) — response generation on cache miss

## Project Structure

```
semantic-cache-project/
├── .github/
│   └── workflows/
│       └── ci.yml          # GitHub Actions CI pipeline
├── .venv/                  # Local virtual environment (not committed)
├── src/                    # Core application source code
├── tests/                  # Unit tests
├── requirements.txt        # Python dependencies
├── README.md
└── .gitignore
```

## Getting Started

### Prerequisites

- Python 3.12
- pip

### Setup

1. Clone the repository:
   ```
   git clone <your-repo-url>
   cd semantic-cache-project
   ```

2. Create and activate a virtual environment:
   ```
   py -3.12 -m venv .venv
   .venv\Scripts\Activate.ps1
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Run the Streamlit dashboard:
   ```
   streamlit run app.py
   ```

## Evaluation

The system is evaluated against a no-cache baseline using a test set containing paraphrased query variants, measuring:

- Cache hit rate
- Average response latency (cached vs. LLM-served)
- Estimated reduction in LLM API calls / cost

## Expected Outcomes

- Reduced response latency
- Reduced LLM API calls and operational costs
- Improved scalability of LLM-powered applications
- Better monitoring through performance analytics

## License

This project is licensed under the **MIT License**.

You are free to use, copy, modify, merge, publish, distribute, sublicense, and sell copies of this software, provided that the original copyright notice and permission notice are included in all copies or substantial portions of the software. The software is provided "as is", without warranty of any kind.

| Permissions | Conditions | Limitations |
|---|---|---|
| ✅ Commercial use | 📋 License and copyright notice must be included | ❌ No liability |
| ✅ Modification | | ❌ No warranty |
| ✅ Distribution | | |
| ✅ Private use | | |

MIT License

Copyright (c) 2026 Maheshkumar V

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
