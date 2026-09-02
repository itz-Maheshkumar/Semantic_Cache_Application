"""
app.py — Streamlit Analytics Dashboard & Live Query Playground for Semantic Cache.

Run with:
    streamlit run app.py
"""

import time
import pandas as pd
import streamlit as st

from src.cache import SemanticCache
from src.config import (
    EMBEDDING_MODEL,
    SIMILARITY_THRESHOLD,
    ENABLE_HYBRID_SEARCH,
    VECTOR_WEIGHT,
    BM25_WEIGHT,
    HYBRID_SIMILARITY_THRESHOLD,
    CACHE_TTL_SECONDS,
    CACHE_MAX_SIZE,
)
from src.embedder import Embedder
from src.llm import LLMClient
from src.pipeline import CachePipeline
from src.request_logger import RequestLogger

# ── Page Configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Semantic Cache Analytics Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom Styling ─────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.0rem;
        color: #6c757d;
        margin-bottom: 1.5rem;
    }
    .kpi-card {
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        text-align: center;
    }
    .kpi-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #0f172a;
    }
    .kpi-label {
        font-size: 0.85rem;
        font-weight: 500;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .badge-hit {
        background-color: #dcfce7;
        color: #15803d;
        padding: 0.3rem 0.8rem;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.9rem;
        display: inline-block;
    }
    .badge-miss {
        background-color: #fee2e2;
        color: #b91c1c;
        padding: 0.3rem 0.8rem;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.9rem;
        display: inline-block;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── State Initialization ─────────────────────────────────────────────────────
@st.cache_resource
def get_embedder() -> Embedder:
    """Load and cache the sentence transformer embedder instance."""
    return Embedder(model_name=EMBEDDING_MODEL)


@st.cache_resource
def get_components(threshold: float):
    """Instantiate and cache core pipeline components."""
    embedder = get_embedder()
    cache = SemanticCache(embedder=embedder, threshold=threshold)
    llm_client = LLMClient()
    request_logger = RequestLogger()
    pipeline = CachePipeline(cache=cache, llm_client=llm_client, request_logger=request_logger)
    return cache, request_logger, pipeline


# ── Sidebar Controls ──────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/96/lightning-bolt.png", width=64)
    st.title("Settings & Control")

    st.subheader("Cache Configuration")

    hybrid_mode = st.toggle(
        "🔀 Hybrid Search (Vector + BM25)",
        value=ENABLE_HYBRID_SEARCH,
        help="Blend FAISS vector similarity with BM25 keyword scoring instead "
        "of relying on vector similarity alone.",
    )

    if hybrid_mode:
        similarity_threshold = st.slider(
            "Hybrid Similarity Threshold",
            min_value=0.10,
            max_value=0.99,
            value=float(HYBRID_SIMILARITY_THRESHOLD),
            step=0.01,
            help="Queries with blended hybrid score >= this value will trigger a Cache HIT.",
        )
        vector_weight = st.slider(
            "Vector Weight", min_value=0.0, max_value=1.0, value=float(VECTOR_WEIGHT), step=0.05,
            help="Weight given to cosine similarity in the blended score.",
        )
        bm25_weight = st.slider(
            "BM25 Weight", min_value=0.0, max_value=1.0, value=float(BM25_WEIGHT), step=0.05,
            help="Weight given to normalized BM25 keyword score in the blended score.",
        )
    else:
        similarity_threshold = st.slider(
            "Similarity Threshold (Cosine)",
            min_value=0.50,
            max_value=0.99,
            value=float(SIMILARITY_THRESHOLD),
            step=0.01,
            help="Queries with similarity score >= this value will trigger a Cache HIT.",
        )
        vector_weight, bm25_weight = VECTOR_WEIGHT, BM25_WEIGHT

    st.markdown("---")
    st.subheader("System Info")
    st.write(f"**Embedding Model:** `{EMBEDDING_MODEL}`")
    st.write(f"**Retrieval Mode:** `{'Hybrid (Vector + BM25)' if hybrid_mode else 'Vector Only'}`")

    # Retrieve instances (cached across reruns; live settings are applied below)
    cache, request_logger, pipeline = get_components(similarity_threshold)
    cache.hybrid_enabled = hybrid_mode
    if hybrid_mode:
        cache.hybrid_threshold = similarity_threshold
        cache.vector_weight = vector_weight
        cache.bm25_weight = bm25_weight
    else:
        cache.threshold = similarity_threshold

    st.write(f"**Cached Entries:** `{cache.size}`")

    st.markdown("---")
    st.subheader("Eviction Policy")

    ttl_enabled = st.toggle(
        "⏳ TTL Expiry",
        value=CACHE_TTL_SECONDS > 0,
        help="Automatically evict entries older than a fixed age.",
    )
    ttl_hours = 0.0
    if ttl_enabled:
        ttl_hours = st.slider(
            "TTL (hours)",
            min_value=0.5,
            max_value=24 * 30,  # up to 30 days
            value=max(float(CACHE_TTL_SECONDS) / 3600.0, 1.0),
            step=0.5,
            help="Entries older than this (since they were cached) are pruned automatically.",
        )

    lru_enabled = st.toggle(
        "📌 LRU Capacity Cap",
        value=CACHE_MAX_SIZE > 0,
        help="Cap the cache at a maximum number of entries, evicting the "
        "least-recently-used ones once it's full.",
    )
    max_size = 0
    if lru_enabled:
        max_size = st.number_input(
            "Max Entries", min_value=1, max_value=1_000_000,
            value=CACHE_MAX_SIZE if CACHE_MAX_SIZE > 0 else 500, step=10,
        )

    cache.ttl_seconds = int(ttl_hours * 3600) if ttl_enabled else 0
    cache.max_size = int(max_size) if lru_enabled else 0

    if st.button("🧹 Prune Now", help="Run TTL and LRU eviction immediately, without waiting for the next query."):
        expired = cache.prune_expired()
        evicted = cache.enforce_capacity()
        cache.save()
        st.toast(f"Pruned {expired} expired + {evicted} LRU-evicted entries.", icon="🧹")
        st.rerun()

    st.markdown("---")
    st.subheader("Management")

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("🗑️ Clear Cache", help="Clear all stored vectors and metadata."):
            cache.clear(delete_files=True)
            st.toast("Cache cleared successfully!", icon="✅")
            st.rerun()

    with col_btn2:
        if st.button("📊 Reset Logs", help="Truncate all request logs in SQLite."):
            request_logger.clear()
            st.toast("Logs reset successfully!", icon="✅")
            st.rerun()


# ── Dashboard Header ──────────────────────────────────────────────────────────
st.markdown('<div class="main-title">⚡ Semantic Cache Analytics</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Intelligent LLM Response Caching with FAISS Vector Search & Real-Time Performance Monitoring</div>',
    unsafe_allow_html=True,
)

# Fetch aggregate statistics
stats = request_logger.get_stats()

# ── Top KPI Bar ───────────────────────────────────────────────────────────────
kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)

with kpi1:
    st.metric(label="Total Requests", value=stats["total_requests"])

with kpi2:
    st.metric(
        label="Cache Hit Rate",
        value=f"{stats['hit_rate_pct']:.1f}%",
        delta=f"{stats['total_hits']} Hits / {stats['total_misses']} Misses",
    )

with kpi3:
    st.metric(
        label="Avg Hit Latency",
        value=f"{stats['avg_hit_latency_ms']:.1f} ms",
        delta="-Fast" if stats["avg_hit_latency_ms"] < 100 else None,
    )

with kpi4:
    st.metric(
        label="Avg Miss Latency",
        value=f"{stats['avg_miss_latency_ms']:.1f} ms",
    )

with kpi5:
    st.metric(
        label="Est. Cost Saved",
        value=f"${stats['estimated_cost_saved_usd']:.4f}",
        delta=f"{stats['estimated_tokens_saved']} tokens",
    )

st.markdown("---")

# ── Navigation Tabs ───────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🧪 Live Query Playground",
    "📊 Performance Analytics",
    "📜 Request Logs",
    "🗄️ Cache Inspector",
])

# ── TAB 1: Live Query Playground ──────────────────────────────────────────────
with tab1:
    st.subheader("Test Semantic Cache in Real Time")
    st.caption("Submit queries to see if they hit the cache or invoke the LLM.")

    with st.form("query_form"):
        user_query = st.text_area(
            "Enter your query:",
            placeholder="e.g. How do I reset my account password?",
            height=100,
        )

        mock_llm_mode = st.checkbox(
            "Use Mock LLM (Simulated response without API key)",
            value=True,
            help="Uncheck if you have set your OPENAI_API_KEY in .env.",
        )

        submit_btn = st.form_submit_button("🚀 Submit Query", type="primary")

    if submit_btn and user_query.strip():
        if mock_llm_mode:
            # Inject mock generator for local testing
            pipeline.llm_client.generate = lambda q, system_prompt=None: (
                f"This is a generated response for your query: '{q}'."
            )

        with st.spinner("Processing query through semantic pipeline..."):
            result = pipeline.process_query(user_query.strip())

        st.markdown("### Result")
        res_col1, res_col2 = st.columns([1, 2])

        with res_col1:
            if result["is_hit"]:
                st.markdown('<div class="badge-hit">✅ CACHE HIT</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="badge-miss">❌ CACHE MISS</div>', unsafe_allow_html=True)

            st.write(f"**Latency:** `{result['latency_ms']:.2f} ms`")
            score_label = "Hybrid Score" if result.get("vector_score") is not None else "Similarity Score"
            st.write(f"**{score_label}:** `{result['similarity_score']:.4f}`")
            if result.get("vector_score") is not None:
                st.caption(
                    f"Vector: `{result['vector_score']:.4f}` · BM25: `{result['bm25_score']:.4f}`"
                )
            if result["matched_query"]:
                st.write(f"**Matched Cache Query:** *\"{result['matched_query']}\"*")

        with res_col2:
            st.subheader("Response")
            st.info(result["response"])

        st.rerun()

# ── TAB 2: Performance Analytics ─────────────────────────────────────────────
with tab2:
    st.subheader("Cache Performance Metrics")

    logs = request_logger.get_all()
    if not logs:
        st.info("No request logs recorded yet. Use the Live Query Playground to generate data.")
    else:
        df = pd.DataFrame([l.to_dict() for l in logs])

        col_chart1, col_chart2 = st.columns(2)

        with col_chart1:
            st.markdown("#### Hit vs. Miss Distribution")
            hit_counts = df["is_hit"].value_counts().rename({True: "Cache Hit", False: "Cache Miss"})
            st.bar_chart(hit_counts, color="#3b82f6")

        with col_chart2:
            st.markdown("#### Latency Comparison (ms)")
            latency_df = df.groupby("is_hit")["latency_ms"].mean().rename({True: "Cache Hit", False: "Cache Miss"})
            st.bar_chart(latency_df, color="#10b981")

        st.markdown("---")
        st.markdown("#### Similarity Score Distribution (Cache Hits)")
        hits_df = df[df["is_hit"] == True]
        if not hits_df.empty:
            st.line_chart(hits_df["similarity_score"].reset_index(drop=True))
        else:
            st.caption("No cache hits recorded yet to plot similarity distribution.")

# ── TAB 3: Request Logs ───────────────────────────────────────────────────────
with tab3:
    st.subheader("Request History Log")

    logs = request_logger.get_all()
    if not logs:
        st.info("No logs available.")
    else:
        df = pd.DataFrame([l.to_dict() for l in logs])

        # Filter controls
        filter_col1, filter_col2 = st.columns([1, 3])
        with filter_col1:
            status_filter = st.selectbox("Filter Status", ["All", "Hits Only", "Misses Only"])

        with filter_col2:
            search_term = st.text_input("Search query text:", "")

        filtered_df = df.copy()
        if status_filter == "Hits Only":
            filtered_df = filtered_df[filtered_df["is_hit"] == True]
        elif status_filter == "Misses Only":
            filtered_df = filtered_df[filtered_df["is_hit"] == False]

        if search_term.strip():
            filtered_df = filtered_df[
                filtered_df["query"].str.contains(search_term, case=False, na=False)
            ]

        st.dataframe(
            filtered_df,
            column_config={
                "is_hit": st.column_config.CheckboxColumn("Hit?"),
                "similarity_score": st.column_config.NumberColumn("Similarity", format="%.4f"),
                "latency_ms": st.column_config.NumberColumn("Latency (ms)", format="%.2f"),
            },
            use_container_width=True,
            hide_index=True,
        )

# ── TAB 4: Cache Vector Store Inspector ───────────────────────────────────────
with tab4:
    st.subheader("Stored Vector Cache Index")
    st.caption("Inspect all query-response pairs currently indexed in FAISS.")

    policy_bits = []
    if cache.ttl_seconds > 0:
        policy_bits.append(f"TTL {cache.ttl_seconds / 3600:.1f}h")
    if cache.max_size > 0:
        policy_bits.append(f"Max {cache.max_size} entries")
    st.caption(f"Eviction policy: {' · '.join(policy_bits) if policy_bits else 'None (unbounded growth)'}")

    if cache.is_empty:
        st.info("The cache is currently empty.")
    else:
        st.write(f"Total Vector Entries: **{cache.size}**")
        cache_df = pd.DataFrame(cache.metadata)
        cache_df.insert(0, "Index ID", range(len(cache_df)))
        st.dataframe(cache_df, use_container_width=True, hide_index=True)
