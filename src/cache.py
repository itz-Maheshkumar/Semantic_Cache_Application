"""
cache.py — Semantic Cache Engine using FAISS vector search and JSON metadata persistence.
"""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Dict, Any
import faiss
import numpy as np

from src.bm25_index import BM25Index
from src.config import (
    SIMILARITY_THRESHOLD,
    FAISS_INDEX_PATH,
    CACHE_METADATA_PATH,
    ENABLE_HYBRID_SEARCH,
    VECTOR_WEIGHT,
    BM25_WEIGHT,
    HYBRID_TOP_K,
    HYBRID_SIMILARITY_THRESHOLD,
    CACHE_TTL_SECONDS,
    CACHE_MAX_SIZE,
)
from src.embedder import Embedder
from src.hybrid_search import blend
from src.logger import get_logger
from src.modality import Modality
from src.models import CacheResult

log = get_logger(__name__)


class SemanticCache:
    """
    FAISS-backed semantic cache layer for storing and retrieving query-response pairs based on embedding similarity.

    Optionally runs in hybrid mode, where lookups blend FAISS vector
    similarity with BM25 keyword scoring (src/bm25_index.py, src/hybrid_search.py)
    over the same query corpus, instead of relying on vector similarity alone.

    Also optionally prunes itself: a TTL evicts entries past a maximum age,
    and an LRU cap evicts the least-recently-accessed entries once the cache
    grows past a maximum size. Both are off (unbounded growth) by default.

    Also supports non-text modalities (src/modality.py) for multi-modal
    caching: pass modality=Modality.IMAGE with an ImageEmbedder
    (src/image_embedder.py) instead of a text Embedder to cache image
    queries by CLIP vector similarity. Hybrid/BM25 search is keyword search
    over text and has no meaning for an image, so it is always forced off
    for any non-TEXT modality, regardless of hybrid_enabled. Everything
    else — eviction, save/load, dashboard display — works the same way
    across modalities, since the FAISS/eviction machinery only ever deals
    in vectors, not the original query type.
    """

    def __init__(
        self,
        embedder: Optional[Embedder] = None,
        threshold: float = SIMILARITY_THRESHOLD,
        index_path: Path = FAISS_INDEX_PATH,
        metadata_path: Path = CACHE_METADATA_PATH,
        auto_load: bool = True,
        hybrid_enabled: bool = ENABLE_HYBRID_SEARCH,
        vector_weight: float = VECTOR_WEIGHT,
        bm25_weight: float = BM25_WEIGHT,
        hybrid_top_k: int = HYBRID_TOP_K,
        hybrid_threshold: float = HYBRID_SIMILARITY_THRESHOLD,
        ttl_seconds: int = CACHE_TTL_SECONDS,
        max_size: int = CACHE_MAX_SIZE,
        modality: Modality = Modality.TEXT,
    ):
        """
        Initialize the SemanticCache engine.

        Args:
            embedder: Optional embedder instance (an Embedder for TEXT, an
                ImageEmbedder for IMAGE — anything with .embed(x) and
                .embedding_dim). If None, creates a new text Embedder.
            threshold: Minimum cosine similarity score (0.0 - 1.0) to consider a cache hit
                in vector-only mode (i.e. when hybrid_enabled is False, or the modality
                isn't TEXT — hybrid is always vector-only for non-text modalities).
            index_path: Path for saving/loading the FAISS index file.
            metadata_path: Path for saving/loading the cache metadata JSON file.
            auto_load: If True, attempts to load existing index/metadata from disk if present.
            hybrid_enabled: If True, lookups blend vector similarity with BM25 keyword
                scoring instead of using vector similarity alone. Ignored (forced off)
                unless modality is TEXT.
            vector_weight: Weight given to the vector (cosine similarity) score when blending.
            bm25_weight: Weight given to the normalized BM25 (keyword) score when blending.
            hybrid_top_k: Number of top candidates each retriever contributes to the merged
                candidate pool before ranking.
            hybrid_threshold: Minimum blended hybrid score (0.0 - 1.0) to consider a cache
                hit when hybrid_enabled is True.
            ttl_seconds: Entries older than this (by `cached_at`) are pruned automatically.
                0 disables TTL pruning.
            max_size: Maximum number of entries to retain; once exceeded, the
                least-recently-accessed entries are evicted. 0 disables the cap.
            modality: What kind of query this cache indexes — Modality.TEXT (default)
                or Modality.IMAGE. See src/modality.py.
        """
        self.embedder = embedder if embedder is not None else Embedder()
        self.threshold = threshold
        self.index_path = Path(index_path)
        self.metadata_path = Path(metadata_path)

        self.modality = Modality(modality)
        # BM25/hybrid search is keyword search over text; it has no meaning
        # for a non-text modality, so it's never available for one, no
        # matter what the caller (or a live dashboard toggle) sets it to.
        self.hybrid_enabled = hybrid_enabled and self.modality is Modality.TEXT
        self.vector_weight = vector_weight
        self.bm25_weight = bm25_weight
        self.hybrid_top_k = hybrid_top_k
        self.hybrid_threshold = hybrid_threshold

        self.ttl_seconds = ttl_seconds
        self.max_size = max_size

        self.dimension = self.embedder.embedding_dim
        # IndexFlatIP (Inner Product) measures dot product. With L2-normalized vectors, dot product equals cosine similarity.
        self.index: faiss.IndexFlatIP = faiss.IndexFlatIP(self.dimension)
        self.metadata: List[Dict[str, Any]] = []
        # Keyword index mirroring `metadata`'s queries, index-aligned with the FAISS rows.
        self.bm25_index = BM25Index()

        if auto_load:
            self.load()

    def get(self, query: Any, query_label: Optional[str] = None) -> Optional[CacheResult]:
        """
        Look up a query in the cache.

        Args:
            query: For a TEXT cache, the input query string. For an IMAGE
                cache, anything the configured ImageEmbedder accepts (file
                path, bytes, or a PIL.Image).
            query_label: Non-text modalities only — a short human-readable
                label for `query`, used to populate CacheResult.query (which
                is always a string; raw image bytes aren't something you can
                sensibly echo back or log). Defaults to an auto-generated
                content-hash label when omitted. Ignored for TEXT.

        Returns:
            CacheResult if a cached query passes the similarity threshold, else None (cache miss).
        """
        if self.is_empty:
            log.debug("Cache lookup skipped — cache is empty.")
            return None

        # hybrid_enabled is already forced off for non-TEXT modalities at
        # construction time; the modality check here is just defense in
        # depth against a caller flipping it back on after the fact.
        if self.hybrid_enabled and self.modality is Modality.TEXT:
            return self._get_hybrid(query)
        return self._get_vector_only(query, query_label)

    def display_query(self, query: Any, query_label: Optional[str]) -> str:
        """
        The string to store/echo back as a CacheResult's `query`/
        `matched_query`, and (for a caller like CachePipeline) to log and
        show in place of a raw, non-displayable query object.

        For a TEXT cache this is just `query` itself. For a non-text cache
        it's `query_label` if the caller supplied one, else an
        auto-generated content-hash label (see _auto_label()) — public so
        callers that build their own query_label (or need to know what
        label a put()/get() call used/will use) don't have to reimplement
        this fallback themselves.
        """
        if self.modality is Modality.TEXT:
            return query
        return query_label or self._auto_label(query)

    def _auto_label(self, query: Any) -> str:
        """
        Derive a stable, displayable label for a non-text query that wasn't
        given an explicit `query_label` — e.g. an image opened straight from
        bytes with no filename to fall back on.

        Raw image bytes/PIL.Image objects can't be JSON-serialized or stored
        in metadata's string-typed `query` field, so this hashes the query's
        content instead: two identical images always get the same label
        (useful for logs/dedup), and the label is short enough to display.
        """
        if isinstance(query, (str, Path)):
            payload = str(query).encode("utf-8")
        elif isinstance(query, (bytes, bytearray)):
            payload = bytes(query)
        elif hasattr(query, "tobytes"):
            # PIL.Image and numpy arrays both expose .tobytes()
            payload = query.tobytes()
        else:
            payload = repr(query).encode("utf-8")
        digest = hashlib.sha256(payload).hexdigest()[:12]
        return f"{self.modality.value}:{digest}"

    def _get_vector_only(self, query: Any, query_label: Optional[str] = None) -> Optional[CacheResult]:
        """Pure FAISS vector-similarity lookup (the original cache behavior)."""
        query_display = self.display_query(query, query_label)

        # Generate L2-normalized embedding for input query
        query_vector = self.embedder.embed(query)
        # FAISS search expects a 2D float32 array of shape (1, dimension)
        query_matrix = np.expand_dims(query_vector, axis=0)

        # Search FAISS index for top 1 nearest vector
        similarities, indices = self.index.search(query_matrix, k=1)

        best_score = float(similarities[0][0])
        best_idx = int(indices[0][0])

        log.debug(f"Cache search score: {best_score:.4f} (threshold: {self.threshold})")

        # Check if top score meets or exceeds similarity threshold
        if best_idx >= 0 and best_idx < len(self.metadata) and best_score >= self.threshold:
            matched_entry = self.metadata[best_idx]
            self._touch(matched_entry)
            log.info(
                f"Cache HIT [score: {best_score:.4f}] for query: '{query_display}' -> matched: '{matched_entry['query']}'"
            )
            return CacheResult(
                query=query_display,
                response=matched_entry["response"],
                similarity_score=best_score,
                cached_at=matched_entry["cached_at"],
                matched_query=matched_entry["query"],
            )

        log.info(f"Cache MISS [best score: {best_score:.4f}] for query: '{query_display}'")
        return None

    def _get_hybrid(self, query: str) -> Optional[CacheResult]:
        """
        Hybrid lookup: retrieves top candidates from FAISS (semantic) and BM25
        (keyword) independently, blends their scores via src/hybrid_search.blend,
        and accepts the best-ranked candidate as a hit if it clears
        `hybrid_threshold`.
        """
        query_vector = self.embedder.embed(query)
        query_matrix = np.expand_dims(query_vector, axis=0)

        top_k = min(self.hybrid_top_k, self.size)
        raw_scores, raw_indices = self.index.search(query_matrix, k=top_k)
        vector_scores = {
            int(idx): float(score)
            for idx, score in zip(raw_indices[0], raw_scores[0])
            if idx >= 0
        }

        bm25_scores = dict(self.bm25_index.search(query, top_k=top_k))

        candidates = blend(vector_scores, bm25_scores, self.vector_weight, self.bm25_weight)
        if not candidates:
            log.info(f"Cache MISS [no candidates] for query: '{query}'")
            return None

        best = candidates[0]
        log.debug(
            f"Hybrid search best candidate idx={best.index} vector={best.vector_score:.4f} "
            f"bm25={best.bm25_score:.4f} hybrid={best.hybrid_score:.4f} "
            f"(threshold: {self.hybrid_threshold})"
        )

        if 0 <= best.index < len(self.metadata) and best.hybrid_score >= self.hybrid_threshold:
            matched_entry = self.metadata[best.index]
            self._touch(matched_entry)
            log.info(
                f"Cache HIT [hybrid: {best.hybrid_score:.4f}, vector: {best.vector_score:.4f}, "
                f"bm25: {best.bm25_score:.4f}] for query: '{query}' -> matched: '{matched_entry['query']}'"
            )
            return CacheResult(
                query=query,
                response=matched_entry["response"],
                similarity_score=best.hybrid_score,
                cached_at=matched_entry["cached_at"],
                matched_query=matched_entry["query"],
                vector_score=best.vector_score,
                bm25_score=best.bm25_score,
            )

        log.info(f"Cache MISS [best hybrid score: {best.hybrid_score:.4f}] for query: '{query}'")
        return None

    def put(
        self,
        query: Any,
        response: str,
        auto_save: bool = True,
        query_label: Optional[str] = None,
    ) -> None:
        """
        Store a new query-response pair in the vector index and metadata store.

        Args:
            query: Input query. For a TEXT cache, a query string. For an
                IMAGE cache, anything ImageEmbedder.embed() accepts (a file
                path, raw bytes, or a PIL.Image) — not a string, so it can't
                be stored verbatim in metadata/logs.
            response: Generated LLM response string.
            auto_save: If True, persists index and metadata to disk immediately.
            query_label: Optional human-readable label to store/display in
                place of a non-text query (e.g. an uploaded filename). Ignored
                for TEXT caches. When omitted for a non-text query, a stable
                label is auto-derived from the content's hash — see
                _auto_label().
        """
        query_vector = self.embedder.embed(query)
        query_matrix = np.expand_dims(query_vector, axis=0)

        # Add vector to FAISS index
        self.index.add(query_matrix)

        # Add corresponding entry to metadata list. `last_accessed_at` starts
        # equal to `cached_at` — a freshly inserted entry is, by definition,
        # the most recently used one.
        now_iso = datetime.now(timezone.utc).isoformat()
        entry = {
            "query": self.display_query(query, query_label),
            "response": response,
            "cached_at": now_iso,
            "last_accessed_at": now_iso,
        }
        self.metadata.append(entry)

        log.info(f"Added query-response pair to cache (total entries: {self.size}).")

        # Eviction runs on every write (not on every get(), which needs to stay
        # fast): each call below rebuilds the BM25 index itself if it evicts
        # anything, so only rebuild it here when neither one did.
        expired = self.prune_expired()
        evicted = self.enforce_capacity()
        if expired or evicted:
            log.info(f"Eviction on put(): {expired} expired (TTL), {evicted} evicted (LRU capacity).")
        else:
            self._rebuild_bm25()

        if auto_save:
            self.save()

    def save(self) -> None:
        """
        Persist FAISS vector index and metadata JSON to disk.
        """
        try:
            self.index_path.parent.mkdir(parents=True, exist_ok=True)
            self.metadata_path.parent.mkdir(parents=True, exist_ok=True)

            # Write FAISS index
            faiss.write_index(self.index, str(self.index_path))

            # Write metadata JSON
            with open(self.metadata_path, "w", encoding="utf-8") as f:
                json.dump(self.metadata, f, indent=2, ensure_ascii=False)

            log.info(f"Cache saved successfully to disk ({self.size} entries).")
        except Exception as e:
            log.error(f"Failed to save cache to disk: {e}")
            raise

    def load(self) -> bool:
        """
        Load FAISS vector index and metadata JSON from disk if both exist.

        Returns:
            True if successfully loaded, False if files were missing or loading failed.
        """
        if not self.index_path.exists() or not self.metadata_path.exists():
            log.debug("Cache load skipped — missing index or metadata file on disk.")
            return False

        try:
            # Read FAISS index
            loaded_index = faiss.read_index(str(self.index_path))

            # Read metadata JSON
            with open(self.metadata_path, "r", encoding="utf-8") as f:
                loaded_metadata = json.load(f)

            if loaded_index.ntotal != len(loaded_metadata):
                log.warning(
                    f"Cache discrepancy detected: index has {loaded_index.ntotal} vectors, "
                    f"metadata has {len(loaded_metadata)} items. Resetting cache."
                )
                self.clear(delete_files=False)
                return False

            self.index = loaded_index
            self.metadata = loaded_metadata
            # Metadata saved before eviction support existed won't have
            # `last_accessed_at` — backfill it from `cached_at` so LRU has
            # something to sort by.
            for entry in self.metadata:
                entry.setdefault("last_accessed_at", entry["cached_at"])
            self._rebuild_bm25()

            expired = self.prune_expired()
            evicted = self.enforce_capacity()
            if expired or evicted:
                log.info(
                    f"Eviction on load(): {expired} expired (TTL), {evicted} evicted (LRU capacity)."
                )
                self.save()

            log.info(f"Cache loaded successfully from disk ({self.size} entries).")
            return True
        except Exception as e:
            log.error(f"Error loading cache files from disk: {e}")
            return False

    def clear(self, delete_files: bool = True) -> None:
        """
        Clear all entries from in-memory cache and optionally delete persisted files.

        Args:
            delete_files: If True, removes `.faiss` and `.json` files from disk.
        """
        self.index = faiss.IndexFlatIP(self.dimension)
        self.metadata = []
        self._rebuild_bm25()

        if delete_files:
            if self.index_path.exists():
                self.index_path.unlink()
            if self.metadata_path.exists():
                self.metadata_path.unlink()
            log.info("Cache files deleted from disk.")

        log.info("Cache cleared.")

    def _rebuild_bm25(self) -> None:
        """
        Rebuild the BM25 keyword index from the current metadata's queries.

        A no-op for non-TEXT caches: BM25 is keyword search over query text,
        which has no meaning for an image cache (hybrid search is always
        forced off for those — see __init__), so there's nothing to index
        and no reason to pay the rebuild cost. Guarding here (rather than at
        each of this method's call sites — put(), load(), clear(),
        _rebuild_from_indices()) keeps every call site oblivious to modality.
        """
        if self.modality is not Modality.TEXT:
            return
        self.bm25_index.rebuild([entry["query"] for entry in self.metadata])

    def _touch(self, entry: Dict[str, Any]) -> None:
        """
        Mark a metadata entry as just-accessed, for LRU eviction.

        Updates the entry in memory only — not persisted to disk until the
        next put()/save(), so a burst of cache hits doesn't cost any disk
        I/O and hit latency stays fast. Eviction decisions only ever run at
        write time (see put(), load()), by which point this is up to date.
        """
        entry["last_accessed_at"] = datetime.now(timezone.utc).isoformat()

    def prune_expired(self) -> int:
        """
        Evict all entries older than `ttl_seconds` (measured from `cached_at`).

        A no-op (returns 0 immediately, without touching the index) when TTL
        pruning is disabled (ttl_seconds <= 0) or the cache is empty.

        Returns:
            Number of entries evicted.
        """
        if self.ttl_seconds <= 0 or self.is_empty:
            return 0

        now = datetime.now(timezone.utc)
        keep_indices = [
            i for i, entry in enumerate(self.metadata)
            if (now - datetime.fromisoformat(entry["cached_at"])).total_seconds() <= self.ttl_seconds
        ]

        removed = self.size - len(keep_indices)
        if removed:
            log.info(f"TTL pruning evicted {removed} entr{'y' if removed == 1 else 'ies'} (> {self.ttl_seconds}s old).")
            self._rebuild_from_indices(keep_indices)
        return removed

    def enforce_capacity(self) -> int:
        """
        If the cache holds more than `max_size` entries, evict the
        least-recently-accessed ones (by `last_accessed_at`) until it fits.

        A no-op (returns 0 immediately, without touching the index) when the
        capacity cap is disabled (max_size <= 0) or the cache is already
        within it.

        Returns:
            Number of entries evicted.
        """
        if self.max_size <= 0 or self.size <= self.max_size:
            return 0

        # Oldest (least-recently-used) access time first.
        order_by_recency = sorted(
            range(len(self.metadata)),
            key=lambda i: self.metadata[i]["last_accessed_at"],
        )
        num_to_evict = self.size - self.max_size
        evict_set = set(order_by_recency[:num_to_evict])
        keep_indices = [i for i in range(len(self.metadata)) if i not in evict_set]

        log.info(f"LRU eviction removed {num_to_evict} least-recently-used entr{'y' if num_to_evict == 1 else 'ies'} (max_size={self.max_size}).")
        self._rebuild_from_indices(keep_indices)
        return num_to_evict

    def _rebuild_from_indices(self, keep_indices: List[int]) -> None:
        """
        Rebuild the FAISS index, metadata list, and BM25 index to contain
        only the entries at `keep_indices` (in the given order).

        Vectors are reconstructed directly from the existing FAISS index
        (IndexFlat stores them exactly) rather than re-embedding the surviving
        queries' text, so eviction never needs to call the embedding model.
        """
        if keep_indices:
            all_vectors = self.index.reconstruct_n(0, self.index.ntotal)
            surviving_vectors = all_vectors[keep_indices]
        else:
            surviving_vectors = np.empty((0, self.dimension), dtype=np.float32)

        new_index = faiss.IndexFlatIP(self.dimension)
        if len(surviving_vectors):
            new_index.add(surviving_vectors)

        self.index = new_index
        self.metadata = [self.metadata[i] for i in keep_indices]
        self._rebuild_bm25()

    @property
    def size(self) -> int:
        """Return total number of cached entries."""
        return len(self.metadata)

    @property
    def is_empty(self) -> bool:
        """Return True if cache contains zero entries."""
        return self.size == 0
