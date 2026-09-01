"""
cache.py — Semantic Cache Engine using FAISS vector search and JSON metadata persistence.
"""

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
)
from src.embedder import Embedder
from src.hybrid_search import blend
from src.logger import get_logger
from src.models import CacheResult

log = get_logger(__name__)


class SemanticCache:
    """
    FAISS-backed semantic cache layer for storing and retrieving query-response pairs based on embedding similarity.

    Optionally runs in hybrid mode, where lookups blend FAISS vector
    similarity with BM25 keyword scoring (src/bm25_index.py, src/hybrid_search.py)
    over the same query corpus, instead of relying on vector similarity alone.
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
    ):
        """
        Initialize the SemanticCache engine.

        Args:
            embedder: Optional Embedder instance. If None, creates a new Embedder instance.
            threshold: Minimum cosine similarity score (0.0 - 1.0) to consider a cache hit
                in vector-only mode (i.e. when hybrid_enabled is False).
            index_path: Path for saving/loading the FAISS index file.
            metadata_path: Path for saving/loading the cache metadata JSON file.
            auto_load: If True, attempts to load existing index/metadata from disk if present.
            hybrid_enabled: If True, lookups blend vector similarity with BM25 keyword
                scoring instead of using vector similarity alone.
            vector_weight: Weight given to the vector (cosine similarity) score when blending.
            bm25_weight: Weight given to the normalized BM25 (keyword) score when blending.
            hybrid_top_k: Number of top candidates each retriever contributes to the merged
                candidate pool before ranking.
            hybrid_threshold: Minimum blended hybrid score (0.0 - 1.0) to consider a cache
                hit when hybrid_enabled is True.
        """
        self.embedder = embedder if embedder is not None else Embedder()
        self.threshold = threshold
        self.index_path = Path(index_path)
        self.metadata_path = Path(metadata_path)

        self.hybrid_enabled = hybrid_enabled
        self.vector_weight = vector_weight
        self.bm25_weight = bm25_weight
        self.hybrid_top_k = hybrid_top_k
        self.hybrid_threshold = hybrid_threshold

        self.dimension = self.embedder.embedding_dim
        # IndexFlatIP (Inner Product) measures dot product. With L2-normalized vectors, dot product equals cosine similarity.
        self.index: faiss.IndexFlatIP = faiss.IndexFlatIP(self.dimension)
        self.metadata: List[Dict[str, Any]] = []
        # Keyword index mirroring `metadata`'s queries, index-aligned with the FAISS rows.
        self.bm25_index = BM25Index()

        if auto_load:
            self.load()

    def get(self, query: str) -> Optional[CacheResult]:
        """
        Look up a query in the cache.

        Args:
            query: Input user query string.

        Returns:
            CacheResult if a cached query passes the similarity threshold, else None (cache miss).
        """
        if self.is_empty:
            log.debug("Cache lookup skipped — cache is empty.")
            return None

        if self.hybrid_enabled:
            return self._get_hybrid(query)
        return self._get_vector_only(query)

    def _get_vector_only(self, query: str) -> Optional[CacheResult]:
        """Pure FAISS vector-similarity lookup (the original cache behavior)."""
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
            log.info(
                f"Cache HIT [score: {best_score:.4f}] for query: '{query}' -> matched: '{matched_entry['query']}'"
            )
            return CacheResult(
                query=query,
                response=matched_entry["response"],
                similarity_score=best_score,
                cached_at=matched_entry["cached_at"],
                matched_query=matched_entry["query"],
            )

        log.info(f"Cache MISS [best score: {best_score:.4f}] for query: '{query}'")
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

    def put(self, query: str, response: str, auto_save: bool = True) -> None:
        """
        Store a new query-response pair in the vector index and metadata store.

        Args:
            query: Input user query string.
            response: Generated LLM response string.
            auto_save: If True, persists index and metadata to disk immediately.
        """
        query_vector = self.embedder.embed(query)
        query_matrix = np.expand_dims(query_vector, axis=0)

        # Add vector to FAISS index
        self.index.add(query_matrix)

        # Add corresponding entry to metadata list
        now_iso = datetime.now(timezone.utc).isoformat()
        entry = {
            "query": query,
            "response": response,
            "cached_at": now_iso,
        }
        self.metadata.append(entry)
        self._rebuild_bm25()

        log.info(f"Added query-response pair to cache (total entries: {self.size}).")

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
            self._rebuild_bm25()
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
        """Rebuild the BM25 keyword index from the current metadata's queries."""
        self.bm25_index.rebuild([entry["query"] for entry in self.metadata])

    @property
    def size(self) -> int:
        """Return total number of cached entries."""
        return len(self.metadata)

    @property
    def is_empty(self) -> bool:
        """Return True if cache contains zero entries."""
        return self.size == 0
