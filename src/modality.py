"""
modality.py — Shared type for what kind of content a SemanticCache stores.

Kept as its own tiny module (rather than living in cache.py or config.py) so
any module can reference it — embedders, the pipeline, the request logger —
without importing the rest of cache.py.
"""

from enum import Enum


class Modality(str, Enum):
    """
    What a cache instance's entries are keyed by.

    TEXT: a normal string query, embedded with a sentence-transformers text
        model (src/embedder.py). Supports the full feature set: hybrid
        (vector + BM25) search, TTL/LRU eviction.
    IMAGE: an image (file path, raw bytes, or a PIL.Image), embedded with a
        CLIP model (src/image_embedder.py). Vector-similarity lookup only —
        BM25/hybrid search is keyword search over text and has no meaning
        for an image, so it's always disabled for this modality. TTL/LRU
        eviction still applies (they operate on timestamps/vectors, not on
        the query's type).

    Audio isn't a third value here: an audio query is transcribed to text
    first (src/audio_transcriber.py) and then cached exactly like any other
    TEXT entry — see CachePipeline.process_audio_query. "Audio" is tracked
    only as a `source_modality` label on the resulting RequestLog, for
    dashboard/analytics purposes; the cache itself never stores audio.
    """

    TEXT = "text"
    IMAGE = "image"

    def __str__(self) -> str:  # nicer f-string/log output than "Modality.TEXT"
        return self.value
