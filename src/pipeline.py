"""
pipeline.py — Main Cache Pipeline orchestrator for query lookup, LLM fallback, and request logging.
"""

import time
from typing import Dict, Any, Optional

from src.audio_transcriber import AudioInput, AudioTranscriber
from src.cache import SemanticCache
from src.llm import LLMClient, VisionInput
from src.logger import get_logger
from src.models import RequestLog
from src.request_logger import RequestLogger

log = get_logger(__name__)


class CachePipeline:
    """
    Main entry point for the Semantic Cache application.

    Flow:
    1. Receive query.
    2. Check SemanticCache for semantically similar previous queries.
    3. On Cache HIT -> Return cached response instantly (<10ms).
    4. On Cache MISS -> Invoke LLMClient to generate fresh response, save pair to SemanticCache.
    5. Record metrics (latency, hit/miss, similarity score) via RequestLogger.

    Multi-modal support (process_image_query, process_audio_query) is
    entirely additive: `image_cache` and `audio_transcriber` default to None,
    and a pipeline built with only the original constructor arguments behaves
    exactly as it did before — those two methods simply aren't usable without
    opting in to the extra dependencies.
    """

    def __init__(
        self,
        cache: Optional[SemanticCache] = None,
        llm_client: Optional[LLMClient] = None,
        request_logger: Optional[RequestLogger] = None,
        image_cache: Optional[SemanticCache] = None,
        audio_transcriber: Optional[AudioTranscriber] = None,
    ):
        """
        Initialize the CachePipeline.

        Args:
            cache: SemanticCache instance for TEXT queries. Created if None.
            llm_client: LLMClient instance. Created if None.
            request_logger: RequestLogger instance. Created if None.
            image_cache: Optional SemanticCache instance configured with
                modality=Modality.IMAGE (and an ImageEmbedder) for
                process_image_query(). Left as None unless explicitly
                provided — unlike `cache`, one isn't created automatically,
                since it requires downloading a separate CLIP model that a
                caller may not want loaded just to build a text-only pipeline.
            audio_transcriber: Optional AudioTranscriber instance for
                process_audio_query(). Left as None unless explicitly
                provided, for the same reason.
        """
        self.cache = cache if cache is not None else SemanticCache()
        self.llm_client = llm_client if llm_client is not None else LLMClient()
        self.request_logger = (
            request_logger if request_logger is not None else RequestLogger()
        )
        self.image_cache = image_cache
        self.audio_transcriber = audio_transcriber
        log.info("CachePipeline initialized and ready for queries.")

    def process_query(
        self, query: str, system_prompt: Optional[str] = None, modality: str = "text"
    ) -> Dict[str, Any]:
        """
        Process a user query through the semantic cache pipeline.

        Args:
            query: Input prompt string.
            system_prompt: Optional system instruction prompt for LLM generation.
            modality: RequestLog label for this query's origin. Always "text"
                for a direct call; process_audio_query() passes "audio" when
                it forwards a transcript here, so the resulting log entries
                are still distinguishable from a typed query even though
                they flow through the exact same cache/LLM path.

        Returns:
            Dictionary containing query, response, hit status, similarity score, and latency.
        """
        if not query or not query.strip():
            raise ValueError("Query string cannot be empty or whitespace.")

        t_start = time.perf_counter()

        # Step 1: Query semantic cache
        cached_result = self.cache.get(query)

        if cached_result is not None:
            # --- CACHE HIT ---
            latency_ms = (time.perf_counter() - t_start) * 1000.0
            log.info(
                f"Pipeline CACHE HIT [{latency_ms:.2f}ms] — Score: {cached_result.similarity_score:.4f}"
            )

            # Record log
            log_entry = RequestLog(
                query=query,
                response=cached_result.response,
                is_hit=True,
                similarity_score=cached_result.similarity_score,
                latency_ms=latency_ms,
                matched_query=cached_result.matched_query,
                modality=modality,
            )
            self.request_logger.log(log_entry)

            return {
                "query": query,
                "response": cached_result.response,
                "is_hit": True,
                "similarity_score": round(cached_result.similarity_score, 4),
                "latency_ms": round(latency_ms, 2),
                "cached_at": cached_result.cached_at,
                "matched_query": cached_result.matched_query,
                "modality": modality,
                # Populated only when the cache is running in hybrid (vector + BM25)
                # mode; None for a plain vector-only hit.
                "vector_score": (
                    round(cached_result.vector_score, 4)
                    if cached_result.vector_score is not None
                    else None
                ),
                "bm25_score": (
                    round(cached_result.bm25_score, 4)
                    if cached_result.bm25_score is not None
                    else None
                ),
            }

        # --- CACHE MISS ---
        log.info("Pipeline CACHE MISS — Forwarding query to LLM...")
        
        # Step 2: Invoke LLM on cache miss
        llm_response = self.llm_client.generate(query, system_prompt=system_prompt)

        # Step 3: Put fresh pair into cache
        self.cache.put(query, llm_response, auto_save=True)

        latency_ms = (time.perf_counter() - t_start) * 1000.0
        log.info(f"Pipeline LLM response generated and cached [{latency_ms:.2f}ms].")

        # Record log
        log_entry = RequestLog(
            query=query,
            response=llm_response,
            is_hit=False,
            similarity_score=0.0,
            latency_ms=latency_ms,
            matched_query=None,
            modality=modality,
        )
        self.request_logger.log(log_entry)

        return {
            "query": query,
            "response": llm_response,
            "is_hit": False,
            "similarity_score": 0.0,
            "latency_ms": round(latency_ms, 2),
            "cached_at": None,
            "matched_query": None,
            "modality": modality,
            "vector_score": None,
            "bm25_score": None,
        }

    def process_image_query(
        self, image: VisionInput, prompt: Optional[str] = None, image_label: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process an image query through the multi-modal cache pipeline.

        Mirrors process_query()'s flow exactly, but looks the image up in
        `self.image_cache` (CLIP vector similarity, no BM25/hybrid — see
        src/modality.py) and, on a miss, generates a fresh response via
        LLMClient.generate_vision() instead of generate().

        Args:
            image: A file path, raw image bytes, or a PIL.Image — anything
                the pipeline's image_cache's ImageEmbedder accepts.
            prompt: Optional accompanying text prompt for LLM generation on
                a cache miss (e.g. "What's in this image?").
            image_label: Optional human-readable label to store/display in
                place of the raw image (e.g. an uploaded filename). Defaults
                to an auto-generated content-hash label — see
                SemanticCache._auto_label().

        Returns:
            Dictionary with the same shape as process_query()'s, plus a
            "modality" key.

        Raises:
            RuntimeError: If this pipeline wasn't built with an image_cache.
        """
        if self.image_cache is None:
            raise RuntimeError(
                "process_image_query() requires a pipeline built with an "
                "image_cache (a SemanticCache configured with "
                "modality=Modality.IMAGE) — none was provided."
            )

        t_start = time.perf_counter()

        cached_result = self.image_cache.get(image, query_label=image_label)

        if cached_result is not None:
            latency_ms = (time.perf_counter() - t_start) * 1000.0
            log.info(
                f"Pipeline IMAGE CACHE HIT [{latency_ms:.2f}ms] — Score: {cached_result.similarity_score:.4f}"
            )
            log_entry = RequestLog(
                query=cached_result.query,
                response=cached_result.response,
                is_hit=True,
                similarity_score=cached_result.similarity_score,
                latency_ms=latency_ms,
                matched_query=cached_result.matched_query,
                modality="image",
            )
            self.request_logger.log(log_entry)

            return {
                "query": cached_result.query,
                "response": cached_result.response,
                "is_hit": True,
                "similarity_score": round(cached_result.similarity_score, 4),
                "latency_ms": round(latency_ms, 2),
                "cached_at": cached_result.cached_at,
                "matched_query": cached_result.matched_query,
                "modality": "image",
            }

        log.info("Pipeline IMAGE CACHE MISS — Forwarding image to vision LLM...")
        llm_response = self.llm_client.generate_vision(image, query=prompt)

        self.image_cache.put(image, llm_response, auto_save=True, query_label=image_label)
        # The label put() just stored/derived is what should appear in the
        # response and the log below, not the raw image object.
        query_display = self.image_cache.display_query(image, image_label)

        latency_ms = (time.perf_counter() - t_start) * 1000.0
        log.info(f"Pipeline vision LLM response generated and cached [{latency_ms:.2f}ms].")

        log_entry = RequestLog(
            query=query_display,
            response=llm_response,
            is_hit=False,
            similarity_score=0.0,
            latency_ms=latency_ms,
            matched_query=None,
            modality="image",
        )
        self.request_logger.log(log_entry)

        return {
            "query": query_display,
            "response": llm_response,
            "is_hit": False,
            "similarity_score": 0.0,
            "latency_ms": round(latency_ms, 2),
            "cached_at": None,
            "matched_query": None,
            "modality": "image",
        }

    def process_audio_query(
        self, audio: AudioInput, system_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process an audio query through the pipeline.

        Audio isn't embedded/cached directly: it's transcribed to text via
        `self.audio_transcriber` first, and the resulting transcript is run
        through the exact same text pipeline as process_query() — two spoken
        queries that say the same thing become the same cache lookup problem
        the project already solves. "audio" is recorded only as the
        RequestLog's `modality` label; the underlying cache entry is
        indistinguishable from one created by a typed query.

        Args:
            audio: A file path, raw audio bytes, or an open binary file-like
                object — anything AudioTranscriber.transcribe() accepts.
            system_prompt: Optional system instruction prompt, forwarded to
                LLMClient.generate() on a cache miss.

        Returns:
            Dictionary with the same shape as process_query()'s, plus a
            "modality" key and a "transcript" key holding the transcribed
            text (identical to "query" — kept as a separate key so callers
            don't have to know the transcription happened to find the text).

        Raises:
            RuntimeError: If this pipeline wasn't built with an
                audio_transcriber.
        """
        if self.audio_transcriber is None:
            raise RuntimeError(
                "process_audio_query() requires a pipeline built with an "
                "audio_transcriber (an AudioTranscriber instance) — none "
                "was provided."
            )

        transcript = self.audio_transcriber.transcribe(audio)
        if not transcript or not transcript.strip():
            raise ValueError("Audio transcription produced an empty result.")

        result = self.process_query(transcript, system_prompt=system_prompt, modality="audio")
        result["transcript"] = transcript
        return result
