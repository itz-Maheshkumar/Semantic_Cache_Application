"""
pipeline.py — Main Cache Pipeline orchestrator for query lookup, LLM fallback, and request logging.
"""

import time
from typing import Dict, Any, Optional

from src.cache import SemanticCache
from src.llm import LLMClient
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
    """

    def __init__(
        self,
        cache: Optional[SemanticCache] = None,
        llm_client: Optional[LLMClient] = None,
        request_logger: Optional[RequestLogger] = None,
    ):
        """
        Initialize the CachePipeline.

        Args:
            cache: SemanticCache instance. Created if None.
            llm_client: LLMClient instance. Created if None.
            request_logger: RequestLogger instance. Created if None.
        """
        self.cache = cache if cache is not None else SemanticCache()
        self.llm_client = llm_client if llm_client is not None else LLMClient()
        self.request_logger = (
            request_logger if request_logger is not None else RequestLogger()
        )
        log.info("CachePipeline initialized and ready for queries.")

    def process_query(
        self, query: str, system_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process a user query through the semantic cache pipeline.

        Args:
            query: Input prompt string.
            system_prompt: Optional system instruction prompt for LLM generation.

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
        }
