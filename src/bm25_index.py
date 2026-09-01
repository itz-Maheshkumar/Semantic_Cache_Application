"""
bm25_index.py — BM25 keyword search index for hybrid cache retrieval.

Wraps rank_bm25.BM25Okapi to provide lexical (keyword) scoring over the same
query corpus stored in the semantic cache, so exact terms, acronyms, and IDs
that a dense embedding model tends to blur together can still surface a
strong match. Used alongside — never instead of — the FAISS vector index;
see src/hybrid_search.py for how the two scores are combined.
"""

import re
from typing import List, Optional, Sequence, Tuple

from rank_bm25 import BM25Okapi

from src.logger import get_logger

log = get_logger(__name__)

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# A small, generic English stopword list. Filtering these out keeps BM25
# focused on the content words (product names, error codes, domain terms)
# that actually distinguish one cached query from another — without this,
# nearly every query would "match" on shared words like "how" or "the".
_STOPWORDS = frozenset(
    """
    a an the this that these those
    is are was were be been being am
    i you he she it we they me him her us them
    my your his its our their mine yours ours theirs
    do does did doing
    of in on at to for from with without into onto over under
    and or but nor so if then than as
    what which who whom whose when where why how
    can could will would should shall may might must
    not no yes
    """.split()
)


def tokenize(text: str) -> List[str]:
    """
    Lowercase, alphanumeric-only tokenization used for both indexing and
    querying, with common stopwords removed so BM25 scores reflect content
    overlap rather than shared filler words.
    """
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS]


class BM25Index:
    """
    Keyword search index over cached queries, kept in sync with
    SemanticCache.metadata by the owner (SemanticCache calls rebuild()
    whenever the underlying query list changes).

    rank_bm25 has no incremental-update API — every insertion changes
    corpus-wide document-frequency statistics — so the index is rebuilt
    from the full corpus rather than updated in place. For the small-to-
    medium query corpora this cache is designed for, that rebuild is cheap
    enough to do on every write.
    """

    def __init__(self) -> None:
        self._bm25: Optional[BM25Okapi] = None
        self._corpus_size = 0

    def rebuild(self, queries: Sequence[str]) -> None:
        """
        Rebuild the BM25 index from scratch over the given list of query
        strings. `queries` is expected to be index-aligned with the caller's
        metadata list (i.e. queries[i] corresponds to FAISS vector row i).
        """
        if not queries:
            self._bm25 = None
            self._corpus_size = 0
            return

        tokenized_corpus = [tokenize(q) for q in queries]
        self._bm25 = BM25Okapi(tokenized_corpus)
        self._corpus_size = len(queries)
        log.debug(f"BM25 index rebuilt over {self._corpus_size} queries.")

    def search(self, query: str, top_k: int) -> List[Tuple[int, float]]:
        """
        Score `query` against every document currently in the index.

        Args:
            query: Raw input query string (tokenized internally).
            top_k: Maximum number of top-scoring documents to return.

        Returns:
            List of (corpus_index, raw_bm25_score) tuples for the top_k
            highest-scoring documents, sorted descending by score. Empty
            list if the index is empty or the query has no scorable tokens.
        """
        if self._bm25 is None or self._corpus_size == 0:
            return []

        tokenized_query = tokenize(query)
        if not tokenized_query:
            return []

        scores = self._bm25.get_scores(tokenized_query)
        k = max(0, min(top_k, len(scores)))
        if k == 0:
            return []

        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [(i, float(scores[i])) for i in top_indices]

    @property
    def is_empty(self) -> bool:
        """Return True if the index currently has no documents."""
        return self._bm25 is None or self._corpus_size == 0
