"""
hybrid_search.py — Score fusion for combining dense vector similarity with
BM25 keyword search into a single ranked candidate list.

This module is pure score-merging logic with no dependency on FAISS,
sentence-transformers, or rank_bm25 — it just combines two already-computed
score dictionaries. Keeping it separate from SemanticCache makes the fusion
math easy to unit test in isolation.
"""

from typing import Dict, List, NamedTuple


class HybridCandidate(NamedTuple):
    """A single ranked result from blend(), with both component scores kept
    alongside the combined one for transparency/debugging."""

    index: int
    hybrid_score: float
    vector_score: float
    bm25_score: float  # normalized to the 0..1 range, unlike the raw BM25 score


def normalize_bm25_scores(bm25_scores: Dict[int, float]) -> Dict[int, float]:
    """
    Min-max normalize raw BM25 scores onto a 0..1 scale so they can be
    linearly combined with cosine similarity (which is already ~0..1 for
    normalized embeddings).

    BM25 scores are unbounded and depend on corpus size and term rarity, so
    there's no fixed scale to divide by — min-max over the current candidate
    set is the standard way to make them comparable query-to-query.

    Edge case: when every candidate has the same raw score (most commonly
    because there is exactly one candidate), the min-max range is zero. In
    that case a positive score normalizes to 1.0 (some lexical overlap was
    found) and a zero score normalizes to 0.0 (no overlap at all) — this
    matters most for very small caches, where a single BM25 candidate is
    common.
    """
    if not bm25_scores:
        return {}

    values = list(bm25_scores.values())
    lo, hi = min(values), max(values)
    span = hi - lo

    if span <= 0:
        return {idx: (1.0 if score > 0 else 0.0) for idx, score in bm25_scores.items()}

    return {idx: (score - lo) / span for idx, score in bm25_scores.items()}


def blend(
    vector_scores: Dict[int, float],
    bm25_scores: Dict[int, float],
    vector_weight: float,
    bm25_weight: float,
) -> List[HybridCandidate]:
    """
    Merge vector and BM25 candidate pools into a single ranked list.

    Args:
        vector_scores: {corpus_index: cosine_similarity} from the FAISS search.
        bm25_scores: {corpus_index: raw_bm25_score} from the BM25 search.
        vector_weight: Weight applied to the (already 0..1) vector score.
        bm25_weight: Weight applied to the normalized (0..1) BM25 score.

    Returns:
        List of HybridCandidate sorted descending by hybrid_score. A
        candidate found by only one retriever is not dropped — it scores
        0.0 on the signal that missed it and can still win on the strength
        of the one that found it. Empty list if both inputs are empty.
    """
    normalized_bm25 = normalize_bm25_scores(bm25_scores)
    candidate_ids = set(vector_scores) | set(bm25_scores)

    candidates = [
        HybridCandidate(
            index=idx,
            vector_score=vector_scores.get(idx, 0.0),
            bm25_score=normalized_bm25.get(idx, 0.0),
            hybrid_score=(
                vector_weight * vector_scores.get(idx, 0.0)
                + bm25_weight * normalized_bm25.get(idx, 0.0)
            ),
        )
        for idx in candidate_ids
    ]
    candidates.sort(key=lambda c: c.hybrid_score, reverse=True)
    return candidates
