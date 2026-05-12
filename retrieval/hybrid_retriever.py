"""Hybrid retriever combining vector and BM25 search with Reciprocal Rank Fusion."""

from typing import Optional

from models.knowledge import RetrievalResult
from retrieval.bm25_store import BM25Store
from retrieval.vector_store import VectorStore


class HybridRetriever:
    """Combines vector (semantic) and BM25 (keyword) retrieval using RRF."""

    def __init__(
        self,
        vector_store: VectorStore | None = None,
        bm25_store: BM25Store | None = None,
        vector_weight: float = 0.5,
        bm25_weight: float = 0.5,
        rrf_k: int = 60,
    ):
        self._vector_store = vector_store or VectorStore()
        self._bm25_store = bm25_store or BM25Store()
        self._vector_weight = vector_weight
        self._bm25_weight = bm25_weight
        self._rrf_k = rrf_k  # RRF constant

    def search(
        self,
        query: str,
        k: int = 10,
        category_filter: Optional[str] = None,
    ) -> list[RetrievalResult]:
        """Perform hybrid search with Reciprocal Rank Fusion.

        RRF Score = Σ (weight / (rrf_k + rank)) for each retriever.
        This avoids the need for score normalization across different scoring systems.

        Args:
            query: The search query.
            k: Number of final results to return.
            category_filter: Optional category to filter results.

        Returns:
            List of RetrievalResult objects sorted by fused RRF score.
        """
        # Fetch candidates from both retrievers
        fetch_k = k * 3  # Over-fetch for better fusion
        vector_results = self._vector_store.search(query, k=fetch_k, category_filter=category_filter)
        bm25_results = self._bm25_store.search(query, k=fetch_k, category_filter=category_filter)

        # Build RRF scores
        rrf_scores: dict[str, float] = {}
        result_map: dict[str, RetrievalResult] = {}

        # Score vector results
        for rank, result in enumerate(vector_results):
            key = f"{result.article_id}:{result.chunk_text[:100]}"
            rrf_score = self._vector_weight / (self._rrf_k + rank + 1)
            rrf_scores[key] = rrf_scores.get(key, 0.0) + rrf_score
            if key not in result_map:
                result_map[key] = result

        # Score BM25 results
        for rank, result in enumerate(bm25_results):
            key = f"{result.article_id}:{result.chunk_text[:100]}"
            rrf_score = self._bm25_weight / (self._rrf_k + rank + 1)
            rrf_scores[key] = rrf_scores.get(key, 0.0) + rrf_score
            if key not in result_map:
                result_map[key] = result

        # Sort by fused RRF score
        sorted_keys = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)

        # Build final results
        final_results = []
        for key in sorted_keys[:k]:
            result = result_map[key]
            final_results.append(RetrievalResult(
                article_id=result.article_id,
                article_title=result.article_title,
                category=result.category,
                chunk_text=result.chunk_text,
                relevance_score=rrf_scores[key],
                retrieval_method="hybrid",
                metadata=result.metadata,
            ))

        return final_results
