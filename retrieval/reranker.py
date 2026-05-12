"""Cross-encoder re-ranker for improving retrieval precision."""

from models.knowledge import RetrievalResult


class Reranker:
    """Re-ranks retrieval results using a cross-encoder model.

    Cross-encoders process query-document pairs jointly, providing
    much higher precision than bi-encoders at the cost of speed.
    Used as a final refinement step after hybrid retrieval.

    Uses lazy loading — CrossEncoder model is NOT imported or loaded
    at __init__ time, only on the first rerank() call. This keeps
    startup memory under 512MB on Render free tier.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self._model_name = model_name
        self._model = None  # Lazy: loaded on first rerank() call

    def _get_model(self):
        """Load the cross-encoder model lazily on first use."""
        if self._model is None:
            from sentence_transformers import CrossEncoder  # Lazy import
            self._model = CrossEncoder(self._model_name)
        return self._model

    def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
        top_k: int = 5,
    ) -> list[RetrievalResult]:
        """Re-rank results using cross-encoder scores.

        Args:
            query: The original search query.
            results: Candidate results from hybrid retrieval.
            top_k: Number of top results to return after re-ranking.

        Returns:
            Re-ranked list of RetrievalResult objects.
        """
        if not results:
            return []

        model = self._get_model()

        # Create query-document pairs for the cross-encoder
        pairs = [(query, result.chunk_text) for result in results]

        # Score all pairs
        scores = model.predict(pairs)

        # Attach scores and sort
        scored_results = list(zip(results, scores))
        scored_results.sort(key=lambda x: x[1], reverse=True)

        # Build re-ranked results
        reranked = []
        for result, score in scored_results[:top_k]:
            reranked.append(RetrievalResult(
                article_id=result.article_id,
                article_title=result.article_title,
                category=result.category,
                chunk_text=result.chunk_text,
                relevance_score=float(score),
                retrieval_method="hybrid+reranked",
                metadata=result.metadata,
            ))

        return reranked
