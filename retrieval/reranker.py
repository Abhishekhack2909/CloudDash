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
        plan_tier: str | None = None,
        active_integrations: list[str] | None = None,
    ) -> list[RetrievalResult]:
        """Re-rank results using cross-encoder scores with personalized boosting.

        Args:
            query: The original search query.
            results: Candidate results from hybrid retrieval.
            top_k: Number of top results to return after re-ranking.
            plan_tier: Optional customer plan tier (Starter, Pro, Enterprise) for boosting.
            active_integrations: Optional list of integrations used by customer (e.g. AWS, GCP).

        Returns:
            Re-ranked list of RetrievalResult objects.
        """
        if not results:
            return []

        model = self._get_model()

        # Create query-document pairs for the cross-encoder (include article title for context)
        pair_query = f"{query} ({', '.join(active_integrations)})" if active_integrations else query
        pairs = [(pair_query, f"{result.article_title}: {result.chunk_text}") for result in results]

        # Score all pairs
        scores = model.predict(pairs)

        # Apply personalization score boosts based on customer profile
        scored_results = []
        for result, raw_score in zip(results, scores):
            boost = 0.0

            # 1. Plan tier boost & down-weighting
            if plan_tier:
                tier_norm = str(plan_tier).lower().strip()
                applies_to = str(result.metadata.get("applies_to", "")).lower()
                if tier_norm in applies_to:
                    boost += 1.5
                elif tier_norm in ["free", "starter"] and "enterprise" in applies_to and "starter" not in applies_to and "free" not in applies_to:
                    boost -= 3.0  # Down-weight enterprise-only docs for free/starter tier
                elif "enterprise" in applies_to and tier_norm == "enterprise":
                    boost += 2.0

            # 2. Active integration boost
            if active_integrations:
                for integration in active_integrations:
                    if not integration:
                        continue
                    integ_clean = str(integration).lower().strip()
                    tags = str(result.metadata.get("tags", "")).lower()
                    title = result.article_title.lower()
                    chunk = result.chunk_text.lower()
                    if integ_clean in tags or integ_clean in title:
                        boost += 5.0
                        break
                    elif integ_clean in chunk:
                        boost += 2.0
                        break

            final_score = float(raw_score) + boost
            scored_results.append((result, final_score))

        # Sort descending by final score
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
