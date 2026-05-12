"""BM25 keyword-based retrieval store."""

import os
import pickle
from pathlib import Path
from typing import Optional

from models.knowledge import RetrievalResult


class BM25Store:
    """BM25-based keyword retrieval for KB articles."""

    def __init__(self, index_path: str | None = None):
        if index_path is None:
            index_path = str(Path(__file__).parent.parent / "chroma_db" / "bm25_index.pkl")

        self._index = None
        self._chunks = []
        self._index_path = index_path

        if os.path.exists(index_path):
            self._load_index(index_path)

    def _load_index(self, path: str):
        """Load the pre-built BM25 index from disk."""
        with open(path, "rb") as f:
            data = pickle.load(f)
            self._index = data["index"]
            self._chunks = data["chunks"]

    @property
    def is_loaded(self) -> bool:
        """Check if the BM25 index is loaded."""
        return self._index is not None

    def search(
        self,
        query: str,
        k: int = 10,
        category_filter: Optional[str] = None,
    ) -> list[RetrievalResult]:
        """Search for relevant chunks using BM25 keyword matching.

        Args:
            query: The search query.
            k: Number of results to return.
            category_filter: Optional category to filter results.

        Returns:
            List of RetrievalResult objects sorted by BM25 score.
        """
        if not self.is_loaded:
            return []

        tokenized_query = query.lower().split()
        scores = self._index.get_scores(tokenized_query)

        # Create (index, score) pairs and sort by score descending
        scored_indices = [(i, score) for i, score in enumerate(scores) if score > 0]
        scored_indices.sort(key=lambda x: x[1], reverse=True)

        results = []
        for idx, score in scored_indices:
            if len(results) >= k:
                break

            chunk = self._chunks[idx]

            # Apply category filter if specified
            if category_filter and chunk.get("category") != category_filter:
                continue

            results.append(RetrievalResult(
                article_id=chunk.get("article_id", ""),
                article_title=chunk.get("article_title", ""),
                category=chunk.get("category", ""),
                chunk_text=chunk.get("content", ""),
                relevance_score=float(score),
                retrieval_method="bm25",
                metadata={
                    "chunk_id": chunk.get("chunk_id", ""),
                    "tags": chunk.get("tags", []),
                },
            ))

        return results
