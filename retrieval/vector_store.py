"""ChromaDB vector store for semantic search."""

import os
from pathlib import Path
from typing import Any, Optional

import chromadb
from chromadb.utils import embedding_functions

from models.knowledge import RetrievalResult


class VectorStore:
    """ChromaDB-based vector store for KB article retrieval."""

    def __init__(self, db_dir: str | None = None, collection_name: str = "clouddash_kb"):
        if db_dir is None:
            db_dir = str(Path(__file__).parent.parent / "chroma_db")

        self._client = chromadb.PersistentClient(path=db_dir)
        # Use local sentence-transformers for zero-cost embeddings
        self._embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2",
        )
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            embedding_function=self._embedding_fn,
        )

    def search(
        self,
        query: str,
        k: int = 10,
        category_filter: Optional[str] = None,
    ) -> list[RetrievalResult]:
        """Search for relevant chunks using vector similarity.

        Args:
            query: The search query.
            k: Number of results to return.
            category_filter: Optional category to filter results.

        Returns:
            List of RetrievalResult objects sorted by relevance.
        """
        where_filter = None
        if category_filter:
            where_filter = {"category": category_filter}

        results = self._collection.query(
            query_texts=[query],
            n_results=k,
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )

        retrieval_results = []
        if results and results["documents"] and results["documents"][0]:
            for i, doc in enumerate(results["documents"][0]):
                metadata = results["metadatas"][0][i] if results["metadatas"] else {}
                distance = results["distances"][0][i] if results["distances"] else 0.0
                # ChromaDB returns L2 distance; convert to similarity score
                similarity = 1.0 / (1.0 + distance)

                retrieval_results.append(RetrievalResult(
                    article_id=metadata.get("article_id", ""),
                    article_title=metadata.get("article_title", ""),
                    category=metadata.get("category", ""),
                    chunk_text=doc,
                    relevance_score=similarity,
                    retrieval_method="vector",
                    metadata=metadata,
                ))

        return retrieval_results

    @property
    def count(self) -> int:
        """Return the number of documents in the collection."""
        return self._collection.count()
