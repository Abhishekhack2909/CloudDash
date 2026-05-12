"""Main RAG pipeline — orchestrates query rewriting, hybrid retrieval, and re-ranking."""

import os
from typing import Optional

from langchain_groq import ChatGroq

from models.knowledge import RetrievalResult
from retrieval.bm25_store import BM25Store
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.reranker import Reranker
from retrieval.vector_store import VectorStore


class RAGPipeline:
    """Full RAG pipeline: query rewrite → hybrid retrieval → re-ranking.

    The pipeline:
    1. Rewrites the user query using conversation context for better retrieval.
    2. Performs hybrid search (vector + BM25) with RRF fusion.
    3. Re-ranks top candidates using a cross-encoder.
    4. Returns final results with citations.
    """

    def __init__(
        self,
        vector_store: VectorStore | None = None,
        bm25_store: BM25Store | None = None,
        reranker: Reranker | None = None,
        llm_model: str = "llama-3.3-70b-versatile",
    ):
        self._vector_store = vector_store or VectorStore()
        self._bm25_store = bm25_store or BM25Store()
        self._hybrid_retriever = HybridRetriever(
            vector_store=self._vector_store,
            bm25_store=self._bm25_store,
        )
        self._reranker = reranker or Reranker()
        self._llm = ChatGroq(
            model=llm_model,
            temperature=0,
            groq_api_key=os.getenv("GROQ_API_KEY"),
        )

    def rewrite_query(self, query: str, conversation_context: str = "") -> str:
        """Rewrite the user query using conversation context for better retrieval.

        This addresses the vocabulary mismatch problem by expanding the query
        with relevant terms from the conversation history.
        """
        if not conversation_context:
            return query

        rewrite_prompt = f"""Given the conversation context and the latest user query, 
rewrite the query to be a standalone search query that captures the full intent.
Include relevant technical terms, product names, and specific details from the context.

Conversation context:
{conversation_context}

Latest query: {query}

Rewritten search query (respond with ONLY the rewritten query, nothing else):"""

        response = self._llm.invoke(rewrite_prompt)
        return response.content.strip()

    def retrieve(
        self,
        query: str,
        conversation_context: str = "",
        category_filter: Optional[str] = None,
        top_k: int = 5,
        hybrid_k: int = 15,
    ) -> list[RetrievalResult]:
        """Execute the full RAG retrieval pipeline.

        Args:
            query: The user's query.
            conversation_context: Previous conversation for query rewriting.
            category_filter: Optional category to narrow search.
            top_k: Final number of results after re-ranking.
            hybrid_k: Number of candidates from hybrid retrieval before re-ranking.

        Returns:
            List of RetrievalResult objects, re-ranked and ready for citation.
        """
        # Step 1: Query rewriting
        search_query = self.rewrite_query(query, conversation_context)

        # Step 2: Hybrid retrieval (vector + BM25 with RRF)
        hybrid_results = self._hybrid_retriever.search(
            query=search_query,
            k=hybrid_k,
            category_filter=category_filter,
        )

        if not hybrid_results:
            return []

        # Step 3: Cross-encoder re-ranking
        reranked_results = self._reranker.rerank(
            query=search_query,
            results=hybrid_results,
            top_k=top_k,
        )

        return reranked_results

    def format_context(self, results: list[RetrievalResult]) -> str:
        """Format retrieval results as context for the LLM.

        Includes article IDs and titles for citation.
        """
        if not results:
            return "No relevant knowledge base articles found."

        context_parts = []
        for i, result in enumerate(results, 1):
            context_parts.append(
                f"[{result.article_id}: {result.article_title}]\n"
                f"Category: {result.category}\n"
                f"Content: {result.chunk_text}\n"
            )

        return "\n---\n".join(context_parts)
