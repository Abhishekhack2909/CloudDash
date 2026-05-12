"""Knowledge base data models."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class KBArticle(BaseModel):
    """A knowledge base article."""

    id: str
    title: str
    category: str  # faq, troubleshooting, billing, api_docs, account
    tags: list[str] = Field(default_factory=list)
    content: str
    last_updated: str  # ISO date string
    applies_to: list[str] = Field(default_factory=list)  # Plan types


class KBChunk(BaseModel):
    """A chunk of a KB article for embedding and retrieval."""

    chunk_id: str
    article_id: str
    article_title: str
    category: str
    tags: list[str] = Field(default_factory=list)
    content: str
    chunk_index: int = 0
    applies_to: list[str] = Field(default_factory=list)


class RetrievalResult(BaseModel):
    """A result from the RAG retrieval pipeline."""

    article_id: str
    article_title: str
    category: str
    chunk_text: str
    relevance_score: float = 0.0
    retrieval_method: str = "hybrid"  # vector, bm25, hybrid
    metadata: dict = Field(default_factory=dict)
