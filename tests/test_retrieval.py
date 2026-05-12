"""Tests for the RAG retrieval pipeline."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from knowledge_base.ingest import chunk_article, load_articles
from models.knowledge import KBArticle


class TestChunking:
    """Test the article chunking logic."""

    def test_short_article_single_chunk(self):
        """A short article should produce a single chunk."""
        article = KBArticle(
            id="KB-TEST-001",
            title="Test Article",
            category="faq",
            tags=["test"],
            content="This is a short article with minimal content.",
            last_updated="2026-01-01",
            applies_to=["Starter"],
        )
        chunks = chunk_article(article)
        assert len(chunks) >= 1
        assert chunks[0].article_id == "KB-TEST-001"
        assert chunks[0].article_title == "Test Article"

    def test_long_article_multiple_chunks(self):
        """A long article should be split into multiple chunks."""
        long_content = "# Title\n\n" + "\n\n## Section {i}\n\n".join(
            [f"Content for section {i}. " * 50 for i in range(5)]
        )
        article = KBArticle(
            id="KB-TEST-002",
            title="Long Article",
            category="troubleshooting",
            tags=["test"],
            content=long_content,
            last_updated="2026-01-01",
            applies_to=["Pro"],
        )
        chunks = chunk_article(article)
        assert len(chunks) > 1

    def test_chunk_metadata_preserved(self):
        """Each chunk should preserve the article's metadata."""
        article = KBArticle(
            id="KB-TEST-003",
            title="Metadata Test",
            category="billing",
            tags=["refund", "policy"],
            content="# Refund Policy\n\nRefunds are available within 14 days.",
            last_updated="2026-02-01",
            applies_to=["Pro", "Enterprise"],
        )
        chunks = chunk_article(article)
        for chunk in chunks:
            assert chunk.article_id == "KB-TEST-003"
            assert chunk.category == "billing"
            assert "refund" in chunk.tags
            assert "Pro" in chunk.applies_to

    def test_chunk_ids_unique(self):
        """All chunk IDs should be unique."""
        article = KBArticle(
            id="KB-TEST-004",
            title="Unique IDs",
            category="faq",
            tags=["test"],
            content="# Section 1\nContent 1\n\n## Section 2\nContent 2\n\n## Section 3\nContent 3",
            last_updated="2026-01-01",
            applies_to=["Starter"],
        )
        chunks = chunk_article(article)
        chunk_ids = [c.chunk_id for c in chunks]
        assert len(chunk_ids) == len(set(chunk_ids))


class TestArticleLoading:
    """Test KB article loading."""

    def test_load_all_articles(self):
        """All 20 KB articles should load successfully."""
        articles_dir = str(Path(__file__).parent.parent / "knowledge_base" / "articles")
        articles = load_articles(articles_dir)
        assert len(articles) >= 20

    def test_article_schema_valid(self):
        """Each article should have all required fields."""
        articles_dir = str(Path(__file__).parent.parent / "knowledge_base" / "articles")
        articles = load_articles(articles_dir)
        for article in articles:
            assert article.id.startswith("KB-")
            assert len(article.title) > 0
            assert article.category in ["faq", "troubleshooting", "billing", "api_docs", "account"]
            assert len(article.content) > 0
            assert len(article.tags) > 0
            assert len(article.applies_to) > 0

    def test_article_categories_coverage(self):
        """Articles should cover all 5 required categories."""
        articles_dir = str(Path(__file__).parent.parent / "knowledge_base" / "articles")
        articles = load_articles(articles_dir)
        categories = set(a.category for a in articles)
        required = {"faq", "troubleshooting", "billing", "api_docs", "account"}
        assert required.issubset(categories), f"Missing categories: {required - categories}"
