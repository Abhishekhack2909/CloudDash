"""Incremental knowledge base ingestion pipeline.

Features:
- Content-hash tracking (SHA-256) per article.
- Incremental indexing: skips unchanged articles, upserts changed ones, deletes removed ones.
- Synchronizes both ChromaDB vector store and rank-bm25 index.
- CLI support: python -m knowledge_base.ingest [--force]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.knowledge import KBArticle, KBChunk
from observability.logger import get_logger

logger = get_logger(__name__)


def compute_article_hash(article: KBArticle) -> str:
    """Compute SHA-256 content hash of an article."""
    payload = f"{article.title}:{article.category}:{','.join(sorted(article.tags))}:{article.content}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_articles(articles_dir: str) -> list[KBArticle]:
    """Load all KB articles from JSON files in the given directory."""
    articles = []
    articles_path = Path(articles_dir)

    for filepath in sorted(articles_path.glob("KB-*.json")):
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
            articles.append(KBArticle(**data))

    return articles


def chunk_article(article: KBArticle, max_chunk_size: int = 500, overlap: int = 50) -> list[KBChunk]:
    """Split an article into chunks by sections, with fallback to size-based splitting.

    Strategy:
    1. Split by markdown headers (##, ###) to preserve semantic boundaries.
    2. If a section exceeds max_chunk_size characters, split it further.
    3. Each chunk retains the article title as context prefix.
    """
    content = article.content
    chunks = []

    # Split by markdown headers
    sections = re.split(r'\n(?=#{1,3}\s)', content)

    chunk_index = 0
    for section in sections:
        section = section.strip()
        if not section:
            continue

        # If section is small enough, keep as one chunk
        if len(section) <= max_chunk_size:
            chunks.append(KBChunk(
                chunk_id=f"{article.id}-chunk-{chunk_index}",
                article_id=article.id,
                article_title=article.title,
                category=article.category,
                tags=article.tags,
                content=section,
                chunk_index=chunk_index,
                applies_to=article.applies_to,
            ))
            chunk_index += 1
        else:
            # Split large sections by paragraphs or fixed size
            paragraphs = section.split('\n\n')
            current_chunk = ""

            for para in paragraphs:
                if len(current_chunk) + len(para) <= max_chunk_size:
                    current_chunk += ("\n\n" + para if current_chunk else para)
                else:
                    if current_chunk:
                        chunks.append(KBChunk(
                            chunk_id=f"{article.id}-chunk-{chunk_index}",
                            article_id=article.id,
                            article_title=article.title,
                            category=article.category,
                            tags=article.tags,
                            content=current_chunk,
                            chunk_index=chunk_index,
                            applies_to=article.applies_to,
                        ))
                        chunk_index += 1
                    current_chunk = para

            if current_chunk:
                chunks.append(KBChunk(
                    chunk_id=f"{article.id}-chunk-{chunk_index}",
                    article_id=article.id,
                    article_title=article.title,
                    category=article.category,
                    tags=article.tags,
                    content=current_chunk,
                    chunk_index=chunk_index,
                    applies_to=article.applies_to,
                ))
                chunk_index += 1

    return chunks


class KBIngestionPipeline:
    """Manages incremental ingestion into ChromaDB and BM25."""

    def __init__(self, articles_dir: str | None = None, db_dir: str | None = None):
        project_root = Path(__file__).parent.parent
        self.articles_dir = articles_dir or str(Path(__file__).parent / "articles")
        self.db_dir = db_dir or str(project_root / "chroma_db")
        self.manifest_path = os.path.join(self.db_dir, "ingestion_manifest.json")

    def _load_manifest(self) -> dict[str, dict[str, Any]]:
        """Load manifest mapping article_id -> {content_hash, chunk_ids, title}."""
        if os.path.exists(self.manifest_path):
            try:
                with open(self.manifest_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning("failed_to_load_manifest", error=str(e))
        return {}

    def _save_manifest(self, manifest: dict[str, dict[str, Any]]) -> None:
        """Save ingestion manifest to disk."""
        os.makedirs(self.db_dir, exist_ok=True)
        with open(self.manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

    def run(self, force: bool = False) -> dict[str, Any]:
        """Execute incremental or full ingestion.

        Args:
            force: If True, completely drops and re-indexes all articles.

        Returns:
            Ingestion summary dictionary.
        """
        import chromadb
        from chromadb.utils import embedding_functions
        from rank_bm25 import BM25Okapi

        print("=" * 60)
        print("CloudDash KB Incremental Ingestion Pipeline")
        print("=" * 60)

        # Step 1: Load active articles
        articles = load_articles(self.articles_dir)
        article_map = {a.id: a for a in articles}
        print(f"Loaded {len(articles)} articles from {self.articles_dir}")

        manifest = self._load_manifest() if not force else {}

        # Categorize articles
        added_ids = []
        updated_ids = []
        unchanged_ids = []
        deleted_ids = [aid for aid in manifest if aid not in article_map]

        for a in articles:
            chash = compute_article_hash(a)
            if a.id not in manifest:
                added_ids.append(a.id)
            elif manifest[a.id].get("content_hash") != chash:
                updated_ids.append(a.id)
            else:
                unchanged_ids.append(a.id)

        print(f"Changes detected: +{len(added_ids)} added, ~{len(updated_ids)} updated, -{len(deleted_ids)} deleted, ={len(unchanged_ids)} unchanged")

        # ChromaDB setup
        os.makedirs(self.db_dir, exist_ok=True)
        st_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2",
        )
        client = chromadb.PersistentClient(path=self.db_dir)

        if force:
            try:
                client.delete_collection("clouddash_kb")
            except Exception:
                pass
            collection = client.create_collection(
                name="clouddash_kb",
                embedding_function=st_ef,
                metadata={"description": "CloudDash Knowledge Base"},
            )
            manifest = {}
        else:
            try:
                collection = client.get_collection(
                    name="clouddash_kb",
                    embedding_function=st_ef,
                )
            except Exception:
                collection = client.create_collection(
                    name="clouddash_kb",
                    embedding_function=st_ef,
                    metadata={"description": "CloudDash Knowledge Base"},
                )

        # Handle deletions
        for aid in deleted_ids:
            old_chunk_ids = manifest[aid].get("chunk_ids", [])
            if old_chunk_ids:
                try:
                    collection.delete(ids=old_chunk_ids)
                except Exception as e:
                    logger.warning("chroma_delete_failed", article_id=aid, error=str(e))
            del manifest[aid]

        # Handle updates (delete old chunks first)
        for aid in updated_ids:
            old_chunk_ids = manifest[aid].get("chunk_ids", [])
            if old_chunk_ids:
                try:
                    collection.delete(ids=old_chunk_ids)
                except Exception as e:
                    logger.warning("chroma_delete_failed", article_id=aid, error=str(e))

        # Chunk and index added + updated articles
        to_index = added_ids + updated_ids
        chunks_to_add: list[KBChunk] = []

        for aid in to_index:
            article = article_map[aid]
            chunks = chunk_article(article)
            chunks_to_add.extend(chunks)
            manifest[aid] = {
                "title": article.title,
                "category": article.category,
                "content_hash": compute_article_hash(article),
                "chunk_ids": [c.chunk_id for c in chunks],
                "last_updated": article.last_updated,
            }

        if chunks_to_add:
            batch_size = 50
            for i in range(0, len(chunks_to_add), batch_size):
                batch = chunks_to_add[i : i + batch_size]
                collection.upsert(
                    ids=[c.chunk_id for c in batch],
                    documents=[c.content for c in batch],
                    metadatas=[
                        {
                            "article_id": c.article_id,
                            "article_title": c.article_title,
                            "category": c.category,
                            "tags": ",".join(c.tags),
                            "chunk_index": c.chunk_index,
                            "applies_to": ",".join(c.applies_to),
                        }
                        for c in batch
                    ],
                )
            print(f"Indexed {len(chunks_to_add)} new/updated chunks into ChromaDB.")

        # Rebuild BM25 index if any changes or if BM25 file doesn't exist
        bm25_path = os.path.join(self.db_dir, "bm25_index.pkl")
        has_changes = (len(added_ids) + len(updated_ids) + len(deleted_ids) > 0)

        all_current_chunks: list[KBChunk] = []
        for aid in sorted(article_map.keys()):
            all_current_chunks.extend(chunk_article(article_map[aid]))

        if has_changes or not os.path.exists(bm25_path) or force:
            tokenized_chunks = [c.content.lower().split() for c in all_current_chunks]
            bm25_index = BM25Okapi(tokenized_chunks)
            bm25_data = {
                "index": bm25_index,
                "chunks": [c.model_dump() for c in all_current_chunks],
            }
            with open(bm25_path, "wb") as f:
                pickle.dump(bm25_data, f)
            print(f"Rebuilt BM25 index with {len(all_current_chunks)} total chunks.")

        # Persist updated manifest
        self._save_manifest(manifest)

        summary = {
            "status": "success",
            "added": len(added_ids),
            "updated": len(updated_ids),
            "deleted": len(deleted_ids),
            "unchanged": len(unchanged_ids),
            "total_articles": len(articles),
            "total_chunks": len(all_current_chunks),
            "chroma_docs_count": collection.count(),
        }

        logger.info("kb_ingestion_summary", **summary)

        print("\n" + "=" * 60)
        print("Ingestion Summary:")
        print(f"  Added:     {summary['added']}")
        print(f"  Updated:   {summary['updated']}")
        print(f"  Deleted:   {summary['deleted']}")
        print(f"  Unchanged: {summary['unchanged']}")
        print(f"  Total Articles: {summary['total_articles']}")
        print(f"  Total Chunks:   {summary['total_chunks']}")
        print("=" * 60)

        return summary


def ingest(articles_dir: str | None = None, db_dir: str | None = None, force: bool = False) -> dict[str, Any]:
    """Convenience function for running ingestion."""
    pipeline = KBIngestionPipeline(articles_dir=articles_dir, db_dir=db_dir)
    return pipeline.run(force=force)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CloudDash KB Ingestion")
    parser.add_argument("--force", action="store_true", help="Force complete re-indexing")
    parser.add_argument("--incremental", action="store_true", help="Run incremental update (default)")
    parser.add_argument("--articles-dir", type=str, default=None, help="Custom articles directory")
    parser.add_argument("--db-dir", type=str, default=None, help="Custom ChromaDB directory")

    args = parser.parse_args()
    ingest(articles_dir=args.articles_dir, db_dir=args.db_dir, force=args.force)
