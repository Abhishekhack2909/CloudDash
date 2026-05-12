"""Knowledge base ingestion script.

Reads KB articles from JSON files, chunks them, generates embeddings,
and indexes into ChromaDB + builds a BM25 index.
"""

import json
import os
import pickle
import re
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.knowledge import KBArticle, KBChunk


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


def ingest(articles_dir: str | None = None, db_dir: str | None = None):
    """Main ingestion pipeline: load → chunk → embed → index."""
    from dotenv import load_dotenv
    load_dotenv()

    project_root = Path(__file__).parent.parent
    if articles_dir is None:
        articles_dir = str(Path(__file__).parent / "articles")
    if db_dir is None:
        db_dir = str(project_root / "chroma_db")

    print("=" * 60)
    print("CloudDash KB Ingestion Pipeline")
    print("=" * 60)

    # Step 1: Load articles
    print("\n[1/4] Loading articles...")
    articles = load_articles(articles_dir)
    print(f"  Loaded {len(articles)} articles")

    # Step 2: Chunk articles
    print("\n[2/4] Chunking articles...")
    all_chunks: list[KBChunk] = []
    for article in articles:
        chunks = chunk_article(article)
        all_chunks.extend(chunks)
        print(f"  {article.id}: {article.title} -> {len(chunks)} chunks")
    print(f"  Total chunks: {len(all_chunks)}")

    # Step 3: Embed and index into ChromaDB
    print("\n[3/4] Embedding and indexing into ChromaDB...")
    print("  Using local sentence-transformers model (free, no API calls)...")
    import chromadb
    from chromadb.utils import embedding_functions

    # Use free local embeddings (sentence-transformers) instead of OpenAI
    # This avoids API costs and works offline
    st_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2",
    )

    client = chromadb.PersistentClient(path=db_dir)

    # Delete existing collection if it exists
    try:
        client.delete_collection("clouddash_kb")
    except Exception:
        pass

    collection = client.create_collection(
        name="clouddash_kb",
        embedding_function=st_ef,
        metadata={"description": "CloudDash Knowledge Base"},
    )

    # Add chunks in batches
    batch_size = 50
    for i in range(0, len(all_chunks), batch_size):
        batch = all_chunks[i:i + batch_size]
        collection.add(
            ids=[chunk.chunk_id for chunk in batch],
            documents=[chunk.content for chunk in batch],
            metadatas=[
                {
                    "article_id": chunk.article_id,
                    "article_title": chunk.article_title,
                    "category": chunk.category,
                    "tags": ",".join(chunk.tags),
                    "chunk_index": chunk.chunk_index,
                    "applies_to": ",".join(chunk.applies_to),
                }
                for chunk in batch
            ],
        )
        print(f"  Indexed batch {i // batch_size + 1}/{(len(all_chunks) + batch_size - 1) // batch_size}")

    print(f"  ChromaDB collection '{collection.name}' has {collection.count()} documents")

    # Step 4: Build BM25 index
    print("\n[4/4] Building BM25 index...")
    from rank_bm25 import BM25Okapi

    # Tokenize chunks for BM25
    tokenized_chunks = [chunk.content.lower().split() for chunk in all_chunks]
    bm25_index = BM25Okapi(tokenized_chunks)

    # Save BM25 index and chunk metadata
    bm25_data = {
        "index": bm25_index,
        "chunks": [chunk.model_dump() for chunk in all_chunks],
    }
    bm25_path = os.path.join(db_dir, "bm25_index.pkl")
    os.makedirs(db_dir, exist_ok=True)
    with open(bm25_path, "wb") as f:
        pickle.dump(bm25_data, f)
    print(f"  BM25 index saved to {bm25_path}")

    print("\n" + "=" * 60)
    print("Ingestion complete!")
    print(f"  Articles: {len(articles)}")
    print(f"  Chunks: {len(all_chunks)}")
    print(f"  ChromaDB: {db_dir}")
    print(f"  BM25: {bm25_path}")
    print("=" * 60)


if __name__ == "__main__":
    ingest()
