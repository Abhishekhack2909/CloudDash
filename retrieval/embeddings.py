"""Embedding generation wrapper using OpenAI."""

import os
from functools import lru_cache

from langchain_openai import OpenAIEmbeddings


@lru_cache(maxsize=1)
def get_embedding_model(model_name: str = "text-embedding-3-small") -> OpenAIEmbeddings:
    """Get a cached OpenAI embedding model instance."""
    return OpenAIEmbeddings(
        model=model_name,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
    )


def embed_query(query: str, model_name: str = "text-embedding-3-small") -> list[float]:
    """Generate an embedding for a single query string."""
    model = get_embedding_model(model_name)
    return model.embed_query(query)


def embed_documents(
    documents: list[str], model_name: str = "text-embedding-3-small"
) -> list[list[float]]:
    """Generate embeddings for a list of documents."""
    model = get_embedding_model(model_name)
    return model.embed_documents(documents)
