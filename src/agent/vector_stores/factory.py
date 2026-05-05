"""
This module provides factory functions to create and cache vector store instances.
"""

from __future__ import annotations

import os
from functools import lru_cache

from langchain_postgres import PGVector
from langchain_postgres.vectorstores import DistanceStrategy
from langchain_qdrant import FastEmbedSparse, QdrantVectorStore, RetrievalMode

from agent.ingestion.embeddings import get_embedding
from agent.vector_stores.postgres_store import PostgresVectorStore
from agent.vector_stores.qdrant_store import QdrantStore


@lru_cache(maxsize=1)
def create_postgres_vector_store() -> PostgresVectorStore:
    connection = os.getenv("POSTGRESQL_CONNECTION")
    if not connection:
        raise RuntimeError("POSTGRESQL_CONNECTION is not set.")

    collection_name = os.getenv("VECTOR_COLLECTION_NAME", "document_embeddings")

    store = PGVector(
        embeddings=get_embedding(),
        connection=connection,
        collection_name=collection_name,
        embedding_length=1536,
        distance_strategy=DistanceStrategy.COSINE,
        use_jsonb=True,
        create_extension=True
    )

    return PostgresVectorStore(store=store, table_name=collection_name)


def _get_qdrant_retrieval_mode() -> RetrievalMode:
    raw_mode = os.getenv("QDRANT_RETRIEVAL_MODE", "hybrid").lower()
    try:
        return RetrievalMode(raw_mode)
    except ValueError as exc:
        valid_modes = ", ".join(mode.value for mode in RetrievalMode)
        raise RuntimeError(
            f"Invalid QDRANT_RETRIEVAL_MODE '{raw_mode}'. Expected one of: {valid_modes}."
        ) from exc


@lru_cache(maxsize=1)
def create_qdrant_vector_store() -> QdrantStore:
    collection_name = os.getenv("QDRANT_COLLECTION_NAME", "document_embeddings")
    retrieval_mode = _get_qdrant_retrieval_mode()

    url = os.getenv("QDRANT_CLUSTER_ENDPOINT")
    api_key = os.getenv("QDRANT_API_KEY")

    sparse_embedding = None
    if retrieval_mode in {RetrievalMode.SPARSE, RetrievalMode.HYBRID}:
        sparse_embedding = FastEmbedSparse(
            model_name=os.getenv("QDRANT_SPARSE_MODEL", "Qdrant/bm25")
        )

    store = QdrantVectorStore.construct_instance(
        embedding=get_embedding() if retrieval_mode != RetrievalMode.SPARSE else None,
        sparse_embedding=sparse_embedding,
        retrieval_mode=retrieval_mode,
        collection_name=collection_name,
        client_options={
            "url": url,
            "api_key": api_key,
            "port": int(os.getenv("QDRANT_PORT", "6333")),
            "grpc_port": int(os.getenv("QDRANT_GRPC_PORT", "6334")),
            "prefer_grpc": os.getenv("QDRANT_PREFER_GRPC", "false").lower() == "true",
        },
    )

    return QdrantStore(store=store, collection_name=collection_name)
