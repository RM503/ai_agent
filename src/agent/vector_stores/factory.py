from __future__ import annotations

import os
from functools import lru_cache

from langchain_postgres import PGVector
from langchain_postgres.vectorstores import DistanceStrategy

from agent.ingestion.embeddings import get_embedding
from agent.vector_stores.postgres_store import PostgresVectorStore


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
