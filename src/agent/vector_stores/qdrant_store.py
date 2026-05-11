from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore

from .base import BaseVectorStore
from agent.common.logging_config import get_logger

logger = get_logger(__name__)


class QdrantStore(BaseVectorStore):
    """Qdrant vector store implementation."""

    def __init__(self, store: QdrantVectorStore, collection_name: str):
        self.store = store
        self.collection_name = collection_name

    def add_documents(
        self,
        documents: Sequence[Document],
        ids: Sequence[str] | None = None,
        **kwargs: Any
    ) -> list[str] | None:
        """Inserts documents into Qdrant vector store and returns IDs."""
        try:
            document_ids = self.store.add_documents(list(documents), ids=ids, **kwargs)
            logger.info(f"Added {len(documents)} documents to Qdrant vector store.")
            return document_ids
        except ValueError:
            logger.exception("Invalid documents or IDs passed to Qdrant vector store.")
            raise
        except TimeoutError:
            logger.exception("Connection to Qdrant timed out.")
            raise
        except ConnectionError:
            logger.exception("Failed to connect to Qdrant.")
            raise
        except Exception:
            logger.exception("An error occurred while adding documents to Qdrant vector store.")
            raise

    def similarity_search(
        self,
        query: str,
        k: int=5,
        filter: dict | None = None,
        **kwargs: Any
    ) -> list[Document]:
        """Performs similarity search on query in Qdrant vector store."""
        results = self.store.similarity_search(query, k=k, filter=filter, **kwargs)
        logger.info(f"Performed Qdrant similarity search for query: '{query}' with k={k}.")
        return results

    def similarity_search_with_score(
            self,
            query: str,
            k: int=5,
            filter: dict | None = None,
            score_threshold: float = 0.5,
            **kwargs: Any
    ) -> list[tuple[Document, float]]:
        """Performs similarity search with score on query in Qdrant vector store."""
        results = self.store.similarity_search_with_score(
            query=query,
            k=k,
            filter=filter,
            score_threshold=score_threshold,
            **kwargs
        )
        logger.info(
            f"Performed Qdrant similarity search with score for query: '{query}' wih k={k}."
        )
        return results

    def delete_documents(self, ids: Sequence[str], **kwargs: Any) -> bool | None:
        """Delete documents by ID from Qdrant vector store."""
        try:
            result = self.store.delete(ids=list(ids), **kwargs)
            logger.info(f"Deleted documents with IDs: {ids} from Qdrant vector store.")
            return result
        except ValueError:
            logger.exception("Invalid IDs passed for Qdrant deletion.")
            raise
        except TimeoutError:
            logger.exception("Connection to Qdrant timed out during deletion.")
            raise
        except ConnectionError:
            logger.exception("Failed to connect to Qdrant during deletion.")
            raise
        except Exception:
            logger.exception("An error occurred while deleting documents from Qdrant vector store.")
            raise

    def as_retriever(self, **kwargs: Any) -> Any:
        """Return a retriever-like object for Qdrant vector store."""
        retriever = self.store.as_retriever(**kwargs)
        logger.info("Retriever object created for Qdrant vector store.")
        return retriever
