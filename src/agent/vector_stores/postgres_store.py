from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from langchain_core.documents import Document
from langchain_core.vectorstores import PostgresVectorStore as PGVectorStore

from .base import BaseVectorStore
from src.agent.common.logging_config import get_logger

logger = get_logger(__name__)


class PostgresVectorStore(BaseVectorStore):
    """PostgreSQL vector store implementation."""
    def __init__(self, store: PGVectorStore, table_name: str):
        self.store = store
        self.table_name = table_name

    def add_documents(
        self,
        documents: Sequence[Document],
        ids: Sequence[str] | None = None,
        **kwargs: Any
    ) -> list[str] | None:
        """Inserts documents into PostgreSQL vector store and returns IDs."""
        try:
            self.store.add_documents(documents, ids=ids, **kwargs)
            logger.info(f"Added {len(documents)} documents to PostgreSQL vector store.")
            return ids if ids is not None else [str(i) for i in range(len(documents))]
        except ValueError:
            logger.exception("Invalid documents or IDs passed to vector store.")
            raise
        except TimeoutError:
            logger.exception("Connection to PostgreSQL timed out.")
            raise
        except ConnectionError:
            logger.exception("Failed to connect to PostgreSQL.")
            raise
        except Exception:
            logger.exception("An error occurred while adding documents to PostgreSQL vector store.")
            raise

    def similarity_search(
        self,
        query: str,
        k: int=5,
        filter: dict | None = None,
        **kwargs: Any
    ) -> list[Document]:
        """Performs similarity search on query in PostgreSQL vector store."""
        results = self.store.similarity_search(query, k=k, filter=filter, **kwargs)
        logger.info(f"Performed similarity search for query: '{query}' with k={k}.")
        return results

    def delete_documents(self, ids: Sequence[str], **kwargs: Any) -> bool | None:
        """Delete documents by ID from PostgreSQL vector store."""
        try:
            self.store.delete_documents(ids, **kwargs)
            logger.info(f"Deleted documents with IDs: {ids} from PostgreSQL vector store.")
            return True
        except ValueError:
            logger.exception("Invalid IDs passed for deletion.")
            raise
        except TimeoutError:
            logger.exception("Connection to PostgreSQL timed out during deletion.")
            raise
        except ConnectionError:
            logger.exception("Failed to connect to PostgreSQL during deletion.")
            raise
        except Exception:
            logger.exception("An error occurred while deleting documents from PostgreSQL vector store.")
            raise

    def as_retriever(self, **kwargs: Any) -> Any:
        """Return a retriever-like object for PostgreSQL vector store."""
        retriever = self.store.as_retriever(**kwargs)
        logger.info("Retriever object created for PostgreSQL vector store.")
        return retriever
