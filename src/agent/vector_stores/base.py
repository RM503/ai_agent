"""
This module defines the abstract base class `BaseVectorStore` for vector store
implementations. It defines the interface for adding documents, performing similarity
search, deleting documents and returing a retriever-like object. Each vector store
implementation should inherit from this base class and implement the defined methods.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

from langchain_core.documents import Document

class BaseVectorStore(ABC):
    """
    Abstract base class for vector store implementations, containing specific
    methods to be used in each vector store class.
    """
    @abstractmethod
    def add_documents(
        self,
        documents: Sequence[Document],
        ids: Sequence[str] | None = None,
        **kwargs: Any
    ) -> list[str]:
        """Inserts documents and returns IDs."""
        pass

    @abstractmethod
    def similarity_search(
        self,
        query: str,
        k: int=5,
        filter: dict | None = None,
        **kwargs: Any
    ) -> list[Document]:
        """Performs similarity search onb query."""
        pass

    @abstractmethod
    def delete_documents(self, ids: Sequence[str], **kwargs: Any) -> bool | None:
        """Delete document(s) by ID."""
        pass

    @abstractmethod
    def as_retriever(self, **kwargs: Any) -> Any:
        """Return a retriever-like object."""
        pass
