from __future__ import annotations 

from abc import ABC, abstractmethod
from typing import Any, Sequence

from langchain_core.documents import Document 

class BaseVectorStore(ABC):
    """
    Abstract base class for vector store implementations, containing specific
    methods to be used in each vector store class.
    """
    @abstractmethod 
    def add_documents(self, documents: Sequence[Document], **kwargs: Any) -> list[str]:
        """Inserts documents and returns IDs."""
        pass 

    @abstractmethod
    def similarity_search(self, query: str, k: int=5, **kwargs: Any) -> list[Document]:
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