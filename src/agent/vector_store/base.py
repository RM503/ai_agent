from __future__ import annotations 

from abc import ABC, abstractmethod

from langchain_core.documents import Document 

class BaseVectorStore(ABC):
    @abstractmethod 
    def add_documents(self, documents: list[Document]):
        pass 

    @abstractmethod
    def similarity_search(self, query: str, k: int=5) -> list[Document]:
        pass