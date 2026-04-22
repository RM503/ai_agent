"""
Text-splitter class for RAG pipeline
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from langchain_core.documents import Document

from .registry import get_splitter

@dataclass(slots=True)
class TextSplitter:
    """
    Text-splitter class

    Attributes:
        splitter (Literal["character", "recursive", "sentence"]): the type of splitter to use; defaults to 'recursive'
        chunk_size (int): size of text chunks; defaults to 1000
        chunk_overlap (int): amount of overlap between chunks
        splitter_kwargs (dict[str, Any]): other key-word arguments
    """
    splitter: Literal["character", "recursive", "sentence"] = "recursive"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    splitter_kwargs: dict[str, Any] = field(default_factory=dict)

    def create_splitter(self, **overrides: Any) -> Any:
        """Creates splitter object"""
        factory = get_splitter(self.splitter)
        extra = {**self.splitter_kwargs, **overrides}
        return factory(self.chunk_size, self.chunk_overlap, extra)

    def get_splits(self, data: str | list[Document], **overrides: Any) -> list[Any]:
        """Generates splits"""
        splitter = self.create_splitter(**overrides)

        if isinstance(data, str):
            # If data is a single text blob 
            return splitter.split_text(data)
        return splitter.split_documents(data)