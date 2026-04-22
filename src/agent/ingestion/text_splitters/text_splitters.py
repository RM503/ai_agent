"""
Text-splitter class for RAG pipeline
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

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
        factory = get_splitter(self.splitter)
        extra = {**self.splitter_kwargs, **overrides}
        return factory(self.chunk_size, self.chunk_overlap, extra)

    def get_splits(self, text: str, **overrides: Any) -> list[Any]:
        return self.create_splitter(**overrides).split_text(text)