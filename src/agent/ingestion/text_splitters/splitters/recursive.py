# Implements RecursiveCharacterTextSplitter
from __future__ import annotations

from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter

from ..registry import register_splitter

@register_splitter("recursive")
def factory(chunk_size: int, chunk_overlap: int, extra: dict[str, Any]) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        **extra
    )
