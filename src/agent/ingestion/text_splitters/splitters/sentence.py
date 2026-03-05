# Implements SentenceTransformersTokenTextSplitter
from __future__ import annotations

from typing import Any

from langchain_text_splitters import SentenceTransformersTokenTextSplitter

from ..registry import register

@register("sentence")
def factory(chunk_size: int, chunk_overlap: int, extra: dict[str, Any]) -> SentenceTransformersTokenTextSplitter:
    return SentenceTransformersTokenTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        **extra
    )