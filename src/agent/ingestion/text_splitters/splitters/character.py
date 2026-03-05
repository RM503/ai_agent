# Implements CharacterTextSplitter
from __future__ import annotations

from typing import Any

from langchain_text_splitters import CharacterTextSplitter

from ..registry import register

@register("character")
def factory(chunk_size: int, chunk_overlap: int, extra: dict[str, Any]) -> CharacterTextSplitter:
    return CharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        **extra
    )