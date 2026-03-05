from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .registry import get

@dataclass(slots=True)
class TextSplitter:
    splitter: str = "recursive"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    splitter_kwargs: dict[str, Any] = field(default_factory=dict)

    def create_splitter(self, **overrides: Any) -> Any:
        factory = get(self.splitter)
        extra = {**self.splitter_kwargs, **overrides}
        return factory(self.chunk_size, self.chunk_overlap, extra)

    def get_splits(self, text: str, **overrides: Any) -> list[Any]:
        return self.create_splitter(**overrides).split_text(text)