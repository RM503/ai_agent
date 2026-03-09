from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Optional

Document = Any
LoaderFactory = Callable[[Path, dict[str, Any]], Document]

_LOADERS_BY_EXT: dict[str, LoaderFactory] = {}
_DEFAULT_LOADER: Optional[LoaderFactory] = None

def _norm_ext(ext: str) -> str:
    ext = ext.strip().lower()
    if not ext:
        raise ValueError("Empty extension")
    return ext if ext.startswith(".") else f".{ext}"

def register_loader(*exts: str):
    """The decorator registers a loader for one or more extensions."""
    def decorator(factory: LoaderFactory) -> LoaderFactory:
        for ext in exts:
            if ext in _LOADERS_BY_EXT:
                raise ValueError(f"Loader already registered: {ext}")
            _LOADERS_BY_EXT[ext] = factory
        return factory
    return decorator

def register_default_loader(factory: LoaderFactory) -> LoaderFactory:
    global _DEFAULT_LOADER
    _DEFAULT_LOADER = factory
    return factory

def available_extensions() -> list[str]:
    return sorted(_LOADERS_BY_EXT)

def load_documents(path: str | Path, **kwargs: Any) -> list[Document]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    factory = _LOADERS_BY_EXT.get(path.suffix.lower())
    if factory is None:
        if _DEFAULT_LOADER is None:
            raise NotImplementedError(f"No loader for '{path.suffix}'. Available: {available_extensions()}")
        factory = _DEFAULT_LOADER

    return factory[path, kwargs]