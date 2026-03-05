from __future__ import annotations

from typing import Any, Callable

SplitterFactory = Callable[[int, int, dict[str, Any]], Any]
_FACTORIES: dict[str, SplitterFactory] = {}

def register(name: str) -> Any:
    def decorator(factory: SplitterFactory) -> SplitterFactory:
        if name in _FACTORIES:
            raise ValueError(f"Duplicate splitter name: {name}")
        _FACTORIES[name] = factory
        return factory
    return decorator

def get(name: str) -> SplitterFactory:
    try:
        return _FACTORIES[name]
    except KeyError as e:
        raise NotImplementedError(f"Unknown splitter '{name}'. Available {sorted(_FACTORIES)}") from e

def available() -> list[str]:
    return sorted(_FACTORIES)