"""Registry + resolver for engine adapters."""

from __future__ import annotations

from pathlib import Path

ENGINE_REGISTRY: dict[str, type] = {}


def register_engine(name: str):
    """Decorator to register an engine adapter class."""

    def wrapper(cls):
        ENGINE_REGISTRY[name] = cls
        cls.name = name
        return cls

    return wrapper


def resolve_engine(path: str, engine: str | None = None):
    """Resolve adapter by explicit engine or confidence-based auto-detection."""
    if engine:
        if engine not in ENGINE_REGISTRY:
            raise ValueError(f"Unknown engine '{engine}'. Available: {sorted(ENGINE_REGISTRY)}")
        return ENGINE_REGISTRY[engine]()

    target = Path(path)
    best_name: str | None = None
    best_score = -1.0

    for name, cls in ENGINE_REGISTRY.items():
        score = cls().detect(target)
        if score > best_score:
            best_score = score
            best_name = name

    if best_name is None or best_score <= 0:
        raise ValueError("Could not detect engine. Pass --engine explicitly.")

    return ENGINE_REGISTRY[best_name]()
