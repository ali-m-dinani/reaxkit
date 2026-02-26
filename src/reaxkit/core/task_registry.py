"""Plugin registry for analysis tasks."""

from __future__ import annotations

from typing import Callable

TASK_REGISTRY: dict[str, type] = {}


def register_task(name: str) -> Callable:
    """Register an analysis task class under a CLI-visible name."""

    def wrapper(cls):
        TASK_REGISTRY[name] = cls
        return cls

    return wrapper
