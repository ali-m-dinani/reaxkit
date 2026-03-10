"""Plugin registry for analysis tasks."""

from __future__ import annotations

from typing import Callable

TASK_REGISTRY: dict[str, type] = {}


def register_task(name: str) -> Callable:
    """Register an analysis task class under a CLI-visible name."""

    def wrapper(cls):
        TASK_REGISTRY[name] = cls
        setattr(cls, "_reaxkit_task_name", str(name))
        if "recommended_presentations" not in cls.__dict__:
            task_name = str(name)

            def _recommended_presentations(_result, payload):
                from reaxkit.analysis.base import default_recommended_presentations

                return default_recommended_presentations(payload, task_name=task_name)

            cls.recommended_presentations = staticmethod(_recommended_presentations)
        return cls

    return wrapper
