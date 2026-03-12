"""Plugin registry for analysis tasks."""

from __future__ import annotations

from typing import Callable

TASK_REGISTRY: dict[str, type] = {}
TASK_LABELS: dict[str, str] = {}


def task_display_label(name: str) -> str:
    """Convert a canonical task name into a human-readable label."""
    tokens = [part for part in str(name).strip().split("_") if part]
    if not tokens:
        return "Task"

    acronym_tokens = {"msd", "rdf", "eos"}

    words: list[str] = []
    for token in tokens:
        low = token.lower()
        if low in acronym_tokens:
            words.append(low.upper())
        elif low == "reaxff":
            words.append("ReaxFF")
        else:
            words.append(low.capitalize())
    return " ".join(words)


def register_task(name: str, *, label: str | None = None) -> Callable:
    """Register an analysis task class under a CLI-visible name."""

    def wrapper(cls):
        canonical_name = str(name).strip().lower()
        if not canonical_name:
            raise ValueError("Task name cannot be empty.")
        display_label = str(label).strip() if label is not None else task_display_label(canonical_name)
        if not display_label:
            display_label = task_display_label(canonical_name)

        TASK_REGISTRY[canonical_name] = cls
        TASK_LABELS[canonical_name] = display_label
        setattr(cls, "_reaxkit_task_name", canonical_name)
        setattr(cls, "_reaxkit_task_label", display_label)
        if "recommended_presentations" not in cls.__dict__:
            task_name = canonical_name

            def _recommended_presentations(_result, payload):
                from reaxkit.analysis.base import default_recommended_presentations

                return default_recommended_presentations(payload, task_name=task_name)

            cls.recommended_presentations = staticmethod(_recommended_presentations)
        return cls

    return wrapper
