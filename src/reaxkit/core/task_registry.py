"""Backward-compatible imports for the analysis task registry."""

from reaxkit.core.registry.analysis_task_registry import (
    TASK_LABELS,
    TASK_REGISTRY,
    register_task,
    task_display_label,
)
from reaxkit.analysis.trajectory.msd_task import MSDTask

TASK_REGISTRY.setdefault("msd", MSDTask)
TASK_LABELS.setdefault("msd", "MSD")

__all__ = ["TASK_LABELS", "TASK_REGISTRY", "register_task", "task_display_label"]
