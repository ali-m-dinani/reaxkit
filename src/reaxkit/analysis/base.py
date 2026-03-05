"""Base interfaces for analysis layer."""

from __future__ import annotations

from abc import ABC, abstractmethod
from functools import wraps


class AnalysisTask(ABC):
    """Abstract analysis task with declarative data requirement."""

    required_data = None

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        run_fn = cls.__dict__.get("run")
        if run_fn is None or getattr(run_fn, "_reaxkit_validated_run", False):
            return

        @wraps(run_fn)
        def _validated_run(self, data, request, *args, **kwargs):
            from reaxkit.analysis.validation import validate_task_inputs

            validate_task_inputs(self, data, request)
            return run_fn(self, data, request, *args, **kwargs)

        _validated_run._reaxkit_validated_run = True
        cls.run = _validated_run

    @abstractmethod
    def run(self, data, request, reporter=None):
        """Run scientific analysis on normalized domain data."""
