"""Base interfaces for analysis layer."""

from __future__ import annotations

from abc import ABC, abstractmethod


class AnalysisTask(ABC):
    """Abstract analysis task with declarative data requirement."""

    required_data = None

    @abstractmethod
    def run(self, data, request, reporter=None):
        """Run scientific analysis on normalized domain data."""
