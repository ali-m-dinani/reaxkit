"""Generic engine adapter API."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class EngineAdapter(ABC):
    """Engine adapter interface for detection + typed data loading."""

    name: str = "base"

    @abstractmethod
    def detect(self, path: str | Path) -> float:
        """Return confidence score [0, 1]."""

    @abstractmethod
    def load(self, data_type, args: dict):
        """Load requested domain data type from engine-specific sources."""
