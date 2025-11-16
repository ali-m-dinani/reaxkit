"""the main abstract handler for parsing and cleaning data"""
from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any
import pandas as pd

class FileHandler(ABC):
    """
    Minimal base for file handlers:
      - hold path
      - lazy parse-once
      - expose a canonical DataFrame and lightweight metadata
    """

    def __init__(self, file_path: str | Path):
        self.path = Path(file_path)
        self._parsed = False
        self._df: pd.DataFrame | None = None
        self._meta: dict[str, Any] = {}

    # ---- public API
    def parse(self) -> None:
        if not self._parsed:
            df, meta = self._parse()
            self._df = df
            self._meta = meta or {}
            self._parsed = True

    def dataframe(self) -> pd.DataFrame:
        if not self._parsed:
            self.parse()
        assert self._df is not None
        return self._df

    def metadata(self) -> dict[str, Any]:
        if not self._parsed:
            self.parse()
        return dict(self._meta)

    # ---- subclasses must implement
    @abstractmethod
    def _parse(self) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Read+parse file and return (df, metadata)."""
        ...
