"""Reusable bounded reducers for frame-pipeline results."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from reaxkit.core.runtime.artifacts import BufferedTableSink


class TableAccumulator:
    """Buffer compact rows or flush optional detail rows to an artifact sink."""

    def __init__(
        self,
        *,
        retain: bool = True,
        sink: BufferedTableSink | None = None,
        flush_rows: int = 2048,
    ) -> None:
        self.retain = retain
        self.sink = sink
        self.flush_rows = max(1, int(flush_rows))
        self._retained: list[dict[str, Any]] = []
        self._buffer: list[dict[str, Any]] = []

    def add(self, rows: Iterable[Mapping[str, Any]]) -> None:
        batch = [dict(row) for row in rows]
        if not batch:
            return
        if self.retain:
            self._retained.extend(batch)
        if self.sink is not None:
            self._buffer.extend(batch)
            if len(self._buffer) >= self.flush_rows:
                self.flush()

    def flush(self) -> None:
        if self.sink is not None and self._buffer:
            self.sink.append(self._buffer)
            self._buffer.clear()

    def finalize(self) -> pd.DataFrame:
        self.flush()
        return pd.DataFrame(self._retained)


@dataclass(slots=True)
class CountSumReducer:
    """Accumulate per-bin counts and finite-value sums with NumPy."""

    size: int
    counts: np.ndarray = field(init=False)
    sums: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.counts = np.zeros(int(self.size), dtype=np.int64)
        self.sums = np.zeros(int(self.size), dtype=float)

    def add(self, bins: np.ndarray, values: np.ndarray) -> None:
        indices = np.asarray(bins, dtype=int).reshape(-1)
        numeric = np.asarray(values, dtype=float).reshape(-1)
        valid = (
            (indices >= 0)
            & (indices < int(self.size))
            & np.isfinite(numeric)
        )
        self.counts += np.bincount(indices[valid], minlength=int(self.size))
        self.sums += np.bincount(
            indices[valid], weights=numeric[valid], minlength=int(self.size)
        )

    def finalize(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        means = np.divide(
            self.sums,
            self.counts,
            out=np.full(int(self.size), np.nan, dtype=float),
            where=self.counts > 0,
        )
        return self.counts.copy(), self.sums.copy(), means


class HistogramReducer:
    def __init__(self, edges: np.ndarray) -> None:
        self.edges = np.asarray(edges, dtype=float)
        self.counts = np.zeros(len(self.edges) - 1, dtype=np.int64)

    def add(self, values: np.ndarray) -> None:
        counts, _ = np.histogram(np.asarray(values, dtype=float), bins=self.edges)
        self.counts += counts

    def finalize(self) -> tuple[np.ndarray, np.ndarray]:
        return self.counts.copy(), self.edges.copy()


class PlotMatrixReducer:
    """Retain only fixed-width rows needed by a requested plot matrix."""

    def __init__(self, width: int) -> None:
        self.width = int(width)
        self.frames: list[int] = []
        self.rows: list[np.ndarray] = []

    def add(self, frame_index: int, values: np.ndarray) -> None:
        row = np.asarray(values, dtype=float).reshape(-1)
        if row.size != self.width:
            raise ValueError(f"Expected plot row width {self.width}, received {row.size}.")
        self.frames.append(int(frame_index))
        self.rows.append(row.copy())

    def finalize(self) -> tuple[np.ndarray, np.ndarray]:
        matrix = np.vstack(self.rows) if self.rows else np.empty((0, self.width), dtype=float)
        return np.asarray(self.frames, dtype=int), matrix


__all__ = [
    "CountSumReducer",
    "HistogramReducer",
    "PlotMatrixReducer",
    "TableAccumulator",
]
