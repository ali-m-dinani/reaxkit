"""Shared helpers for extracting tabular payload rows."""

from __future__ import annotations

from typing import Any


def extract_tabular_rows(payload: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Return tabular rows from a payload, preferring canonical table keys."""
    if not isinstance(payload, dict):
        return []

    def _rows_from(value: Any) -> list[dict[str, Any]]:
        if not isinstance(value, list):
            return []
        out: list[dict[str, Any]] = []
        for row in value:
            if isinstance(row, dict):
                out.append(dict(row))
        return out

    # Prefer explicit tabular fields before generic list-like payload entries.
    for key in ("table", "rows", "records", "data"):
        rows = _rows_from(payload.get(key))
        if rows:
            return rows

    for value in payload.values():
        rows = _rows_from(value)
        if rows:
            return rows
    return []


def infer_columns(rows: list[dict[str, Any]]) -> list[str]:
    """Infer ordered columns from the first row."""
    if not rows:
        return []
    return [str(col) for col in rows[0].keys()]


def infer_numeric_columns(rows: list[dict[str, Any]], *, sample_size: int = 50) -> list[str]:
    """Infer numeric columns using a lightweight sample."""
    cols = infer_columns(rows)
    if not cols:
        return []
    sample = rows[: max(1, int(sample_size))]
    out: list[str] = []
    for col in cols:
        for row in sample:
            val = row.get(col)
            if isinstance(val, bool):
                continue
            if isinstance(val, (int, float)):
                out.append(col)
                break
    return out
