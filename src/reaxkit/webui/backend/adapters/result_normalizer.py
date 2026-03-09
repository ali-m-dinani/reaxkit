"""Convert heterogeneous analysis results into JSON-serializable payloads."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any

try:  # pragma: no cover - optional runtime dependency
    import pandas as pd
except Exception:  # pragma: no cover - fallback for minimal environments
    pd = None


def _serialize_value(value: Any) -> Any:
    if pd is not None:
        if isinstance(value, pd.DataFrame):
            return value.to_dict(orient="records")
        if isinstance(value, pd.Series):
            return value.to_list()
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, (list, dict, str, int, float, bool)) or value is None:
        return value
    return str(value)


def normalize_result(result: object) -> dict[str, Any]:
    """Normalize a result object to a dictionary payload."""
    if is_dataclass(result):
        raw = asdict(result)
    elif hasattr(result, "__dict__"):
        raw = dict(vars(result))
    else:
        raw = {"value": result}

    payload: dict[str, Any] = {}
    for key, value in raw.items():
        payload[key] = _serialize_value(value)
    return payload


def recommend_views(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Generate default view hints from normalized payload."""
    views: list[dict[str, Any]] = [{"type": "table", "label": "Table"}]
    records = None
    for value in payload.values():
        if isinstance(value, list) and value and isinstance(value[0], dict):
            records = value
            break
    if records:
        sample = records[0]
        if {"frame_index", "msd"}.issubset(sample.keys()):
            x_axis = "iter" if "iter" in sample else "frame_index"
            views.append(
                {
                    "type": "plot",
                    "label": "MSD vs Time",
                    "x": x_axis,
                    "y": "msd",
                    "group_by": "atom_id" if "atom_id" in sample else None,
                }
            )
            views.append({"type": "histogram", "label": "MSD Distribution", "value": "msd"})
            views.append({"type": "3d", "label": "3D"})
            return views

        numeric_cols = []
        for key, val in sample.items():
            if isinstance(val, (int, float)):
                numeric_cols.append(key)
        if len(numeric_cols) >= 2:
            views.append(
                {
                    "type": "plot",
                    "label": "Recommended Plot",
                    "x": numeric_cols[0],
                    "y": numeric_cols[1],
                }
            )
        if numeric_cols:
            views.append({"type": "histogram", "label": "Histogram", "value": numeric_cols[0]})
    views.append({"type": "3d", "label": "3D"})
    return views
