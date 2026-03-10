"""Convert heterogeneous analysis results into JSON-serializable payloads."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any

try:  # pragma: no cover - optional runtime dependency
    import pandas as pd
except Exception:  # pragma: no cover - fallback for minimal environments
    pd = None

from reaxkit.webui.backend.tabular_payload import infer_columns, infer_numeric_columns


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
        sample = records[0] if isinstance(records[0], dict) else {}
        cols = infer_columns(records)
        numeric_cols = infer_numeric_columns(records)

        x_pref = ("iter", "frame_index", "time", "step", "x")
        x_col = next((name for name in x_pref if name in cols), numeric_cols[0] if numeric_cols else (cols[0] if cols else ""))
        y_col = ""
        for name in numeric_cols:
            if name != x_col:
                y_col = name
                break
        if not y_col and numeric_cols:
            y_col = numeric_cols[0]

        if x_col and y_col:
            group_col = "atom_id" if "atom_id" in cols else None
            views.append(
                {
                    "type": "plot",
                    "label": f"{y_col} vs {x_col}",
                    "x": x_col,
                    "y": y_col,
                    "group_by": group_col,
                }
            )

        hist_col = y_col or (numeric_cols[0] if numeric_cols else "")
        if hist_col:
            views.append({"type": "histogram", "label": f"{hist_col} Distribution", "value": hist_col})

        if len(numeric_cols) >= 3 or {"x", "y", "z"}.issubset(set(cols)):
            z_col = "z" if "z" in cols else (numeric_cols[2] if len(numeric_cols) > 2 else y_col)
            if x_col and y_col and z_col:
                views.append({"type": "3d", "label": "3D", "x": x_col, "y": y_col, "z": z_col})
                return views
    views.append({"type": "3d", "label": "3D"})
    return views
