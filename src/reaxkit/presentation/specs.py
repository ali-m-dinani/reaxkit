"""Typed presentation specs shared by CLI and Web UI frontends."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any

try:  # pragma: no cover - optional runtime dependency
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None


@dataclass(frozen=True)
class PresentationSpec:
    """Renderer-neutral visualization contract from analysis tasks."""

    renderer: str
    label: str
    mapping: dict[str, str] = field(default_factory=dict)
    options: dict[str, Any] = field(default_factory=dict)
    view_type: str | None = None


def _infer_view_type(renderer: str) -> str:
    key = str(renderer or "").strip().lower()
    if key in {"table"}:
        return "table"
    if key in {"scatter3d_points", "scatter3d"}:
        return "scatter3d"
    if key in {"histogram"}:
        return "histogram"
    return "plot2d"


def ensure_presentation_spec(value: Any) -> PresentationSpec | None:
    """Normalize arbitrary representation to PresentationSpec when possible."""
    if isinstance(value, PresentationSpec):
        return value
    if is_dataclass(value):
        raw = asdict(value)
    elif isinstance(value, dict):
        raw = dict(value)
    else:
        return None

    if "presentation" in raw and isinstance(raw["presentation"], dict):
        merged = dict(raw["presentation"])
        if "label" not in merged and "label" in raw:
            merged["label"] = raw.get("label")
        if "view_type" not in merged and "type" in raw:
            merged["view_type"] = raw.get("type")
        raw = merged

    renderer = raw.get("renderer") or raw.get("plot_type")
    if not renderer and raw.get("type") == "table":
        renderer = "table"
    if not renderer and raw.get("type") in {"plot", "plot2d"}:
        renderer = "single_plot"
    if not renderer and raw.get("type") == "histogram":
        renderer = "histogram"
    if not renderer and raw.get("type") in {"3d", "scatter3d"}:
        renderer = "scatter3d_points"
    if not renderer:
        return None

    mapping = raw.get("mapping")
    if not isinstance(mapping, dict):
        mapping = {}
    options = raw.get("options")
    if not isinstance(options, dict):
        options = {}

    # Backward-compatible axis aliases from legacy dicts.
    for src, dst in (
        ("x", "x_col"),
        ("y", "y_col"),
        ("z", "z_col"),
        ("color", "color_col"),
        ("group_by", "group_by_col"),
        ("value", "value_col"),
    ):
        if src in raw and dst not in mapping:
            mapping[dst] = str(raw.get(src) or "")

    label = str(raw.get("label") or renderer)
    view_type = raw.get("view_type")
    if view_type is not None:
        view_type = str(view_type)

    return PresentationSpec(
        renderer=str(renderer),
        label=label,
        mapping={str(k): str(v) for k, v in mapping.items() if v is not None},
        options=dict(options),
        view_type=view_type,
    )


def serialize_presentation_specs(values: list[Any]) -> list[dict[str, Any]]:
    """Serialize spec objects into plain dicts for artifacts/API."""
    out: list[dict[str, Any]] = []
    for value in values:
        spec = ensure_presentation_spec(value)
        if spec is None:
            continue
        item = asdict(spec)
        item["type"] = _infer_view_type(spec.view_type or spec.renderer)
        item["presentation"] = {
            "renderer": spec.renderer,
            "mapping": dict(spec.mapping),
            "options": dict(spec.options),
            "view_type": spec.view_type or item["type"],
        }
        # Keep common aliases for existing UI paths.
        item["x"] = spec.mapping.get("x_col", "")
        item["y"] = spec.mapping.get("y_col", "")
        item["z"] = spec.mapping.get("z_col", "")
        item["color"] = spec.mapping.get("color_col", "")
        item["group_by"] = spec.mapping.get("group_by_col", "")
        item["value"] = spec.mapping.get("value_col", "")
        out.append(item)
    return out


def spec_to_dash_request(value: Any) -> dict[str, str]:
    """Convert a PresentationSpec into Dash visualization node request."""
    spec = ensure_presentation_spec(value)
    if spec is None:
        return {
            "visualization_type": "plot2d",
            "x_col": "iter",
            "y_col": "msd",
            "z_col": "",
            "color_col": "",
            "group_col": "",
        }
    vtype = _infer_view_type(spec.view_type or spec.renderer)
    if vtype == "plot":
        vtype = "plot2d"
    return {
        "visualization_type": vtype,
        "x_col": str(spec.mapping.get("x_col", "")),
        "y_col": str(spec.mapping.get("y_col", "")),
        "z_col": str(spec.mapping.get("z_col", "")),
        "color_col": str(spec.mapping.get("color_col", "")),
        "group_col": str(spec.mapping.get("group_by_col", "")),
    }


def spec_to_plot_payload(value: Any, result: Any) -> dict[str, Any] | None:
    """Convert PresentationSpec into renderer payload for CLI plotting."""
    spec = ensure_presentation_spec(value)
    if spec is None:
        return None
    if spec.renderer == "table":
        return None

    payload: dict[str, Any] = {"plot_type": spec.renderer}
    payload.update(spec.options)

    table = getattr(result, "table", None)
    if pd is not None and isinstance(table, pd.DataFrame):
        rows = table.to_dict(orient="records")
    elif isinstance(table, list):
        rows = [dict(r) for r in table if isinstance(r, dict)]
    else:
        rows = []
    if not rows:
        return payload

    x_col = spec.mapping.get("x_col")
    y_col = spec.mapping.get("y_col")
    g_col = spec.mapping.get("group_by_col")

    if spec.renderer == "single_plot" and x_col and y_col:
        if g_col:
            grouped: dict[str, list[dict[str, Any]]] = {}
            for row in rows:
                grouped.setdefault(str(row.get(g_col, "all")), []).append(row)
            series = []
            for key, bucket in grouped.items():
                bucket.sort(key=lambda r: float(r.get(x_col, 0.0)))
                series.append(
                    {
                        "x": [r.get(x_col) for r in bucket],
                        "y": [r.get(y_col) for r in bucket],
                        "label": key,
                    }
                )
            payload["series"] = series
        else:
            ordered = sorted(rows, key=lambda r: float(r.get(x_col, 0.0)))
            payload["x"] = [r.get(x_col) for r in ordered]
            payload["y"] = [r.get(y_col) for r in ordered]
    return payload
