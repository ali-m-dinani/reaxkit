"""File-based UI performance settings for WebUI rendering."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


_DEFAULTS: dict[str, int] = {
    "plot2d_scattergl_threshold": 4000,
    "plot2d_max_points": 60000,
    "plot2d_min_points_per_trace": 600,
    "plot2d_initial_max_points": 12000,
    "plot2d_zoom_max_points": 120000,
    "plot2d_max_curves_display": 10,
}


def ui_performance_config_path() -> Path:
    """Return the package-local JSON path for UI performance settings."""
    return Path(__file__).resolve().with_name("ui_performance.json")


def _safe_int(value: Any, default: int, *, minimum: int = 1) -> int:
    try:
        parsed = int(str(value).strip())
    except Exception:
        parsed = int(default)
    return max(int(minimum), parsed)


def load_ui_performance_config() -> dict[str, int]:
    """Load UI performance settings from JSON, with strict safe defaults."""
    cfg_path = ui_performance_config_path()
    raw: dict[str, Any] = {}
    try:
        if cfg_path.exists():
            payload = json.loads(cfg_path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                raw = payload
    except Exception:
        raw = {}
    return {
        "plot2d_scattergl_threshold": _safe_int(
            raw.get("plot2d_scattergl_threshold"),
            _DEFAULTS["plot2d_scattergl_threshold"],
        ),
        "plot2d_max_points": _safe_int(
            raw.get("plot2d_max_points"),
            _DEFAULTS["plot2d_max_points"],
        ),
        "plot2d_min_points_per_trace": _safe_int(
            raw.get("plot2d_min_points_per_trace"),
            _DEFAULTS["plot2d_min_points_per_trace"],
        ),
        "plot2d_initial_max_points": _safe_int(
            raw.get("plot2d_initial_max_points"),
            _DEFAULTS["plot2d_initial_max_points"],
        ),
        "plot2d_zoom_max_points": _safe_int(
            raw.get("plot2d_zoom_max_points"),
            _DEFAULTS["plot2d_zoom_max_points"],
        ),
        "plot2d_max_curves_display": _safe_int(
            raw.get("plot2d_max_curves_display"),
            _DEFAULTS["plot2d_max_curves_display"],
        ),
    }
