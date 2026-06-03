"""Helpers for recording reproducible run settings."""

from __future__ import annotations

from argparse import Namespace
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


INTERNAL_ARG_PREFIXES = ("_",)
INTERNAL_ARG_NAMES = {
    "func",
    "_run",
}


def _mapping_from_args(args: Any) -> dict[str, Any]:
    if args is None:
        return {}
    if isinstance(args, dict):
        return dict(args)
    if isinstance(args, Namespace):
        return vars(args)
    if hasattr(args, "__dict__"):
        return vars(args)
    return dict(args)


def json_safe(value: Any) -> Any:
    """Convert common runtime values into JSON-serializable values."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Namespace):
        return json_safe(vars(value))
    if is_dataclass(value) and not isinstance(value, type):
        return json_safe(asdict(value))
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, pd.DataFrame):
        return {
            "type": "DataFrame",
            "rows": int(len(value)),
            "columns": [str(col) for col in value.columns],
        }
    return str(value)


def effective_settings_from_args(args: Any) -> dict[str, Any]:
    """Return public, effective user settings after parser defaults were applied."""
    raw = _mapping_from_args(args)
    settings: dict[str, Any] = {}
    for key, value in raw.items():
        key_str = str(key)
        if key_str in INTERNAL_ARG_NAMES or key_str.startswith(INTERNAL_ARG_PREFIXES):
            continue
        settings[key_str] = json_safe(value)
    return settings


def user_settings_from_args(args: Any) -> dict[str, Any]:
    """Return user-facing settings, omitting unset ``None`` values."""
    return {
        key: value
        for key, value in effective_settings_from_args(args).items()
        if value is not None
    }


def runtime_metadata_from_args(args: Any) -> dict[str, Any]:
    """Return internal runtime IDs and metadata separately from user settings."""
    raw = _mapping_from_args(args)
    keys = (
        "run_id",
        "analysis_id",
        "_analysis_id",
        "_parsed_id",
        "project_root",
        "cache_dir",
        "_snapshot_source_dir",
    )
    out = {key: json_safe(raw.get(key)) for key in keys if key in raw}
    if "_analysis_id" in out and "analysis_id" not in out:
        out["analysis_id"] = out["_analysis_id"]
    if "_parsed_id" in out and "parsed_id" not in out:
        out["parsed_id"] = out["_parsed_id"]
    return out
