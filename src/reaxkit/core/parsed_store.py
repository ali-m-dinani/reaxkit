"""Persistence helpers for durable parsed domain artifacts."""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import pickle
import re
from typing import Any

import numpy as np
import pandas as pd

try:  # pragma: no cover - import depends on optional runtime dependency
    import h5py  # type: ignore
except Exception:  # pragma: no cover
    h5py = None  # type: ignore


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sanitize_key(name: str) -> str:
    return re.sub(r"[^0-9a-zA-Z_.-]", "_", str(name))


def _write_scalar_dataset(group, key: str, value: Any) -> None:
    if isinstance(value, str):
        group.create_dataset(key, data=np.asarray(value, dtype=h5py.string_dtype(encoding="utf-8")))
        return
    group.create_dataset(key, data=value)


def _write_dataframe(group, key: str, frame: pd.DataFrame) -> None:
    g = group.create_group(key)
    g.attrs["reaxkit_type"] = "dataframe"
    g.create_dataset(
        "columns",
        data=np.asarray([str(c) for c in frame.columns], dtype=h5py.string_dtype(encoding="utf-8")),
    )
    index_values = frame.index.to_numpy()
    if index_values.dtype.kind in {"i", "u", "f", "b"}:
        g.create_dataset("index", data=index_values)
    else:
        g.create_dataset("index", data=index_values.astype(str).astype(h5py.string_dtype(encoding="utf-8")))
    for col in frame.columns:
        col_key = _sanitize_key(str(col))
        series = frame[col]
        if pd.api.types.is_numeric_dtype(series):
            g.create_dataset(f"col__{col_key}", data=series.to_numpy())
        else:
            values = series.fillna("").astype(str).to_numpy(dtype=h5py.string_dtype(encoding="utf-8"))
            g.create_dataset(f"col__{col_key}", data=values)
        g.attrs[f"dtype__{col_key}"] = str(series.dtype)


def _write_sequence(group, key: str, values: list[Any]) -> None:
    if not values:
        g = group.create_group(key)
        g.attrs["reaxkit_type"] = "sequence"
        g.attrs["length"] = 0
        return
    if all(isinstance(v, (str, int, float, bool, np.integer, np.floating, np.bool_)) or v is None for v in values):
        arr = np.asarray(["" if v is None else v for v in values], dtype=object)
        if arr.dtype.kind in {"U", "O"}:
            arr = arr.astype(h5py.string_dtype(encoding="utf-8"))
        group.create_dataset(key, data=arr)
        return
    g = group.create_group(key)
    g.attrs["reaxkit_type"] = "sequence"
    g.attrs["length"] = int(len(values))
    for i, item in enumerate(values):
        _write_any(g, f"item_{i}", item)


def _write_fallback_pickle(group, key: str, value: Any) -> None:
    payload = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
    g = group.create_group(key)
    g.attrs["reaxkit_type"] = "pickle_fallback"
    g.create_dataset("payload", data=np.frombuffer(payload, dtype=np.uint8))


def _write_any(group, key: str, value: Any) -> None:
    key = _sanitize_key(key)
    if value is None:
        g = group.create_group(key)
        g.attrs["reaxkit_type"] = "none"
        return
    if is_dataclass(value):
        g = group.create_group(key)
        g.attrs["reaxkit_type"] = "dataclass"
        g.attrs["class"] = f"{value.__class__.__module__}.{value.__class__.__qualname__}"
        for f in fields(value):
            _write_any(g, f.name, getattr(value, f.name))
        return
    if isinstance(value, pd.DataFrame):
        _write_dataframe(group, key, value)
        return
    if isinstance(value, np.ndarray):
        group.create_dataset(key, data=value)
        return
    if isinstance(value, dict):
        g = group.create_group(key)
        g.attrs["reaxkit_type"] = "dict"
        for k, v in value.items():
            _write_any(g, str(k), v)
        return
    if isinstance(value, (list, tuple)):
        _write_sequence(group, key, list(value))
        return
    if isinstance(value, (str, int, float, bool, np.integer, np.floating, np.bool_)):
        _write_scalar_dataset(group, key, value)
        return
    _write_fallback_pickle(group, key, value)


def write_parsed_hdf5(path: Path, data: Any) -> Path:
    """Write parsed domain data to HDF5 with a best-effort structured layout."""
    if h5py is None:  # pragma: no cover
        raise RuntimeError("h5py is required to persist parsed artifacts as HDF5.")

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp")
    if tmp.exists():
        tmp.unlink()
    with h5py.File(tmp, "w") as h5:
        h5.attrs["reaxkit_format"] = "parsed-hdf5-v1"
        h5.attrs["created_at_utc"] = _utc_now_iso()
        h5.attrs["root_class"] = f"{data.__class__.__module__}.{data.__class__.__qualname__}"
        _write_any(h5, "payload", data)
    tmp.replace(path)
    return path


def update_parsed_meta(parsed_dir: Path, *, parsed_id: str, artifact_name: str, file_name: str) -> Path:
    """Update parsed meta manifest with saved artifact information."""
    meta_path = parsed_dir / "meta.json"
    payload: dict[str, Any] = {}
    if meta_path.exists():
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
    payload.setdefault("parsed_id", parsed_id)
    payload.setdefault("artifacts", {})
    payload["artifacts"][artifact_name] = {
        "file": file_name,
        "format": "hdf5",
        "updated_at": _utc_now_iso(),
    }
    meta_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return meta_path
