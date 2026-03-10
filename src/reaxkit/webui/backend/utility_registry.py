"""Utility node registry shared by Web UI backend and UI editors."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any
import math


@dataclass(frozen=True)
class UtilityFieldSpec:
    """Schema for one utility request field."""

    name: str
    type: str
    default: Any = None
    required: bool = False
    help: str = ""


@dataclass(frozen=True)
class UtilitySpec:
    """Utility node contract."""

    name: str
    label: str
    aliases: tuple[str, ...] = field(default_factory=tuple)
    fields: tuple[UtilityFieldSpec, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["fields"] = [asdict(field) for field in self.fields]
        return payload


_UTILITY_SPECS: dict[str, UtilitySpec] = {
    "join_tables": UtilitySpec(
        name="join_tables",
        label="Join tables",
        fields=(
            UtilityFieldSpec("right_source_node_id", "str", default="", required=True, help="Node id of the second table source."),
            UtilityFieldSpec("keys", "list[str]", default=["frame_index", "iter", "atom_id"], help="Columns used to join rows."),
            UtilityFieldSpec("how", "str", default="inner", help="Join mode: inner, left, or outer."),
            UtilityFieldSpec("left_suffix", "str", default="", help="Suffix added to overlapping left columns."),
            UtilityFieldSpec("right_suffix", "str", default="_right", help="Suffix added to overlapping right columns."),
        ),
    ),
    "filter_rows": UtilitySpec(
        name="filter_rows",
        label="Filter rows",
        aliases=("filter_atoms",),
        fields=(
            UtilityFieldSpec("column", "str", default=""),
            UtilityFieldSpec("values", "str", default="", help="Comma-separated values."),
        ),
    ),
    "frame_range": UtilitySpec(
        name="frame_range",
        label="Frame range",
        fields=(
            UtilityFieldSpec("min_frame", "int | None", default=None),
            UtilityFieldSpec("max_frame", "int | None", default=None),
        ),
    ),
    "denoise_ema": UtilitySpec(
        name="denoise_ema",
        label="Denoise (EMA)",
        fields=(
            UtilityFieldSpec("column", "str", default=""),
            UtilityFieldSpec("alpha", "float", default=0.3),
            UtilityFieldSpec("group_by", "str", default=""),
            UtilityFieldSpec("x_col", "str", default=""),
        ),
    ),
    "denoise_sma": UtilitySpec(
        name="denoise_sma",
        label="Denoise (SMA)",
        fields=(
            UtilityFieldSpec("column", "str", default=""),
            UtilityFieldSpec("window", "int", default=5),
            UtilityFieldSpec("group_by", "str", default=""),
            UtilityFieldSpec("x_col", "str", default=""),
        ),
    ),
    "column_transform": UtilitySpec(
        name="column_transform",
        label="Column transform",
        fields=(
            UtilityFieldSpec("source", "str", default=""),
            UtilityFieldSpec("new_column", "str", default=""),
            UtilityFieldSpec("scale", "float", default=1.0),
            UtilityFieldSpec("offset", "float", default=0.0),
        ),
    ),
}

_UTILITY_ALIASES: dict[str, str] = {}
for _name, _spec in _UTILITY_SPECS.items():
    _UTILITY_ALIASES[_name] = _name
    for _alias in _spec.aliases:
        _UTILITY_ALIASES[str(_alias).strip().lower()] = _name


def utility_specs() -> dict[str, UtilitySpec]:
    """Return utility specs by canonical utility name."""
    return dict(_UTILITY_SPECS)


def utility_specs_payload() -> list[dict[str, Any]]:
    """Serialize utility specs for API/catalog usage."""
    out: list[dict[str, Any]] = []
    for name in sorted(_UTILITY_SPECS.keys()):
        out.append(_UTILITY_SPECS[name].to_dict())
    return out


def canonical_utility_name(raw_name: str | None) -> str:
    """Normalize utility name and resolve aliases."""
    key = str(raw_name or "").strip().lower()
    return _UTILITY_ALIASES.get(key, key)


def _pick_column(columns: list[str], preferred: tuple[str, ...], fallback: str = "") -> str:
    for candidate in preferred:
        if candidate in columns:
            return candidate
    return columns[0] if columns else fallback


def default_utility_request(
    utility_name: str,
    *,
    columns: list[str] | None = None,
    numeric_columns: list[str] | None = None,
) -> dict[str, Any]:
    """Build default request payload for a utility using optional column hints."""
    name = canonical_utility_name(utility_name)
    cols = [str(col) for col in (columns or [])]
    ncols = [str(col) for col in (numeric_columns or [])] or cols
    if name == "filter_rows":
        return {
            "column": _pick_column(cols, ("atom_id", "id", "atom", "frame_index"), ""),
            "values": "",
        }
    if name == "join_tables":
        keys = [col for col in ("frame_index", "iter", "atom_id") if col in cols]
        return {
            "right_source_node_id": "",
            "keys": keys,
            "how": "inner",
            "left_suffix": "",
            "right_suffix": "_right",
        }
    if name == "frame_range":
        return {"min_frame": None, "max_frame": None}
    if name == "denoise_ema":
        value_col = _pick_column(ncols, ("msd", "value", "y"), "")
        return {
            "column": value_col,
            "alpha": 0.3,
            "group_by": _pick_column(cols, ("atom_id", "id", "group"), ""),
            "x_col": _pick_column(cols, ("iter", "frame_index", "time", "step"), ""),
        }
    if name == "denoise_sma":
        value_col = _pick_column(ncols, ("msd", "value", "y"), "")
        return {
            "column": value_col,
            "window": 5,
            "group_by": _pick_column(cols, ("atom_id", "id", "group"), ""),
            "x_col": _pick_column(cols, ("iter", "frame_index", "time", "step"), ""),
        }
    if name == "column_transform":
        src = _pick_column(ncols, ("msd", "value", "y"), "")
        return {
            "source": src,
            "new_column": f"{src}_transformed" if src else "transformed_value",
            "scale": 1.0,
            "offset": 0.0,
        }
    return {}


def apply_utility_rows(
    utility_name: str,
    rows: list[dict[str, Any]],
    request: dict[str, Any] | None,
    *,
    other_rows: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Apply a utility transformation to tabular rows."""
    name = canonical_utility_name(utility_name)
    req = dict(request or {})
    defaults = default_utility_request(name, columns=list(rows[0].keys()) if rows else [], numeric_columns=_numeric_columns(rows))
    if defaults:
        merged = dict(defaults)
        merged.update(req)
        req = merged

    if name == "join_tables":
        return _join_rows(
            rows,
            other_rows or [],
            keys=_normalize_join_keys(req.get("keys")),
            how=str(req.get("how") or "inner").strip().lower(),
            left_suffix=str(req.get("left_suffix") or ""),
            right_suffix=str(req.get("right_suffix") or "_right"),
        )

    if name == "filter_rows":
        column = str(req.get("column") or defaults.get("column") or "")
        values = req.get("values")
        keep: set[str] = set()
        if isinstance(values, str):
            keep = {token.strip() for token in values.split(",") if token.strip()}
        elif isinstance(values, list):
            keep = {str(v) for v in values if str(v).strip()}
        if not keep or not column:
            return [dict(row) for row in rows]
        return [dict(row) for row in rows if str(row.get(column)) in keep]

    if name == "frame_range":
        fmin = req.get("min_frame")
        fmax = req.get("max_frame")
        out: list[dict[str, Any]] = []
        for row in rows:
            fv = row.get("frame_index")
            if fv is None:
                continue
            try:
                frame_value = int(fv)
            except Exception:
                continue
            if fmin is not None and frame_value < int(fmin):
                continue
            if fmax is not None and frame_value > int(fmax):
                continue
            out.append(dict(row))
        return out

    if name in {"denoise_ema", "denoise_sma"}:
        column = str(req.get("column") or "")
        group_col = str(req.get("group_by") or "")
        x_col = str(req.get("x_col") or "")
        if not column:
            return [dict(row) for row in rows]

        groups: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            key = str(row.get(group_col, "all")) if group_col else "all"
            groups.setdefault(key, []).append(dict(row))

        out_rows: list[dict[str, Any]] = []
        for group_rows in groups.values():
            if x_col:
                group_rows.sort(key=lambda r: _to_float(r.get(x_col), 0.0))
            if name == "denoise_ema":
                alpha = _to_float(req.get("alpha"), 0.3)
                alpha = max(0.001, min(1.0, alpha))
                ema: float | None = None
                for row in group_rows:
                    raw = row.get(column)
                    try:
                        val = float(raw)
                    except Exception:
                        out_rows.append(row)
                        continue
                    ema = val if ema is None else (alpha * val + (1.0 - alpha) * ema)
                    row[f"{column}_ema"] = ema
                    out_rows.append(row)
            else:
                window = max(1, int(_to_float(req.get("window"), 5)))
                queue: list[float] = []
                for row in group_rows:
                    raw = row.get(column)
                    try:
                        val = float(raw)
                    except Exception:
                        out_rows.append(row)
                        continue
                    queue.append(val)
                    if len(queue) > window:
                        queue.pop(0)
                    row[f"{column}_sma"] = sum(queue) / float(len(queue))
                    out_rows.append(row)
        return out_rows

    if name == "column_transform":
        src = str(req.get("source") or "")
        dst = str(req.get("new_column") or f"{src}_transformed")
        scale = _to_float(req.get("scale"), 1.0)
        offset = _to_float(req.get("offset"), 0.0)
        out: list[dict[str, Any]] = []
        for row in rows:
            next_row = dict(row)
            try:
                next_row[dst] = float(next_row.get(src)) * scale + offset
            except Exception:
                next_row[dst] = math.nan
            out.append(next_row)
        return out

    raise ValueError(f"Unsupported utility node '{utility_name}'")


def _normalize_join_keys(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        return [token.strip() for token in raw.split(",") if token.strip()]
    if isinstance(raw, (list, tuple, set)):
        return [str(token).strip() for token in raw if str(token).strip()]
    return []


def _infer_join_keys(left_rows: list[dict[str, Any]], right_rows: list[dict[str, Any]]) -> list[str]:
    left_cols = infer_columns_for_rows(left_rows)
    right_cols = set(infer_columns_for_rows(right_rows))
    preferred = [col for col in ("frame_index", "iter", "atom_id") if col in left_cols and col in right_cols]
    if preferred:
        return preferred
    return [col for col in left_cols if col in right_cols]


def _join_rows(
    left_rows: list[dict[str, Any]],
    right_rows: list[dict[str, Any]],
    *,
    keys: list[str],
    how: str,
    left_suffix: str,
    right_suffix: str,
) -> list[dict[str, Any]]:
    if not left_rows:
        if how == "outer":
            return [_merge_join_pair(None, row, keys=keys, left_overlap=set(), right_overlap=set(), left_suffix=left_suffix, right_suffix=right_suffix) for row in right_rows]
        return []
    if not right_rows:
        if how in {"left", "outer"}:
            return [_merge_join_pair(row, None, keys=keys, left_overlap=set(), right_overlap=set(), left_suffix=left_suffix, right_suffix=right_suffix) for row in left_rows]
        return []

    join_keys = keys or _infer_join_keys(left_rows, right_rows)
    if not join_keys:
        raise ValueError("join_tables could not infer join keys. Select one or more shared columns.")

    for key in join_keys:
        if key not in left_rows[0] or key not in right_rows[0]:
            raise ValueError(f"join_tables key '{key}' must exist in both tables.")

    how_norm = str(how or "inner").strip().lower()
    if how_norm not in {"inner", "left", "outer"}:
        raise ValueError("join_tables how must be 'inner', 'left', or 'outer'.")

    left_cols = set(infer_columns_for_rows(left_rows))
    right_cols = set(infer_columns_for_rows(right_rows))
    overlap = {col for col in left_cols.intersection(right_cols) if col not in set(join_keys)}

    right_map: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in right_rows:
        right_map.setdefault(tuple(row.get(key) for key in join_keys), []).append(dict(row))

    out: list[dict[str, Any]] = []
    matched_right_ids: set[int] = set()
    for left_row in left_rows:
        left_copy = dict(left_row)
        key_tuple = tuple(left_copy.get(key) for key in join_keys)
        matches = right_map.get(key_tuple, [])
        if matches:
            for right_row in matches:
                out.append(
                    _merge_join_pair(
                        left_copy,
                        right_row,
                        keys=join_keys,
                        left_overlap=overlap,
                        right_overlap=overlap,
                        left_suffix=left_suffix,
                        right_suffix=right_suffix,
                    )
                )
                matched_right_ids.add(id(right_row))
        elif how_norm in {"left", "outer"}:
            out.append(
                _merge_join_pair(
                    left_copy,
                    None,
                    keys=join_keys,
                    left_overlap=overlap,
                    right_overlap=overlap,
                    left_suffix=left_suffix,
                    right_suffix=right_suffix,
                )
            )

    if how_norm == "outer":
        for right_row in right_rows:
            if id(right_row) in matched_right_ids:
                continue
            out.append(
                _merge_join_pair(
                    None,
                    dict(right_row),
                    keys=join_keys,
                    left_overlap=overlap,
                    right_overlap=overlap,
                    left_suffix=left_suffix,
                    right_suffix=right_suffix,
                )
            )
    return out


def _merge_join_pair(
    left_row: dict[str, Any] | None,
    right_row: dict[str, Any] | None,
    *,
    keys: list[str],
    left_overlap: set[str],
    right_overlap: set[str],
    left_suffix: str,
    right_suffix: str,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if left_row is not None:
        for key, value in left_row.items():
            if key in keys:
                out[key] = value
            elif key in left_overlap and left_suffix:
                out[f"{key}{left_suffix}"] = value
            else:
                out[key] = value
    if right_row is not None:
        for key, value in right_row.items():
            if key in keys:
                out.setdefault(key, value)
            elif key in right_overlap:
                out[f"{key}{right_suffix}"] = value
            else:
                out[key] = value
    return out


def infer_columns_for_rows(rows: list[dict[str, Any]]) -> list[str]:
    if not rows:
        return []
    return [str(col) for col in rows[0].keys()]


def _numeric_columns(rows: list[dict[str, Any]], *, sample_size: int = 50) -> list[str]:
    if not rows:
        return []
    cols = [str(col) for col in rows[0].keys()]
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


def _to_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)
