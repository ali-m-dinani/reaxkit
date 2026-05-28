"""Format active-site outputs into TRACT-compatible table schemas.

This module reshapes structural and event tables into canonical TRACT column
sets with optional strict required-column validation. It is scoped to table
projection/validation and does not run active-site analyses.

**Usage context**

- Output normalization: Convert analyzer tables to TRACT column conventions.
- Validation mode: Enforce strict required-column presence for QA pipelines.
- Export preparation: Provide stable table layouts for downstream tooling.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

STRICT_STRUCTURAL_REQUIRED_COLUMNS: tuple[str, ...] = (
    "atom_id",
    "element",
    "x",
    "y",
    "z",
    "n_bonds",
    "is_undercoord",
    "has_hetero_bond",
    "d_pyr",
    "label",
)

STRICT_EVENTS_REQUIRED_COLUMNS: tuple[str, ...] = (
    "atom_id",
    "n_events_O",
    "n_events_Si",
    "first_event_frame_O",
    "first_event_frame_Si",
    "is_reactive_O",
    "is_reactive_Si",
    "total_bound_frames_O",
    "total_bound_frames_Si",
)

TRACT_STRUCTURAL_COLUMNS: tuple[str, ...] = (
    "atom_id",
    "element",
    "x",
    "y",
    "z",
    "n_bonds",
    "is_undercoord",
    "has_hetero_bond",
    "d_pyr",
    "d_ang_deg",
    "local_roughness",
    "bond_strain",
    "angle_strain",
    "psi6_re",
    "psi6_im",
    "psi6_mag",
    "psi6_ang",
    "grain_id",
    "ring_size_min",
    "ring_size_max",
    "in_non6_ring",
    "defect_type",
    "label",
    "seg_id",
    "soap_pc1",
    "soap_pc2",
    "soap_pc3",
    "soap_score",
)

TRACT_EVENTS_COLUMNS: tuple[str, ...] = (
    "atom_id",
    "n_events_O",
    "n_events_Si",
    "first_event_frame_O",
    "first_event_frame_Si",
    "is_reactive_O",
    "is_reactive_Si",
    "total_bound_frames_O",
    "total_bound_frames_Si",
    "mean_r_O_when_bound",
    "mean_r_Si_when_bound",
)


def _series_or_default(table: pd.DataFrame, name: str, default, length: int) -> pd.Series:
    if name in table.columns:
        return table[name]
    return pd.Series([default] * length, index=table.index)


def _validate_strict_required_columns(
    table: pd.DataFrame,
    required: tuple[str, ...],
    *,
    table_name: str,
) -> None:
    missing = [col for col in required if col not in table.columns]
    empty = []
    if len(table) > 0:
        empty = [col for col in required if col in table.columns and not table[col].notna().any()]

    issues = []
    if missing:
        issues.append(f"missing required columns: {', '.join(missing)}")
    if empty:
        issues.append(f"columns contain only NaN/NA: {', '.join(empty)}")
    if issues:
        raise ValueError(
            f"STRICT TRACT compatibility failed for {table_name}: " + "; ".join(issues)
        )


def to_tract_structural_table(table: pd.DataFrame, *, strict: bool = False) -> pd.DataFrame:
    """Return a TRACT-column-compatible structural table view.

    This function is a pure data-shaping adapter: it does not write files.

    Parameters
    -----
    table : pd.DataFrame
        Input structural descriptor table from active-site structural analysis.
    strict : bool, optional
        If `True`, validate required TRACT structural columns and non-empty
        required values before formatting.

    Returns
    -----
    pd.DataFrame
        Structural table containing TRACT structural columns in canonical order.

    Examples
    -----
    ```python
    tract_df = to_tract_structural_table(structural_df, strict=False)
    ```
    Sample output:
    `pd.DataFrame` with columns from `TRACT_STRUCTURAL_COLUMNS`.
    Meaning:
    Missing optional fields are filled with defaults and column order is fixed.
    """
    if strict:
        _validate_strict_required_columns(
            table,
            STRICT_STRUCTURAL_REQUIRED_COLUMNS,
            table_name="structural",
        )

    n = len(table)
    out = pd.DataFrame(
        {
            "atom_id": _series_or_default(table, "atom_id", -1, n).astype(int),
            "element": _series_or_default(table, "element", "X", n).astype(object),
            "x": _series_or_default(table, "x", np.nan, n).astype(float),
            "y": _series_or_default(table, "y", np.nan, n).astype(float),
            "z": _series_or_default(table, "z", np.nan, n).astype(float),
            "n_bonds": _series_or_default(table, "n_bonds", 0, n).astype(int),
            "is_undercoord": _series_or_default(table, "is_undercoord", False, n).astype(bool),
            "has_hetero_bond": _series_or_default(table, "has_hetero_bond", False, n).astype(bool),
            "d_pyr": _series_or_default(table, "d_pyr", np.nan, n).astype(float),
            "d_ang_deg": _series_or_default(table, "d_ang_deg", np.nan, n).astype(float),
            "local_roughness": _series_or_default(table, "local_roughness", np.nan, n).astype(float),
            "bond_strain": _series_or_default(table, "bond_strain", np.nan, n).astype(float),
            "angle_strain": _series_or_default(table, "angle_strain", np.nan, n).astype(float),
            "psi6_re": _series_or_default(table, "psi6_re", np.nan, n).astype(float),
            "psi6_im": _series_or_default(table, "psi6_im", np.nan, n).astype(float),
            "psi6_mag": _series_or_default(table, "psi6_mag", np.nan, n).astype(float),
            "psi6_ang": _series_or_default(table, "psi6_ang", np.nan, n).astype(float),
            "grain_id": _series_or_default(table, "grain_id", -1, n).astype(int),
            "ring_size_min": _series_or_default(table, "ring_size_min", -1, n).astype(int),
            "ring_size_max": _series_or_default(table, "ring_size_max", -1, n).astype(int),
            "in_non6_ring": _series_or_default(table, "in_non6_ring", False, n).astype(bool),
            "defect_type": _series_or_default(table, "defect_type", "none", n).astype(object),
            "label": _series_or_default(table, "label", "other", n).astype(object),
            "seg_id": _series_or_default(table, "seg_id", -1, n).astype(int),
            "soap_pc1": _series_or_default(table, "soap_pc1", np.nan, n).astype(float),
            "soap_pc2": _series_or_default(table, "soap_pc2", np.nan, n).astype(float),
            "soap_pc3": _series_or_default(table, "soap_pc3", np.nan, n).astype(float),
            "soap_score": _series_or_default(table, "soap_score", np.nan, n).astype(float),
        }
    )
    return out.loc[:, list(TRACT_STRUCTURAL_COLUMNS)]


def to_tract_events_table(table: pd.DataFrame, *, strict: bool = False) -> pd.DataFrame:
    """Return a TRACT-column-compatible events table view.

    This function is a pure data-shaping adapter: it does not write files.

    Parameters
    -----
    table : pd.DataFrame
        Input events table from active-site event extraction.
    strict : bool, optional
        If `True`, validate required TRACT events columns and non-empty required
        values before formatting.

    Returns
    -----
    pd.DataFrame
        Events table containing TRACT event columns in canonical order.

    Examples
    -----
    ```python
    tract_df = to_tract_events_table(events_df, strict=True)
    ```
    Sample output:
    `pd.DataFrame` with columns from `TRACT_EVENTS_COLUMNS`.
    Meaning:
    Event rows are normalized to TRACT naming, defaults, and column ordering.
    """
    if strict:
        _validate_strict_required_columns(
            table,
            STRICT_EVENTS_REQUIRED_COLUMNS,
            table_name="events",
        )

    n = len(table)
    mean_o = _series_or_default(table, "mean_r_O_when_bound", np.nan, n)
    mean_si = _series_or_default(table, "mean_r_Si_when_bound", np.nan, n)
    if "mean_r_O_when_bound" not in table.columns:
        mean_o = _series_or_default(table, "mean_contact_O_when_bound", np.nan, n)
    if "mean_r_Si_when_bound" not in table.columns:
        mean_si = _series_or_default(table, "mean_contact_Si_when_bound", np.nan, n)

    out = pd.DataFrame(
        {
            "atom_id": _series_or_default(table, "atom_id", -1, n).astype(int),
            "n_events_O": _series_or_default(table, "n_events_O", 0, n).astype(int),
            "n_events_Si": _series_or_default(table, "n_events_Si", 0, n).astype(int),
            "first_event_frame_O": _series_or_default(table, "first_event_frame_O", -1, n).astype(int),
            "first_event_frame_Si": _series_or_default(table, "first_event_frame_Si", -1, n).astype(int),
            "is_reactive_O": _series_or_default(table, "is_reactive_O", False, n).astype(bool),
            "is_reactive_Si": _series_or_default(table, "is_reactive_Si", False, n).astype(bool),
            "total_bound_frames_O": _series_or_default(table, "total_bound_frames_O", 0, n).astype(int),
            "total_bound_frames_Si": _series_or_default(table, "total_bound_frames_Si", 0, n).astype(int),
            "mean_r_O_when_bound": mean_o.astype(float),
            "mean_r_Si_when_bound": mean_si.astype(float),
        }
    )
    return out.loc[:, list(TRACT_EVENTS_COLUMNS)]
