"""Classify per-atom coordination states from connectivity-domain inputs.

This module evaluates coordination status by comparing observed bond-order
coordination against expected valence targets. It is scoped to coordination
label assignment and tabular output, and it does not infer hybridization states.

**Usage context**

- Coordination auditing: Label atoms as under/over/coordinated per frame.
- Quality control: Compare observed coordination against force-field valences.
- Relabeling support: Supply coordination tags for downstream trajectory tasks.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from reaxkit.analysis.base import AnalysisTask
from reaxkit.core.registry.analysis_task_registry import register_task
from reaxkit.domain.base_request import BaseRequest
from reaxkit.domain.base_result import BaseResult
from reaxkit.domain.data_models import CoordinationStatusBundleData, ForceFieldParametersData
from reaxkit.presentation.specs import PresentationSpec


@dataclass
class CoordinationStatusRequest(BaseRequest):
    """Request for per-atom coordination-status classification.

    Fields
    -----
    valences : Optional[Mapping[str, float]]
        Optional element-to-valence map used for classification.
    threshold : float
        Absolute tolerance around target valence for assigning coordinated state.
    frames : Optional[Sequence[int]]
        Optional frame indices to analyze. `None` means all frames.
    every : int
        Frame stride after frame selection. Must be `>= 1`.
    require_all_valences : bool
        If `True`, missing valence mappings raise; otherwise rows may be emitted
        with undefined status values.

    Examples
    -----
    ```python
    req = CoordinationStatusRequest(valences={"C": 4.0, "O": 2.0}, threshold=0.9)
    ```
    Sample output:
    `CoordinationStatusRequest(...)`
    Meaning:
    The request defines valence targets and tolerance for status labeling.
    """

    valences: Optional[Mapping[str, float]] = dc_field(
        default=None,
        metadata={
            'label': 'Valences',
            'help': (
                "Optional element->valence map. "
                "Example: {'C': 4, 'O': 2, 'H': 1}."
            ),
        },
    )
    threshold: float = dc_field(
        default=0.9,
        metadata={
            'label': 'Threshold',
            'help': "Absolute tolerance for assigning coordinated status. Example: 0.9.",
            'min': 0.0,
        },
    )
    frames: Optional[Sequence[int]] = dc_field(
        default=None,
        metadata={
            'label': 'Frames',
            'help': "Optional frame indices to analyze. Example: [0, 10, 20].",
            'units': 'frame_index',
        },
    )
    every: int = dc_field(
        default=1,
        metadata={
            'label': 'Every',
            'help': "Stride for selected frames. Example: every=5.",
            'min': 1,
            'units': 'frames',
        },
    )
    require_all_valences: bool = dc_field(
        default=True,
        metadata={
            'label': 'Require All Valences',
            'help': (
                "If true, raise when any atom type has no valence mapping. "
                "If false, keep rows with undefined status."
            ),
            'choices': [True, False],
        },
    )


@dataclass
class CoordinationStatusResult(BaseResult):
    """Coordination-status analysis result.

    Fields
    -----
    table : pd.DataFrame
        Output table with one row per atom per analyzed frame. Typical columns:
        `frame_index`, `iter`, `atom_id`, `atom_type`, `sum_BOs`, `valence`,
        `delta`, `status`, `status_label`.
    request : CoordinationStatusRequest
        Request object used for this analysis run.

    Examples
    -----
    ```python
    result = CoordinationStatusTask().run(bundle, req)
    result.table.head()
    ```
    Sample output:
    DataFrame rows labeled as `under`, `coord`, or `over`.
    Meaning:
    Each row is one atom-frame coordination classification.
    """

    table: pd.DataFrame
    request: CoordinationStatusRequest


def _classify_coordination_for_frame(
    *,
    sum_bos: Sequence[float],
    atom_types: Sequence[str],
    valences: Mapping[str, float],
    threshold: float = 0.3,
    require_all_valences: bool = True,
) -> pd.DataFrame:
    """Classify atoms in one frame as under/coord/over coordinated."""
    sum_bos_arr = np.asarray(sum_bos, dtype=float)
    types = np.asarray(atom_types, dtype=object)
    if sum_bos_arr.shape[0] != types.shape[0]:
        raise ValueError(f"Length mismatch: sum_bos({sum_bos_arr.shape[0]}) vs atom_types({types.shape[0]})")

    val_arr = np.empty_like(sum_bos_arr, dtype=float)
    missing: list[str] = []
    for i, t in enumerate(types):
        key = str(t)
        if key in valences:
            val_arr[i] = float(valences[key])
        else:
            missing.append(key)
            val_arr[i] = np.nan

    if missing and require_all_valences:
        uniq = ", ".join(sorted(set(missing)))
        raise KeyError(f"Missing valence(s) for atom types: {uniq}")

    delta = sum_bos_arr - val_arr
    status = np.full(sum_bos_arr.shape, np.nan)
    if np.isfinite(threshold) and threshold >= 0:
        status = np.where(delta < -threshold, -1, status)
        status = np.where(np.abs(delta) <= threshold, 0, status)
        status = np.where(delta > threshold, +1, status)

    out = pd.DataFrame(
        {
            "atom_id": np.arange(1, len(sum_bos_arr) + 1, dtype=int),
            "atom_type": types.astype(str),
            "sum_BOs": sum_bos_arr.astype(float),
            "valence": val_arr.astype(float),
            "delta": delta.astype(float),
            "status": status.astype(float),
        }
    )
    if out["status"].notna().all():
        out["status"] = out["status"].astype(int)
    return out


def _status_label(series: Sequence[int | float]) -> list[Optional[str]]:
    """Convert numeric status to labels: -1->under, 0->coord, +1->over."""
    labels: list[Optional[str]] = []
    for v in series:
        if pd.isna(v):
            labels.append(None)
        else:
            vi = int(v)
            labels.append("under" if vi == -1 else ("coord" if vi == 0 else "over"))
    return labels


def _valence_map_from_request(
    req: CoordinationStatusRequest,
    *,
    force_field_data: ForceFieldParametersData,
) -> dict[str, float]:
    if req.valences:
        return {str(k): float(v) for k, v in req.valences.items()}
    atom_df = force_field_data.atom_parameters
    if atom_df is None or atom_df.empty:
        raise ValueError("force_field.atom_parameters is empty; cannot infer valences.")

    value_col = "valency"
    if value_col not in atom_df.columns:
        raise ValueError("force_field.atom_parameters must include 'valency' column.")
    if "symbol" not in atom_df.columns:
        raise ValueError("force_field.atom_parameters must include 'symbol' column.")

    val_map: dict[str, float] = {}
    for _, row in atom_df.iterrows():
        sym = str(row["symbol"]).strip()
        if not sym or sym.lower() == "nan" or pd.isna(row[value_col]):
            continue
        val_map[sym] = float(row[value_col])
    if not val_map:
        raise ValueError(f"No usable '{value_col}' values found in force_field.atom_parameters.")
    return val_map


def _sum_bond_orders_matrix(data: ConnectivityData) -> np.ndarray:
    if data.sum_bond_orders is not None:
        arr = np.asarray(data.sum_bond_orders, dtype=float)
        if arr.ndim != 2:
            raise ValueError("ConnectivityData.sum_bond_orders must have shape (n_frames, n_atoms).")
        return arr

    bo = data.bond_orders
    if bo is None:
        raise ValueError("ConnectivityData requires either sum_bond_orders or bond_orders.")

    if isinstance(bo, np.ndarray):
        if bo.ndim != 3:
            raise ValueError("ConnectivityData.bond_orders ndarray must have shape (n_frames, n_atoms, n_atoms).")
        return np.sum(bo.astype(float), axis=2)

    if isinstance(bo, (list, tuple)):
        frames = []
        for fr in bo:
            if hasattr(fr, "sum"):
                # works for scipy sparse + dense matrices
                s = np.asarray(fr.sum(axis=1)).reshape(-1).astype(float)
            else:
                s = np.sum(np.asarray(fr, dtype=float), axis=1)
            frames.append(s)
        if not frames:
            return np.empty((0, 0), dtype=float)
        return np.vstack(frames)

    raise TypeError("Unsupported bond_orders type; use ndarray, sparse-per-frame list, or provide sum_bond_orders.")


@register_task("get_coordination", label="Coordination")
class CoordinationStatusTask(AnalysisTask):
    """Per-atom coordination status over selected frames."""

    required_data = CoordinationStatusBundleData

    @staticmethod
    def recommended_presentations(
        _result: CoordinationStatusResult,
        payload: dict[str, Any],
    ) -> list[PresentationSpec]:
        """Build default table/plot presentations for coordination outputs.

        Works on
        -----
        Analyzer task output payloads

        Parameters
        -----
        _result : CoordinationStatusResult
            Analysis result object for the executed task.
        payload : dict[str, Any]
            Serialized result payload used by presentation dispatch.

        Returns
        -----
        list[PresentationSpec]
            Recommended renderer specs for table and trend plotting.

        Examples
        -----
        ```python
        specs = CoordinationStatusTask.recommended_presentations(result, payload)
        ```
        Sample output:
        A table view plus a delta/sum_BOs-vs-frame plot view when columns exist.
        Meaning:
        Coordination outputs can be rendered with default mappings.
        """
        rows = payload.get("table")
        if not isinstance(rows, list) or not rows:
            return [PresentationSpec(renderer="table", label="Table", view_type="table")]

        sample = rows[0] if isinstance(rows[0], dict) else {}
        if not isinstance(sample, dict):
            return [PresentationSpec(renderer="table", label="Table", view_type="table")]

        x_col = "frame_index" if "frame_index" in sample else ("iter" if "iter" in sample else "")
        y_col = "delta" if "delta" in sample else ("sum_BOs" if "sum_BOs" in sample else "")
        group_col = "atom_id" if "atom_id" in sample else ""
        if not x_col or not y_col:
            return [PresentationSpec(renderer="table", label="Table", view_type="table")]

        return [
            PresentationSpec(renderer="table", label="Table", view_type="table"),
            PresentationSpec(
                renderer="single_plot",
                label=f"{y_col} vs {x_col}",
                mapping={"x_col": x_col, "y_col": y_col, "group_by_col": group_col},
                options={
                    "title": f"Coordination Status: {y_col} vs {x_col}",
                    "xlabel": x_col,
                    "ylabel": y_col,
                    "legend": bool(group_col),
                },
                view_type="plot2d",
            ),
        ]

    def run(
        self,
        data: CoordinationStatusBundleData,
        request: CoordinationStatusRequest,
        reporter=None,
    ) -> CoordinationStatusResult:
        """Compute per-atom coordination status across selected frames.

        Classify atoms as under-, coordinated-, or over-coordinated. Classification compares bond-order totals against target valences from explicit maps or inferred values.
        For example, if an atom's valence is 3 and the threshold is 0.5, then:
          - sum_BOs < 2.5 -> under-coordinated
          - 2.5 <= sum_BOs <= 3.5 -> coordinated
          - sum_BOs > 3.5 -> over-coordinated

        Works on
        -----
        `CoordinationStatusBundleData` plus `CoordinationStatusRequest` inputs

        Parameters
        -----
        data : CoordinationStatusBundleData
            Bundle containing connectivity and force-field parameter data.
        request : CoordinationStatusRequest
            Selection, valence mapping, and tolerance configuration.
        reporter : Any, optional
            Optional progress callback invoked during frame processing.

        Returns
        -----
        CoordinationStatusResult
            Coordination status table with one row per atom-frame.

        Examples
        -----
        ```python
        result = CoordinationStatusTask().run(bundle, req)
        ```
        Sample output:
        `result.table` with `status` and `status_label` columns.
        Meaning:
        Atom coordination is classified using BO sums versus valence targets.
        """
        connectivity = data.connectivity
        force_field = data.force_field_parameters
        sum_bos_m = _sum_bond_orders_matrix(connectivity)
        if sum_bos_m.size == 0:
            return CoordinationStatusResult(
                table=pd.DataFrame(
                    columns=[
                        "frame_index",
                        "iter",
                        "atom_id",
                        "atom_type",
                        "sum_BOs",
                        "valence",
                        "delta",
                        "status",
                        "status_label",
                    ]
                ),
                request=request,
            )

        n_frames, n_atoms = sum_bos_m.shape
        elements = connectivity.elements if connectivity.elements is not None else ["X"] * n_atoms
        if len(elements) != n_atoms:
            raise ValueError(f"elements length ({len(elements)}) must match n_atoms ({n_atoms}).")

        if connectivity.atom_ids is not None:
            atom_ids = np.asarray(connectivity.atom_ids, dtype=int).reshape(-1)
            if atom_ids.shape[0] != n_atoms:
                raise ValueError(f"atom_ids length ({atom_ids.shape[0]}) must match n_atoms ({n_atoms}).")
        else:
            atom_ids = np.arange(1, n_atoms + 1, dtype=int)

        frame_idx = list(range(n_frames)) if request.frames is None else [int(i) for i in request.frames]
        frame_idx = [i for i in frame_idx if 0 <= i < n_frames][:: max(1, int(request.every))]
        if not frame_idx:
            return CoordinationStatusResult(table=pd.DataFrame(), request=request)

        val_map = _valence_map_from_request(request, force_field_data=force_field)
        rows: list[pd.DataFrame] = []
        total = max(1, len(frame_idx))
        if reporter:
            reporter("analyze", 0, total, "Preparing coordination status")
        for step_i, fi in enumerate(frame_idx, start=1):
            per_atom = _classify_coordination_for_frame(
                sum_bos=sum_bos_m[fi, :],
                atom_types=elements,
                valences=val_map,
                threshold=float(request.threshold),
                require_all_valences=bool(request.require_all_valences),
            )
            per_atom["atom_id"] = atom_ids
            iter_val = int(connectivity.iterations[fi]) if connectivity.iterations is not None else int(fi)
            per_atom.insert(0, "iter", iter_val)
            per_atom.insert(0, "frame_index", int(fi))
            per_atom["status_label"] = _status_label(per_atom["status"])
            rows.append(per_atom)
            if reporter:
                reporter("analyze", step_i, total, "Computing coordination status")

        out = pd.concat(rows, ignore_index=True)
        out = out.sort_values(["frame_index", "atom_id"], kind="mergesort").reset_index(drop=True)
        if reporter:
            reporter("analyze", total, total, "Finished coordination status")
        return CoordinationStatusResult(table=out, request=request)


__all__ = [
    "CoordinationStatusRequest",
    "CoordinationStatusResult",
    "CoordinationStatusTask",
]
