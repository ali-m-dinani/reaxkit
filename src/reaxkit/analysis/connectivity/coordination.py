"""Engine-agnostic coordination analysis task."""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from typing import Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from reaxkit.analysis.base import AnalysisTask
from reaxkit.core.analysis_task_registry import register_task
from reaxkit.domain.base_request import BaseRequest
from reaxkit.domain.base_result import BaseResult
from reaxkit.domain.data_models import ConnectivityData, ForceFieldParametersData


@dataclass
class CoordinationStatusRequest(BaseRequest):
    """Request for per-atom coordination status classification."""

    valences: Optional[Mapping[str, float]] = dc_field(
        default=None,
        metadata={'label': 'Valences', 'help': 'Valences parameter for CoordinationStatusRequest.'},
    )
    force_field: Optional[ForceFieldParametersData] = dc_field(
        default=None,
        metadata={'label': 'Force Field', 'help': 'Force Field parameter for CoordinationStatusRequest.'},
    )
    valence_key: str = dc_field(
        default="valency",
        metadata={'label': 'Valence Key', 'help': 'Valence Key parameter for CoordinationStatusRequest.'},
    )
    threshold: float = dc_field(
        default=0.9,
        metadata={'label': 'Threshold', 'help': 'Threshold parameter for CoordinationStatusRequest.', 'min': 0.0},
    )
    frames: Optional[Sequence[int]] = dc_field(
        default=None,
        metadata={'label': 'Frames', 'help': 'Frames parameter for CoordinationStatusRequest.', 'units': 'frame_index'},
    )
    every: int = dc_field(
        default=1,
        metadata={'label': 'Every', 'help': 'Every parameter for CoordinationStatusRequest.', 'min': 1, 'units': 'frames'},
    )
    require_all_valences: bool = dc_field(
        default=True,
        metadata={'label': 'Require All Valences', 'help': 'Require All Valences parameter for CoordinationStatusRequest.', 'choices': [True, False]},
    )


@dataclass
class CoordinationStatusResult(BaseResult):
    """Result of coordination status classification."""

    table: pd.DataFrame


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


def _valence_map_from_request(req: CoordinationStatusRequest) -> dict[str, float]:
    if req.valences:
        return {str(k): float(v) for k, v in req.valences.items()}
    if req.force_field is None:
        raise ValueError("Provide either request.valences or request.force_field with atom valence data.")

    atom_df = req.force_field.atom_parameters
    if atom_df is None or atom_df.empty:
        raise ValueError("force_field.atom_parameters is empty; cannot infer valences.")

    key = str(req.valence_key)
    candidate_keys = [key]
    if key == "valence":
        candidate_keys.append("valency")

    value_col = next((k for k in candidate_keys if k in atom_df.columns), None)
    if value_col is None:
        raise ValueError(f"No valence column found in force_field.atom_parameters for keys={candidate_keys}.")
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


@register_task("coordination")
class CoordinationStatusTask(AnalysisTask):
    """Per-atom coordination status over selected frames."""

    required_data = ConnectivityData

    def run(self, data: ConnectivityData, request: CoordinationStatusRequest) -> CoordinationStatusResult:
        sum_bos_m = _sum_bond_orders_matrix(data)
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
                )
            )

        n_frames, n_atoms = sum_bos_m.shape
        elements = data.elements if data.elements is not None else ["X"] * n_atoms
        if len(elements) != n_atoms:
            raise ValueError(f"elements length ({len(elements)}) must match n_atoms ({n_atoms}).")

        if data.atom_ids is not None:
            atom_ids = np.asarray(data.atom_ids, dtype=int).reshape(-1)
            if atom_ids.shape[0] != n_atoms:
                raise ValueError(f"atom_ids length ({atom_ids.shape[0]}) must match n_atoms ({n_atoms}).")
        else:
            atom_ids = np.arange(1, n_atoms + 1, dtype=int)

        frame_idx = list(range(n_frames)) if request.frames is None else [int(i) for i in request.frames]
        frame_idx = [i for i in frame_idx if 0 <= i < n_frames][:: max(1, int(request.every))]
        if not frame_idx:
            return CoordinationStatusResult(table=pd.DataFrame())

        val_map = _valence_map_from_request(request)
        rows: list[pd.DataFrame] = []
        for fi in frame_idx:
            per_atom = _classify_coordination_for_frame(
                sum_bos=sum_bos_m[fi, :],
                atom_types=elements,
                valences=val_map,
                threshold=float(request.threshold),
                require_all_valences=bool(request.require_all_valences),
            )
            per_atom["atom_id"] = atom_ids
            iter_val = int(data.iterations[fi]) if data.iterations is not None else int(fi)
            per_atom.insert(0, "iter", iter_val)
            per_atom.insert(0, "frame_index", int(fi))
            per_atom["status_label"] = _status_label(per_atom["status"])
            rows.append(per_atom)

        out = pd.concat(rows, ignore_index=True)
        out = out.sort_values(["frame_index", "atom_id"], kind="mergesort").reset_index(drop=True)
        return CoordinationStatusResult(table=out)


__all__ = [
    "CoordinationStatusRequest",
    "CoordinationStatusResult",
    "CoordinationStatusTask",
]
