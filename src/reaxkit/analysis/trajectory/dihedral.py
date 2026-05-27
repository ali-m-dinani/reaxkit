"""Engine-agnostic dihedral angle analysis task."""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd

from reaxkit.analysis.base import AnalysisTask
from reaxkit.core.analysis_task_registry import register_task
from reaxkit.domain.base_request import BaseRequest
from reaxkit.domain.base_result import BaseResult
from reaxkit.domain.data_models import TrajectoryData
from reaxkit.presentation.specs import PresentationSpec


def calculate_dihedral_numpy(
    p0: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
    p3: np.ndarray,
    *,
    degrees: bool = True,
) -> np.ndarray:
    """Compute signed dihedral angles from four points using NumPy."""

    a0 = np.asarray(p0, dtype=float)
    a1 = np.asarray(p1, dtype=float)
    a2 = np.asarray(p2, dtype=float)
    a3 = np.asarray(p3, dtype=float)

    b0 = a1 - a0
    b1 = a2 - a1
    b2 = a3 - a2

    n0 = np.cross(b0, b1)
    n1 = np.cross(b1, b2)

    n0_norm = np.linalg.norm(n0, axis=-1)
    n1_norm = np.linalg.norm(n1, axis=-1)
    b1_norm = np.linalg.norm(b1, axis=-1)

    with np.errstate(divide="ignore", invalid="ignore"):
        n0_hat = n0 / n0_norm[..., None]
        n1_hat = n1 / n1_norm[..., None]
        b1_hat = b1 / b1_norm[..., None]

    m1 = np.cross(n0_hat, b1_hat)
    x = np.sum(n0_hat * n1_hat, axis=-1)
    y = np.sum(m1 * n1_hat, axis=-1)
    ang = np.arctan2(y, x)

    invalid = (n0_norm <= 1e-15) | (n1_norm <= 1e-15) | (b1_norm <= 1e-15)
    ang = np.where(invalid, np.nan, ang)
    return np.degrees(ang) if degrees else ang


def calculate_dihedral_mdanalysis(
    p0: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
    p3: np.ndarray,
    *,
    degrees: bool = True,
) -> np.ndarray:
    """Compute signed dihedral angles using MDAnalysis if available."""

    try:
        from MDAnalysis.lib.distances import calc_dihedrals
    except Exception as e:
        raise ImportError(
            "MDAnalysis backend selected but MDAnalysis is not available. "
            "Install MDAnalysis or use backend='numpy'."
        ) from e

    rad = calc_dihedrals(
        np.asarray(p0, dtype=float),
        np.asarray(p1, dtype=float),
        np.asarray(p2, dtype=float),
        np.asarray(p3, dtype=float),
    )
    return np.degrees(rad) if degrees else np.asarray(rad, dtype=float)


@dataclass
class DihedralRequest(BaseRequest):
    """Request for trajectory dihedral-angle analysis."""

    atom_ids: Sequence[int] = dc_field(
        default=(1, 2, 3, 4),
        metadata={"label": "Atom IDs", "help": "Exactly four atom IDs that define the dihedral.", "units": "index"},
    )
    frames: Optional[Sequence[int]] = dc_field(
        default=None,
        metadata={"label": "Frames", "help": "Frame indices to evaluate. Empty means all frames.", "units": "frame_index"},
    )
    every: int = dc_field(
        default=1,
        metadata={"label": "Stride", "help": "Stride over selected frames.", "min": 1, "units": "frames"},
    )
    units: str = dc_field(
        default="deg",
        metadata={"label": "Units", "help": "Angle units.", "choices": ["deg", "rad"]},
    )
    backend: str = dc_field(
        default="numpy",
        metadata={"label": "Backend", "help": "Computation backend.", "choices": ["numpy", "mdanalysis"]},
    )


@dataclass
class DihedralResult(BaseResult):
    """Result of dihedral-angle analysis."""

    table: pd.DataFrame
    request: DihedralRequest


@register_task("get_dihedral", label="Dihedral")
class DihedralTask(AnalysisTask):
    """Compute the signed dihedral angle for one atom quadruplet over time."""

    required_data = TrajectoryData

    @staticmethod
    def recommended_presentations(_result: DihedralResult, payload: dict[str, Any]) -> list[PresentationSpec]:
        table_rows = payload.get("table")
        if not isinstance(table_rows, list) or not table_rows:
            return [PresentationSpec(renderer="table", label="Table", view_type="table")]
        sample = table_rows[0] if isinstance(table_rows[0], dict) else {}
        x_axis = "iter" if "iter" in sample else "frame_index"
        return [
            PresentationSpec(renderer="table", label="Table", view_type="table"),
            PresentationSpec(
                renderer="single_plot",
                label="Dihedral vs Time",
                mapping={"x_col": x_axis, "y_col": "dihedral", "group_by_col": ""},
                options={"title": "Dihedral vs Time", "xlabel": x_axis, "ylabel": "dihedral", "legend": False},
                view_type="plot2d",
            ),
        ]

    def run(self, data: TrajectoryData, request: DihedralRequest, reporter=None) -> DihedralResult:
        out_cols = ["frame_index", "iter", "atom1_id", "atom2_id", "atom3_id", "atom4_id", "dihedral", "units", "backend"]
        atom_ids = [int(aid) for aid in request.atom_ids]
        if len(atom_ids) != 4:
            raise ValueError("atom_ids must contain exactly 4 atom IDs.")
        if len(set(atom_ids)) != 4:
            raise ValueError("atom_ids must be unique.")

        idx_map = {int(atom_id): i for i, atom_id in enumerate(data.atom_ids)}
        try:
            i0, i1, i2, i3 = [idx_map[aid] for aid in atom_ids]
        except KeyError as e:
            raise ValueError(f"atom_id {int(e.args[0])} was not found in trajectory atom_ids.") from e

        n_frames = int(data.positions.shape[0])
        frame_idx = list(range(n_frames)) if request.frames is None else [int(i) for i in request.frames]
        frame_idx = frame_idx[:: max(1, int(request.every))]
        if not frame_idx:
            return DihedralResult(table=pd.DataFrame(columns=out_cols), request=request)

        selected = np.asarray(frame_idx, dtype=int)
        coords = np.asarray(data.positions, dtype=float)
        p0 = coords[selected, i0, :]
        p1 = coords[selected, i1, :]
        p2 = coords[selected, i2, :]
        p3 = coords[selected, i3, :]

        units = str(request.units).strip().lower()
        if units not in {"deg", "rad"}:
            raise ValueError("units must be 'deg' or 'rad'.")
        backend = str(request.backend).strip().lower()
        if backend not in {"numpy", "mdanalysis"}:
            raise ValueError("backend must be 'numpy' or 'mdanalysis'.")

        if backend == "mdanalysis":
            angles = calculate_dihedral_mdanalysis(p0, p1, p2, p3, degrees=(units == "deg"))
        else:
            angles = calculate_dihedral_numpy(p0, p1, p2, p3, degrees=(units == "deg"))

        rows: list[dict[str, float | int | str]] = []
        total = len(frame_idx)
        for step_i, (fi, angle) in enumerate(zip(frame_idx, angles), start=1):
            iter_val = int(data.iterations[fi]) if data.iterations is not None else int(fi)
            rows.append(
                {
                    "frame_index": int(fi),
                    "iter": iter_val,
                    "atom1_id": atom_ids[0],
                    "atom2_id": atom_ids[1],
                    "atom3_id": atom_ids[2],
                    "atom4_id": atom_ids[3],
                    "dihedral": float(angle),
                    "units": units,
                    "backend": backend,
                }
            )
            if reporter:
                reporter("analyze", step_i, total, "Computing dihedral")

        table = pd.DataFrame(rows).sort_values("frame_index", kind="stable").reset_index(drop=True)
        if reporter:
            reporter("analyze", total, total, "Finished dihedral")
        return DihedralResult(table=table, request=request)


__all__ = [
    "calculate_dihedral_numpy",
    "calculate_dihedral_mdanalysis",
    "DihedralRequest",
    "DihedralResult",
    "DihedralTask",
]
