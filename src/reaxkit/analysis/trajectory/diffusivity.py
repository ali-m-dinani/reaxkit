"""Estimate diffusivity from trajectory MSD using Einstein-relation fitting.

This module computes diffusion coefficients by fitting mean-squared
displacement trends over selected frames and dimensions. It is scoped to
diffusivity estimation and relies on MSD-derived displacement data.

**Usage context**

- Transport studies: Estimate diffusion constants for selected atom sets.
- Comparative analysis: Compare diffusivity across atom types or conditions.
- Kinetic reporting: Export fit-ready diffusivity tables and slopes.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd

from reaxkit.analysis.base import AnalysisTask
from reaxkit.core.registry.analysis_task_registry import register_task
from reaxkit.domain.base_request import BaseRequest
from reaxkit.domain.base_result import BaseResult
from reaxkit.domain.data_models import TrajectoryData
from reaxkit.analysis.trajectory.msd import MSDRequest
from reaxkit.analysis.trajectory.msd_task import MSDTask
from reaxkit.presentation.specs import PresentationSpec

DIFFUSIVITY_COLUMNS = ["atom_id", "atom_type", "dim", "d", "x_source", "x_start", "x_end",
                       "n_points", "slope_msd_per_x", "intercept", "diffusivity"]


@dataclass
class DiffusivityRequest(BaseRequest):
    """Request payload for diffusivity analysis via Einstein relation.

    Defines atom selection, frame sampling, MSD dimensional setup, and fitting
    dimensionality used to estimate diffusion coefficients.

    Fields
    -----
    atom_ids : Optional[list[int]]
        Atom IDs to include. When set, takes precedence over `atom_types`.
    atom_types : Optional[list[str]]
        Atom types/elements to include when `atom_ids` is not set.
    dims : Sequence[str]
        Coordinate axes used in MSD (`x`, `y`, `z`). Default is all three.
    origin : str | int
        Reference frame selector for MSD baseline (`"first"` or frame index).
    frames : Optional[Sequence[int]]
        Frame indices to evaluate. `None` means all frames.
    every : int
        Stride over selected frames. Must be `>= 1`.
    d : float
        Dimensionality factor in `MSD = 2*d*D*t`; must be positive.
    unwrap : bool
        Whether to unwrap coordinates across periodic boundaries before MSD.

    Examples
    -----
    ```python
    req = DiffusivityRequest(atom_types=["Li"], dims=("x", "y", "z"), d=3.0)
    ```
    Sample output:
    `DiffusivityRequest(...)`
    Meaning:
    The request configures per-atom diffusivity estimation for selected atoms.
    """

    atom_ids: Optional[list[int]] = dc_field(
        default=None,
        metadata={"label": "Atom IDs", "help": "Atom IDs to include. Empty means all atoms.", "units": "index"},
    )
    atom_types: Optional[list[str]] = dc_field(
        default=None,
        metadata={"label": "Atom types", "help": "Element symbols to include when atom_ids is empty."},
    )
    dims: Sequence[str] = dc_field(
        default=("x", "y", "z"),
        metadata={"label": "Dimensions", "help": "Coordinate axes used in MSD calculation.", "choices": ["x", "y", "z"]},
    )
    origin: str | int = dc_field(
        default="first",
        metadata={"label": "Reference origin", "help": "Reference frame: 'first' or an explicit frame index."},
    )
    frames: Optional[Sequence[int]] = dc_field(
        default=None,
        metadata={"label": "Frames", "help": "Frame indices to evaluate. Empty means all frames.", "units": "frame_index"},
    )
    every: int = dc_field(
        default=1,
        metadata={"label": "Stride", "help": "Stride over selected frames.", "min": 1, "units": "frames"},
    )
    d: float = dc_field(
        default=3.0,
        metadata={"label": "Dimensionality d", "help": "Einstein relation dimensionality in MSD = 2*d*D*t.", "min": 0.0},
    )
    unwrap: bool = dc_field(
        default=True,
        metadata={"label": "Unwrap PBC", "help": "Unwrap coordinates across periodic boundaries when cell data is available."},
    )
    max_lag: Optional[int] = dc_field(
        default=None,
        metadata={
            "label": "Maximum lag",
            "help": "Maximum lag in number of selected frames for MSD fitting.",
            "units": "frames",
        },
    )

    delta_t_ps: float = dc_field(
        default=1.0,
        metadata={
            "label": "Frame spacing",
            "help": "Time between selected trajectory frames.",
            "units": "ps",
        },
    )


@dataclass
class DiffusivityResult(BaseResult):
    """Result payload for diffusivity estimation.

    Stores fit-derived diffusion metrics per atom together with the request
    configuration used to produce them.

    Fields
    -----
    table : pd.DataFrame
        Output table containing per-atom fit metadata and diffusivity values.
        Columns include
        `["atom_id", "atom_type", "dim", "d", "x_source", "x_start", "x_end", "n_points", "slope_msd_per_x", "intercept", "diffusivity"]`.
    request : DiffusivityRequest
        Request object used for this analysis execution.

    Examples
    -----
    ```python
    result = DiffusivityTask().run(data, req)
    result.table[["atom_id", "diffusivity"]]
    ```
    Sample output:
    DataFrame rows mapping each selected atom to a diffusion coefficient.
    Meaning:
    Each row summarizes one linear MSD fit and the resulting `D` estimate.
    """

    table: pd.DataFrame
    request: DiffusivityRequest


def _axis_source(data: TrajectoryData, n_frames: int) -> tuple[np.ndarray, str]:
    if data.simulation is not None and data.simulation.time is not None:
        sim_t = np.asarray(data.simulation.time, dtype=float)
        if sim_t.shape[0] == n_frames:
            return sim_t, "time"
    if data.iterations is not None:
        iters = np.asarray(data.iterations, dtype=float)
        if iters.shape[0] == n_frames:
            return iters, "iter"
    return np.arange(n_frames, dtype=float), "frame_index"


@register_task("get_diffusivity", label="Diffusivity")
class DiffusivityTask(AnalysisTask):
    """Estimate per-atom diffusivity using Einstein's relation.

    Einstein relation:
        MSD(t) = 2 * d * D * t
    """

    required_data = TrajectoryData
    supports_selective_streaming = True

    def run_blocks(self, frames, request, reporter=None, pipeline=None):
        """Fit one atom trace at a time from a single disk-backed input pass.

        Retaining an entire trace preserves the existing all-finite PBC rule
        and polyfit behavior, including a reference later than the first frame.
        Memory is O(frames + atoms), instead of O(frames * atoms).
        """
        from contextlib import closing
        from dataclasses import replace
        from reaxkit.core.runtime.execution_contracts import resolve_execution_policy
        from reaxkit.core.runtime.frame_pipeline import BoundedFramePipeline
        from reaxkit.core.runtime.frame_tables import selected_frame_envelopes
        from reaxkit.core.runtime.trajectory_spool import TrajectorySpool
        pipeline = pipeline or BoundedFramePipeline(resolve_execution_policy(self, request))
        if not np.isfinite(request.d) or request.d <= 0:
            raise ValueError("d must be a positive finite number.")
        if not any(dim in {"x", "y", "z"} for dim in request.dims):
            raise ValueError("dims must include at least one of 'x', 'y', or 'z'.")
        spool = TrajectorySpool()
        sources, axis_source = [], None
        try:
            with closing(pipeline.iter_blocks(selected_frame_envelopes(frames, request))) as blocks:
                for block in blocks:
                    for item in block:
                        data = item.payload
                        if axis_source is None:
                            axis_source = "time" if data.simulation is not None and data.simulation.time is not None else "iter" if data.iterations is not None else "frame_index"
                        if data.iterations is None:
                            data = replace(data, iterations=np.array([item.source_frame]))
                        spool.append(len(sources), data)
                        sources.append(item.source_frame)
            if not sources:
                return DiffusivityResult(table=pd.DataFrame(columns=DIFFUSIVITY_COLUMNS), request=request)
            data = spool.finish()
            origin = 0 if request.origin == "first" else sources.index(int(request.origin))
            if request.atom_ids is not None:
                missing = set(request.atom_ids) - set(data.atom_ids)
                if missing:
                    raise ValueError(f"Requested atom_ids are not present in trajectory: {sorted(missing)}")
            tables = []
            for index, atom in enumerate(data.atom_ids):
                if request.atom_ids is not None and atom not in request.atom_ids:
                    continue
                if request.atom_ids is None and request.atom_types and data.elements[index] not in request.atom_types:
                    continue
                coordinates = np.empty((len(sources), 1, 3), dtype=float)
                with spool.path.open("rb") as raw:
                    for frame in range(len(sources)):
                        raw.seek((frame * len(data.atom_ids) + index) * 3 * 8)
                        coordinates[frame, 0] = np.fromfile(raw, dtype=np.float64, count=3)
                trace = replace(data, positions=coordinates,
                                atom_ids=[atom], elements=[data.elements[index]], atom_labels=None,
                                simulation=replace(data.simulation, atom_ids=[atom], elements=[data.elements[index]]))
                local = replace(request, frames=None, every=1, origin=origin, atom_ids=[atom], atom_types=None)
                table = self.run(trace, local).table
                table["x_source"] = axis_source
                tables.append(table)
                if reporter:
                    reporter("analyze", index + 1, len(data.atom_ids), "Fitting disk-backed atom traces")
            if tables:
                table = pd.concat(tables, ignore_index=True).sort_values("atom_id").reset_index(drop=True)
            else:
                table = pd.DataFrame(columns=DIFFUSIVITY_COLUMNS)
            return DiffusivityResult(table=table, request=request)
        finally:
            spool.close()

    @staticmethod
    def recommended_presentations(_result: DiffusivityResult, payload: dict[str, Any]) -> list[PresentationSpec]:
        """Build default table/plot presentations for diffusivity outputs.

        Works on
        -----
        Analyzer task output payloads

        Parameters
        -----
        _result : DiffusivityResult
            Analysis result object for the executed task.
        payload : dict[str, Any]
            Serialized result payload used by presentation dispatch.

        Returns
        -----
        list[PresentationSpec]
            Recommended table and grouped atom-diffusivity plot views.

        Examples
        -----
        ```python
        specs = DiffusivityTask.recommended_presentations(result, payload)
        ```
        Sample output:
        A list with a table view and a diffusivity-by-atom plot view.
        Meaning:
        UIs can render diffusivity results with default mappings.
        """
        table_rows = payload.get("table")
        if not isinstance(table_rows, list) or not table_rows:
            return [PresentationSpec(renderer="table", label="Table", view_type="table")]
        return [
            PresentationSpec(renderer="table", label="Table", view_type="table"),
            PresentationSpec(
                renderer="single_plot",
                label="Diffusivity by Atom",
                mapping={"x_col": "atom_id", "y_col": "diffusivity", "group_by_col": "atom_type"},
                options={"title": "Diffusivity by Atom", "xlabel": "atom_id", "ylabel": "diffusivity", "legend": True},
                view_type="plot2d",
            ),
        ]

    def run(self, data: TrajectoryData, request: DiffusivityRequest, reporter=None) -> DiffusivityResult:
        """Estimate per-atom diffusivity from frame-relative MSD trends."""

        out_cols = [
            "atom_id",
            "atom_type",
            "dim",
            "d",
            "x_source",
            "x_start",
            "x_end",
            "n_points",
            "slope_msd_per_x",
            "intercept",
            "diffusivity",
        ]

        if data.simulation is None or data.simulation.cell_lengths is None:
            raise ValueError("Diffusivity requires TrajectoryData.simulation.cell_lengths.")

        d_val = float(request.d)
        if not np.isfinite(d_val) or d_val <= 0.0:
            raise ValueError("d must be a positive finite number.")

        dims = tuple(d for d in request.dims if d in ("x", "y", "z"))
        if not dims:
            raise ValueError("dims must include at least one of 'x', 'y', or 'z'.")

        n_frames = data.positions.shape[0]
        if n_frames == 0:
            return DiffusivityResult(table=pd.DataFrame(columns=out_cols), request=request)

        msd_request = MSDRequest(
            atom_ids=request.atom_ids,
            atom_types=request.atom_types,
            dims=dims,
            origin=request.origin,
            frames=request.frames,
            every=request.every,
            unwrap=request.unwrap,
        )

        msd_result = MSDTask().run(data, msd_request, reporter=reporter)
        table_msd = msd_result.table

        if table_msd.empty:
            return DiffusivityResult(table=pd.DataFrame(columns=out_cols), request=request)

        x_all, x_source = _axis_source(data, n_frames)
        rows: list[dict[str, Any]] = []
        grouped = table_msd.sort_values(["atom_id", "frame_index"]).groupby("atom_id", sort=True)
        n_atoms = int(table_msd["atom_id"].nunique())
        for atom_progress, (atom_id, atom_table) in enumerate(grouped, start=1):
            frame_indices = atom_table["frame_index"].to_numpy(dtype=int)
            msd = atom_table["msd"].to_numpy(dtype=float)
            x = np.asarray([float(x_all[index]) for index in frame_indices], dtype=float)
            x -= float(x[0])
            valid = np.isfinite(x) & np.isfinite(msd)
            x_fit = x[valid]
            msd_fit = msd[valid]
            if x_fit.size < 2 or np.unique(x_fit).size < 2:
                continue
            slope, intercept = np.polyfit(x_fit, msd_fit, 1)
            rows.append(
                {
                    "atom_id": int(atom_id),
                    "atom_type": str(atom_table["atom_type"].iloc[0]),
                    "dim": str(atom_table["dim"].iloc[0]),
                    "d": d_val,
                    "x_source": x_source,
                    "x_start": float(np.min(x_fit)),
                    "x_end": float(np.max(x_fit)),
                    "n_points": int(x_fit.size),
                    "slope_msd_per_x": float(slope),
                    "intercept": float(intercept),
                    "diffusivity": float(slope / (2.0 * d_val)),
                }
            )
            if reporter:
                reporter("analyze", atom_progress, n_atoms, "Estimating diffusivity")

        table = pd.DataFrame(rows, columns=out_cols).sort_values("atom_id").reset_index(drop=True)

        if reporter:
            reporter("analyze", n_atoms, n_atoms, "Finished diffusivity")

        return DiffusivityResult(table=table, request=request)
