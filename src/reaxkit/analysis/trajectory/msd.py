"""Engine-agnostic MSD analysis task."""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from typing import Any, Optional, Sequence, Union

import numpy as np
import pandas as pd

from reaxkit.analysis.base import AnalysisTask
from reaxkit.core.analysis_task_registry import register_task
from reaxkit.domain.base_request import BaseRequest
from reaxkit.domain.base_result import BaseResult
from reaxkit.domain.data_models import TrajectoryData
from reaxkit.analysis.trajectory.pbc import maybe_unwrap_selected_positions
from reaxkit.presentation.specs import PresentationSpec


@dataclass
class MSDRequest(BaseRequest):
    """Request for MSD analysis."""

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
    origin: Union[str, int] = dc_field(
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
    unwrap: bool = dc_field(
        default=True,
        metadata={"label": "Unwrap PBC", "help": "Unwrap coordinates across periodic boundaries when cell data is available."},
    )


@dataclass
class MSDResult(BaseResult):
    """Result of MSD analysis.

    Output structure:
    - table: pandas.DataFrame with columns
      ['frame_index', 'iter', 'atom_id', 'atom_type', 'dim', 'msd']
      - frame_index: source frame index
      - iter: iteration for that frame
      - atom_id: atom identifier
      - atom_type: element/type label for the atom
      - dim: dimension selection label used for MSD (for example 'x,y,z')
        Note: when multiple dimensions are selected (for example x,y,z),
        MSD is the summed squared displacement across those dimensions.
      - msd: mean-squared displacement value
    - request: MSDRequest used to produce this result
    """

    table: pd.DataFrame
    request: MSDRequest


@register_task("get_msd", label="MSD")
class MSDTask(AnalysisTask):
    """Per-atom mean-squared displacement over selected frames/dimensions.

    When ``dims`` contains multiple axes (for example ``("x", "y", "z")``),
    the returned ``msd`` is the sum across those axes, and ``dim`` is stored
    as the joined label (for example ``"x,y,z"``).
    """

    required_data = TrajectoryData

    @staticmethod
    def recommended_presentations(_result: MSDResult, payload: dict[str, Any]) -> list[PresentationSpec]:
        """Common/default typed presentation specs for MSD."""
        table_rows = payload.get("table")
        if not isinstance(table_rows, list) or not table_rows:
            return [PresentationSpec(renderer="table", label="Table", view_type="table")]
        sample = table_rows[0] if isinstance(table_rows[0], dict) else {}
        x_axis = "iter" if "iter" in sample else "frame_index"
        group_by = "atom_id" if "atom_id" in sample else ""
        views: list[PresentationSpec] = [
            PresentationSpec(renderer="table", label="Table", view_type="table"),
            PresentationSpec(
                renderer="single_plot",
                label="MSD vs Time",
                mapping={
                    "x_col": x_axis,
                    "y_col": "msd",
                    "group_by_col": group_by,
                },
                options={
                    "title": "MSD vs Time",
                    "xlabel": x_axis,
                    "ylabel": "msd",
                    "legend": bool(group_by),
                },
                view_type="plot2d",
            ),
        ]
        return views

    def run(self, data: TrajectoryData, request: MSDRequest, reporter=None) -> MSDResult:
        out_cols = ["frame_index", "iter", "atom_id", "atom_type", "dim", "msd"]
        if data.simulation is None or data.simulation.cell_lengths is None:
            raise ValueError("MSD requires TrajectoryData.simulation.cell_lengths.")
        dims = tuple(d for d in request.dims if d in ("x", "y", "z"))
        if not dims:
            raise ValueError("dims must include at least one of 'x','y','z'")

        n_frames = data.positions.shape[0]
        if n_frames == 0:
            return MSDResult(table=pd.DataFrame(columns=out_cols), request=request)

        frame_idx = list(range(n_frames)) if request.frames is None else [int(i) for i in request.frames]
        step = max(1, int(request.every))
        frame_idx = frame_idx[::step]
        if not frame_idx:
            return MSDResult(table=pd.DataFrame(columns=out_cols), request=request)

        ref_frame = frame_idx[0] if request.origin == "first" else int(request.origin)
        if ref_frame not in frame_idx:
            raise ValueError("origin must be 'first' or a frame index inside the selected frames")

        if request.atom_ids is not None:
            sel_idx = [data.atom_ids.index(int(aid)) for aid in request.atom_ids]
        elif request.atom_types:
            tset = {str(t) for t in request.atom_types}
            sel_idx = [j for j, t in enumerate(data.elements) if str(t) in tset]
        else:
            sel_idx = list(range(data.positions.shape[1]))

        if not sel_idx:
            return MSDResult(table=pd.DataFrame(columns=out_cols), request=request)

        axes = {"x": 0, "y": 1, "z": 2}
        use_cols = [axes[d] for d in dims]

        atom_ids = [data.atom_ids[i] for i in sel_idx]
        atom_types = [str(data.elements[i]) for i in sel_idx]
        dim_label = ",".join(dims)

        coords_series = maybe_unwrap_selected_positions(
            data,
            frame_idx=frame_idx,
            sel_idx=sel_idx,
            unwrap=bool(request.unwrap),
        )[:, :, use_cols]
        ref_i = int(frame_idx.index(ref_frame))
        r0 = coords_series[ref_i].astype(float)
        rows: list[dict] = []
        n_steps = len(frame_idx)

        for step_i, (i, coords) in enumerate(zip(frame_idx, coords_series, strict=False), start=1):
            coords = np.asarray(coords, dtype=float)
            dr = coords - r0
            sq = np.sum(dr * dr, axis=1)

            iter_val = int(data.iterations[i]) if data.iterations is not None else int(i)
            for atom_id, atom_type, msd_val in zip(atom_ids, atom_types, sq):
                rows.append(
                    {
                        "frame_index": int(i),
                        "iter": iter_val,
                        "atom_id": int(atom_id),
                        "atom_type": atom_type,
                        "dim": dim_label,
                        "msd": float(msd_val),
                    }
                )
            if reporter:
                reporter("analyze", step_i, n_steps, "Computing MSD")

        table = pd.DataFrame(rows).sort_values(["frame_index", "atom_id"]).reset_index(drop=True)
        if reporter:
            reporter("analyze", n_steps, n_steps, "Finished MSD")
        return MSDResult(table=table, request=request)
