"""Compute mean-squared displacement (MSD) from trajectory coordinates.

This module implements an engine-agnostic analyzer task that computes per-atom
MSD over selected frames and coordinate dimensions, with optional periodic
boundary unwrapping before displacement evaluation. It is scoped to displacement
series generation and does not estimate diffusion coefficients directly.

**Usage context**

- Trajectory analysis: Quantify per-atom displacement growth over time.
- Diffusion workflows: Export MSD tables for downstream fitting and plotting.
- Comparative studies: Filter by atom IDs or atom types within one run.

Notes
-----
Diffusivity estimation is implemented in `reaxkit.analysis.trajectory.diffusivity`
and reuses this module's MSD outputs.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from typing import Any, Optional, Sequence, Union

import numpy as np
import pandas as pd

from reaxkit.analysis.base import AnalysisTask
from reaxkit.core.registry.analysis_task_registry import register_task
from reaxkit.domain.base_request import BaseRequest
from reaxkit.domain.base_result import BaseResult
from reaxkit.domain.data_models import TrajectoryData
from reaxkit.analysis.trajectory.pbc import maybe_unwrap_selected_positions
from reaxkit.presentation.specs import PresentationSpec


@dataclass
class MSDRequest(BaseRequest):
    """Request payload for MSD analysis.

    Defines frame, atom-selection, dimensionality, and unwrapping controls used
    by the MSD task to compute displacement values from trajectory data.

    Fields
    -----
    atom_ids : Optional[list[int]]
        Atom IDs to include. When set, this selection takes precedence over
        `atom_types`. Default is `None` (not explicitly selected).
    atom_types : Optional[list[str]]
        Atom type/element filters used when `atom_ids` is empty. Default is
        `None`.
    dims : Sequence[str]
        Coordinate axes included in MSD (`x`, `y`, `z`). Default is
        `("x", "y", "z")`.
    origin : Union[str, int]
        Reference frame selector. Use `"first"` for the first selected frame,
        or provide an explicit frame index. Default is `"first"`.
    frames : Optional[Sequence[int]]
        Frame indices to evaluate. `None` means all frames in trajectory order.
    every : int
        Stride over selected frames; must be >= 1. Default is `1`.
    unwrap : bool
        Whether to unwrap coordinates across periodic boundaries when cell data
        is available. Default is `True`.

    Examples
    -----
    Sample request payload/object:
    `MSDRequest(atom_types=["O"], dims=("x", "y", "z"), frames=[0, 10, 20], every=1, origin="first", unwrap=True)`
    This sample computes oxygen-atom MSD on selected frames using full 3D
    displacement from the first selected frame as reference.
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
    max_lag: Optional[int] = dc_field(
        default=None,
        metadata={
            "label": "Maximum lag",
            "help": "Maximum lag time in number of frames. If empty, all selected frames are used.",
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
class MSDResult(BaseResult):
    """Result payload for MSD analysis.

    Stores the computed MSD table together with the request used to generate
    it, so downstream consumers can preserve analysis provenance.

    Fields
    -----
    table : pd.DataFrame
        Output table with columns `["frame_index", "iter", "atom_id",
        "atom_type", "dim", "msd"]`.
        - `frame_index`: source frame index.
        - `iter`: simulation iteration for that frame.
        - `atom_id`: atom identifier.
        - `atom_type`: element/type label for the atom.
        - `dim`: dimension selection label (for example `"x,y,z"`).
        - `msd`: mean-squared displacement value.
    request : MSDRequest
        Request object used for this analysis run.

    Notes
    -----
    When multiple dimensions are selected (for example `x,y,z`), `msd` is the
    summed squared displacement across those dimensions.

    Examples
    -----
    Sample output payload/object:
    `MSDResult(table=<DataFrame rows with frame_index/iter/atom_id/atom_type/dim/msd>, request=<MSDRequest ...>)`
    The output table represents per-atom MSD values at each evaluated frame,
    with `dim` indicating which coordinate components were included.
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
        """Build default presentation specs for MSD outputs.

        Selects a table view and a 2D plot mapping based on available payload
        columns, preferring `iter` as x-axis when present.

        Works on
        -----
        Analyzer task output payloads

        Parameters
        -----
        _result : MSDResult
            Analysis result object for the executed task.
        payload : dict[str, Any]
            Serialized result payload used by presentation dispatch.

        Returns
        -----
        list[PresentationSpec]
            Recommended table and plot presentation specifications.

        Examples
        -----
        >>> specs = MSDTask.recommended_presentations(result, payload)
        >>> len(specs) >= 1
        True
        Sample output meaning: returned specs describe how MSD results should be
        rendered (table view and, when possible, an MSD-vs-time plot).
        """
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
        """Run time-origin averaged MSD analysis.

        This implementation follows the Vale 2005 Fortran-style MSD algorithm:

            MSD(lag) = average over atoms and all valid time origins of
                       |r(t + lag) - r(t)|^2

        Unlike a simple MSD relative to the first frame, this method averages over
        multiple time origins, which gives smoother and more statistically stable
        MSD curves.

        Expected optional request attributes
        ------------------------------------
        request.max_lag : int, optional
            Maximum lag in number of frames. If missing, uses all selected frames.
        request.delta_t_ps : float, optional
            Time spacing between selected trajectory frames in ps. If missing,
            uses 1.0 ps.
        """

        out_cols = ["lag_frame", "time_ps", "msd"]

        if data.simulation is None or data.simulation.cell_lengths is None:
            raise ValueError("MSD requires TrajectoryData.simulation.cell_lengths.")

        dims = tuple(d for d in request.dims if d in ("x", "y", "z"))
        if not dims:
            raise ValueError("dims must include at least one of 'x', 'y', or 'z'.")

        n_frames = data.positions.shape[0]
        if n_frames == 0:
            return MSDResult(table=pd.DataFrame(columns=out_cols), request=request)

        frame_idx = (
            list(range(n_frames))
            if request.frames is None
            else [int(i) for i in request.frames]
        )

        step = max(1, int(request.every))
        frame_idx = frame_idx[::step]

        if not frame_idx:
            return MSDResult(table=pd.DataFrame(columns=out_cols), request=request)

        for i in frame_idx:
            if i < 0 or i >= n_frames:
                raise IndexError(f"Frame index {i} is outside trajectory range 0-{n_frames - 1}.")

        if request.atom_ids is not None:
            atom_id_to_pos = {int(atom_id): j for j, atom_id in enumerate(data.atom_ids)}
            missing = [int(aid) for aid in request.atom_ids if int(aid) not in atom_id_to_pos]
            if missing:
                raise ValueError(f"Requested atom_ids are not present in trajectory: {missing}")
            sel_idx = [atom_id_to_pos[int(aid)] for aid in request.atom_ids]

        elif request.atom_types:
            tset = {str(t) for t in request.atom_types}
            sel_idx = [j for j, t in enumerate(data.elements) if str(t) in tset]

        else:
            sel_idx = list(range(data.positions.shape[1]))

        if not sel_idx:
            return MSDResult(table=pd.DataFrame(columns=out_cols), request=request)

        axes = {"x": 0, "y": 1, "z": 2}
        use_cols = [axes[d] for d in dims]

        coords_series = maybe_unwrap_selected_positions(
            data,
            frame_idx=frame_idx,
            sel_idx=sel_idx,
            unwrap=bool(request.unwrap),
        )[:, :, use_cols].astype(float)

        n_selected_frames = coords_series.shape[0]
        n_selected_atoms = coords_series.shape[1]

        # print("Selected atom indices:", sel_idx)
        # print("Selected atom ids:", [data.atom_ids[i] for i in sel_idx])
        # print("Selected atom types:", [data.elements[i] for i in sel_idx])
        # print("First-frame selected coords:")
        # print(coords_series[0])
        # print("Second-frame selected coords:")
        # print(coords_series[1])

        # Equivalent to Fortran ndim.
        # Prefer request.max_lag if you add it to MSDRequest.
        max_lag = int(getattr(request, "max_lag", n_selected_frames))

        if max_lag <= 0:
            raise ValueError("max_lag must be positive.")

        max_lag = min(max_lag, n_selected_frames)

        # Equivalent to Fortran deltat.
        # Prefer request.delta_t_ps if you add it to MSDRequest.
        delta_t_ps = float(getattr(request, "delta_t_ps", 1.0))

        rows: list[dict[str, float | int]] = []

        for lag in range(max_lag):
            # For lag = 0, compare each frame with itself.
            # For lag = 1, compare frame t+1 with frame t, etc.
            r_later = coords_series[lag:, :, :]
            r_earlier = coords_series[: n_selected_frames - lag, :, :]

            dr = r_later - r_earlier
            squared_displacements = np.sum(dr * dr, axis=2)

            # Average over all valid time origins and all selected atoms.
            msd = float(np.mean(squared_displacements))

            time_ps = float(lag * delta_t_ps)

            rows.append(
                {
                    "lag_frame": int(lag),
                    "time_ps": time_ps,
                    "msd": msd,
                }
            )

            if reporter:
                reporter("analyze", lag + 1, max_lag, "Computing time-origin averaged MSD")

        table = pd.DataFrame(rows, columns=out_cols)

        if reporter:
            reporter("analyze", max_lag, max_lag, "Finished MSD")

        return MSDResult(table=table, request=request)
