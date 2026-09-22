"""Backward-compatible frame-relative MSD task."""

import numpy as np
import pandas as pd

from reaxkit.analysis.base import AnalysisTask
from reaxkit.analysis.trajectory.msd import MSDRequest, MSDResult
from reaxkit.analysis.trajectory.pbc import maybe_unwrap_selected_positions
from reaxkit.domain.data_models import TrajectoryData


class MSDTask(AnalysisTask):
    """Compute the former per-atom displacement relative to an origin frame."""

    required_data = TrajectoryData

    def run(self, data: TrajectoryData, request: MSDRequest, reporter=None) -> MSDResult:
        columns = ["frame_index", "iter", "atom_id", "atom_type", "dim", "msd"]
        if data.simulation is None or data.simulation.cell_lengths is None:
            raise ValueError("MSD requires TrajectoryData.simulation.cell_lengths.")
        dimensions = tuple(dim for dim in request.dims if dim in ("x", "y", "z"))
        if not dimensions:
            raise ValueError("dims must include at least one of 'x', 'y', or 'z'.")

        frame_indices = (
            list(range(data.positions.shape[0]))
            if request.frames is None
            else [int(index) for index in request.frames]
        )
        frame_indices = frame_indices[::max(1, int(request.every))]
        if not frame_indices:
            return MSDResult(table=pd.DataFrame(columns=columns), request=request)
        origin = frame_indices[0] if request.origin == "first" else int(request.origin)
        if origin not in frame_indices:
            raise ValueError("origin must be 'first' or a frame index inside the selected frames")

        if request.atom_ids is not None:
            position_by_id = {int(atom_id): index for index, atom_id in enumerate(data.atom_ids)}
            missing = [int(atom_id) for atom_id in request.atom_ids if int(atom_id) not in position_by_id]
            if missing:
                raise ValueError(f"Requested atom_ids are not present in trajectory: {missing}")
            selected = [position_by_id[int(atom_id)] for atom_id in request.atom_ids]
        elif request.atom_types:
            wanted = {str(atom_type) for atom_type in request.atom_types}
            selected = [index for index, atom_type in enumerate(data.elements) if str(atom_type) in wanted]
        else:
            selected = list(range(data.positions.shape[1]))
        if not selected:
            return MSDResult(table=pd.DataFrame(columns=columns), request=request)

        axis = {"x": 0, "y": 1, "z": 2}
        coordinate_columns = [axis[dim] for dim in dimensions]
        coordinates = maybe_unwrap_selected_positions(
            data,
            frame_idx=frame_indices,
            sel_idx=selected,
            unwrap=bool(request.unwrap),
        )[:, :, coordinate_columns]
        reference = coordinates[frame_indices.index(origin)].astype(float)
        rows = []
        for progress, (frame_index, frame_coordinates) in enumerate(
            zip(frame_indices, coordinates, strict=False), start=1
        ):
            squared = np.sum((np.asarray(frame_coordinates, dtype=float) - reference) ** 2, axis=1)
            iteration = int(data.iterations[frame_index]) if data.iterations is not None else frame_index
            for selected_index, value in zip(selected, squared, strict=False):
                rows.append(
                    {
                        "frame_index": int(frame_index),
                        "iter": iteration,
                        "atom_id": int(data.atom_ids[selected_index]),
                        "atom_type": str(data.elements[selected_index]),
                        "dim": ",".join(dimensions),
                        "msd": float(value),
                    }
                )
            if reporter:
                reporter("analyze", progress, len(frame_indices), "Computing MSD")
        table = pd.DataFrame(rows, columns=columns).sort_values(["frame_index", "atom_id"]).reset_index(drop=True)
        return MSDResult(table=table, request=request)

__all__ = ["MSDRequest", "MSDResult", "MSDTask"]
