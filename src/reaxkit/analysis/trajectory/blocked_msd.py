"""Disk-backed time-origin MSD with bounded coordinate working sets."""

from contextlib import closing
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from reaxkit.analysis.trajectory.pbc import _cell_matrix
from reaxkit.core.runtime.frame_tables import selected_frame_envelopes


def time_origin_msd(frames, request, pipeline, reporter=None):
    axes = [{"x": 0, "y": 1, "z": 2}[dim] for dim in request.dims if dim in {"x", "y", "z"}]
    if not axes:
        raise ValueError("dims must include at least one of 'x', 'y', or 'z'.")
    columns = ["lag_frame", "time_ps", "msd"]
    selected = None
    finite = None
    count = 0
    identity = None
    with TemporaryDirectory(prefix="reaxkit-msd-") as directory:
        raw_path = Path(directory) / "raw.bin"
        cells_path = Path(directory) / "cells.bin"
        coordinates_path = Path(directory) / "coordinates.bin"
        with raw_path.open("wb") as raw, cells_path.open("wb") as cells:
            with closing(pipeline.iter_blocks(selected_frame_envelopes(frames, request))) as blocks:
                for block in blocks:
                    for envelope in block:
                        data = envelope.payload
                        if data.simulation is None or data.simulation.cell_lengths is None:
                            raise ValueError("MSD requires TrajectoryData.simulation.cell_lengths.")
                        if selected is None:
                            identity = tuple(data.atom_ids)
                            by_id = {int(atom): i for i, atom in enumerate(data.atom_ids)}
                            if request.atom_ids is not None:
                                missing = [int(atom) for atom in request.atom_ids if int(atom) not in by_id]
                                if missing:
                                    raise ValueError(f"Requested atom_ids are not present in trajectory: {missing}")
                                selected = [by_id[int(atom)] for atom in request.atom_ids]
                            elif request.atom_types:
                                selected = [i for i, element in enumerate(data.elements) if str(element) in set(request.atom_types)]
                            else:
                                selected = list(range(len(data.atom_ids)))
                            finite = np.ones(len(selected), dtype=bool)
                        elif tuple(data.atom_ids) != identity:
                            raise ValueError("MSD atom identity/order changed between frames.")
                        positions = np.asarray(data.positions[0, selected, :], dtype=np.float64)
                        finite &= np.isfinite(positions).all(axis=1)
                        positions.tofile(raw)
                        lengths = np.asarray(data.simulation.cell_lengths[0], dtype=np.float64)
                        angles = np.asarray(data.simulation.cell_angles[0], dtype=np.float64) if data.simulation.cell_angles is not None else np.full(3, 90.0)
                        np.concatenate((lengths, angles)).tofile(cells)
                        count += 1
        if not count or not selected:
            return pd.DataFrame(columns=columns)
        atoms = len(selected)
        maximum = count if request.max_lag is None else int(request.max_lag)
        if maximum <= 0:
            raise ValueError("max_lag must be positive.")
        maximum = min(maximum, count)

        # Use exactly the serial minimum-image convention (previous frame's
        # cell). Atoms with any nonfinite coordinate keep their raw trajectory.
        previous_fractional = previous_cell = unwrapped = None
        with raw_path.open("rb") as raw, cells_path.open("rb") as cells, coordinates_path.open("wb") as output:
            for index in range(count):
                coordinates = np.fromfile(raw, dtype=np.float64, count=atoms * 3).reshape(atoms, 3)
                cell_values = np.fromfile(cells, dtype=np.float64, count=6)
                if request.unwrap:
                    cell = _cell_matrix(cell_values[:3], cell_values[3:])
                    fractional = coordinates @ np.linalg.inv(cell)
                    if index == 0:
                        unwrapped = coordinates.copy()
                    else:
                        displacement = fractional - previous_fractional
                        displacement -= np.round(displacement)
                        unwrapped[finite] += (displacement @ previous_cell)[finite]
                        unwrapped[~finite] = coordinates[~finite]
                    previous_fractional, previous_cell = fractional, cell
                    coordinates = unwrapped
                np.ascontiguousarray(coordinates[:, axes]).tofile(output)

        frame_values = atoms * len(axes)
        frame_bytes = frame_values * 8
        block_size = max(1, pipeline.policy.max_in_flight)
        # Keep temporary displacement arrays below a quarter of the budget.
        if pipeline.policy.memory_limit_bytes:
            block_size = min(block_size, max(1, pipeline.policy.memory_limit_bytes // (16 * frame_bytes)))
        rows = []
        with coordinates_path.open("rb") as earlier, coordinates_path.open("rb") as later:
            for lag in range(maximum):
                earlier.seek(0)
                later.seek(lag * frame_bytes)
                total = compensation = 0.0
                for start in range(0, count - lag, block_size):
                    size = min(block_size, count - lag - start)
                    first = np.fromfile(earlier, dtype=np.float64, count=size * frame_values).reshape(size, atoms, len(axes))
                    second = np.fromfile(later, dtype=np.float64, count=size * frame_values).reshape(size, atoms, len(axes))
                    delta = second - first
                    contribution = float(np.sum(np.sum(delta * delta, axis=2)))
                    corrected = contribution - compensation
                    updated = total + corrected
                    compensation = (updated - total) - corrected
                    total = updated
                rows.append((lag, lag * float(request.delta_t_ps), total / ((count - lag) * atoms)))
                if reporter:
                    reporter("analyze", lag + 1, maximum, "Computing blocked time-origin MSD")
        return pd.DataFrame(rows, columns=columns)
