from __future__ import annotations

from pathlib import Path

import numpy as np

from reaxkit.domain.data_models import ConnectivityData, TrajectoryData
from reaxkit.engine.lammps.adapter import LAMMPSAdapter
from reaxkit.engine.lammps.dump_handler import LAMMPSDumpHandler


def _xyz_dump(n_frames: int) -> str:
    blocks = []
    for frame in range(n_frames):
        blocks.append(
            "\n".join(
                [
                    "2",
                    f"LAMMPS timestep: {frame * 10}",
                    f"Al {frame}.0 0 0",
                    f"N {frame + 1}.0 0 0",
                ]
            )
        )
    return "\n".join(blocks) + "\n"


def _item_dump(n_frames: int) -> str:
    blocks = []
    for frame in range(n_frames):
        blocks.append(
            "\n".join(
                [
                    "ITEM: TIMESTEP",
                    str(frame * 10),
                    "ITEM: NUMBER OF ATOMS",
                    "2",
                    "ITEM: BOX BOUNDS pp pp pp",
                    "0 10",
                    "0 11",
                    "0 12",
                    "ITEM: ATOMS id element x y z",
                    f"1 Al {frame}.0 0 0",
                    f"2 N {frame + 1}.0 0 0",
                ]
            )
        )
    return "\n".join(blocks) + "\n"


def test_xyz_dump_streams_only_selected_frames(tmp_path: Path):
    path = tmp_path / "dump.xyz"
    path.write_text(_xyz_dump(6), encoding="utf-8")

    handler = LAMMPSDumpHandler(path, frame_indices=[4, 0, 2])

    assert handler.dataframe()["iter"].tolist() == [40, 0, 20]
    assert handler.metadata()["source_frame_indices"] == [4, 0, 2]
    assert handler.frame(0)["source_index"] == 4
    assert handler.frame(0)["coords"][0, 0] == 4.0


def test_item_dump_streams_only_selected_frames(tmp_path: Path):
    path = tmp_path / "dump.lammpstrj"
    path.write_text(_item_dump(5), encoding="utf-8")

    handler = LAMMPSDumpHandler(path, frame_indices=[3, 1])

    assert handler.dataframe()["iter"].tolist() == [30, 10]
    assert handler.metadata()["source_frame_indices"] == [3, 1]
    assert handler.frame(0)["box_bounds"] == [(0.0, 10.0), (0.0, 11.0), (0.0, 12.0)]


def test_lammps_adapter_preserves_selected_source_frames(tmp_path: Path):
    path = tmp_path / "dump.xyz"
    path.write_text(_xyz_dump(5), encoding="utf-8")

    trajectory = LAMMPSAdapter().load(
        TrajectoryData,
        {"dump": str(path), "frames": [4, 1]},
    )

    assert trajectory.positions.shape == (2, 2, 3)
    assert trajectory.iterations.tolist() == [40, 10]
    assert trajectory.source_frame_indices.tolist() == [4, 1]


def test_lammps_connectivity_placeholder_is_aligned_to_selected_frames():
    connectivity = LAMMPSAdapter().load(
        ConnectivityData,
        {"frames": [5, 1]},
    )

    assert connectivity.sum_bond_orders.shape == (2, 0)
    assert connectivity.source_frame_indices.tolist() == [5, 1]
    assert np.array_equal(connectivity.iterations, connectivity.source_frame_indices)
