"""
Example: basic xmolout analysis with current ReaxKit tasks (no CLI).

This script demonstrates how to:
1) Load an xmolout file with XmoloutHandler
2) Build TrajectoryData via ReaxFF adapter normalizer
3) Extract trajectory coordinate series for one atom/dimension
4) Compute MSD for selected atom(s)
5) Extract cell dimensions from simulation metadata
6) Export results to CSV
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from reaxkit.analysis.timeseries import (
    CellDimensionsRequest,
    CellDimensionsTask,
    TrajectoryCoordinateSeriesRequest,
    TrajectoryCoordinateSeriesTask,
)
from reaxkit.analysis.trajectory.msd import MSDRequest, MSDTask
from reaxkit.engine.reaxff.adapter import trajectory_from_xmolout_handler
from reaxkit.engine.reaxff.io.xmolout_handler import XmoloutHandler


HERE = Path(__file__).resolve().parent
DATA = HERE / "data"

XMOL_PATH = DATA / "small_xmolout"
ATOM_IDS = [1]  # 1-based indexing
FRAMES_SAMPLE = list(range(0, 50, 5))

OUTDIR = Path("reaxkit_outputs/examples/xmolout_basic")
OUTDIR.mkdir(parents=True, exist_ok=True)


def _print_head(df: pd.DataFrame, name: str, n: int = 5) -> None:
    print(f"\n=== {name} (head {n}) ===")
    if df.empty:
        print("(empty)")
    else:
        print(df.head(n).to_string(index=False))


def main() -> None:
    if not XMOL_PATH.exists():
        raise FileNotFoundError(f"Missing xmolout sample file: {XMOL_PATH}")

    xh = XmoloutHandler(str(XMOL_PATH))
    xdf = xh.dataframe()
    print(f"Loaded xmolout frames: {len(xdf)}")
    _print_head(xdf, "xmolout summary dataframe")

    traj_data = trajectory_from_xmolout_handler(xh)

    coords = TrajectoryCoordinateSeriesTask().run(
        traj_data,
        TrajectoryCoordinateSeriesRequest(
            atom_ids=ATOM_IDS,
            dims=("z",),
            frames=FRAMES_SAMPLE,
            every=1,
        ),
    ).table
    _print_head(coords, "atom trajectory coordinate series (z)")
    coords.to_csv(OUTDIR / "atom1_z_trajectory.csv", index=False)

    msd = MSDTask().run(
        traj_data,
        MSDRequest(
            atom_ids=ATOM_IDS,
            dims=("z",),
            frames=FRAMES_SAMPLE,
            every=1,
            origin="first",
            unwrap=True,
        ),
    ).table
    _print_head(msd, "MSD (z)")
    msd.to_csv(OUTDIR / "atom1_z_msd.csv", index=False)

    if traj_data.simulation is not None:
        cell = CellDimensionsTask().run(
            traj_data.simulation,
            CellDimensionsRequest(
                fields=("a", "b", "c", "alpha", "beta", "gamma"),
                frames=FRAMES_SAMPLE,
                every=1,
            ),
        ).table
        _print_head(cell, "cell dimensions")
        cell.to_csv(OUTDIR / "cell_dimensions.csv", index=False)
    else:
        print("\nNo simulation metadata available; skipped cell-dimension extraction.")

    print(f"\nDone. Results written to:\n  {OUTDIR.resolve()}")


if __name__ == "__main__":
    main()
