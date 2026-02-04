"""
Example: Direct use of xmolout_analyzer (no CLI).

This script demonstrates how to:
1. Load an xmolout file with XmoloutHandler
2. Extract atom trajectories
3. Compute mean squared displacement (MSD)
4. Inspect box / thermodynamic information

"""

from pathlib import Path

import pandas as pd

from reaxkit.io.handlers.xmolout_handler import XmoloutHandler
from reaxkit.analysis.per_file.xmolout_analyzer import (
    get_atom_trajectories,
    get_mean_squared_displacement,
    get_unit_cell_dimensions_across_frames,
    get_atom_type_mapping,
)


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

XMOL_PATH = Path("data/small_xmolout")  # relative to docs/examples/
FRAMES = slice(0, 50)                   # first 50 frames
ATOM_ID = [1]                           # 1-based atom indexing
DIMS = ("z",)


# ---------------------------------------------------------
# Load xmolout
# ---------------------------------------------------------

xh = XmoloutHandler(XMOL_PATH)

print(f"Loaded xmolout with {len(xh.dataframe())} frames")
print()


# ---------------------------------------------------------
# Inspect atom types
# ---------------------------------------------------------

type_info = get_atom_type_mapping(xh, frame=0)

print("Atom types in frame 0:")
for t, ids in type_info["type_to_indices"].items():
    print(f"  {t}: {len(ids)} atoms")
print()


# ---------------------------------------------------------
# Extract atom trajectories
# ---------------------------------------------------------

traj = get_atom_trajectories(
    xh,
    frames=FRAMES,
    atoms=ATOM_ID,
    dims=DIMS,
    format="long",
)

print("Trajectory preview:")
print(traj.head())
print()


# ---------------------------------------------------------
# Compute mean squared displacement (MSD)
# ---------------------------------------------------------

msd = get_mean_squared_displacement(
    xh,
    frames=FRAMES,
    atoms=ATOM_ID,
    dims=DIMS,
)

print("MSD preview:")
print(msd.head())
print()


# ---------------------------------------------------------
# Extract box / thermodynamic data
# ---------------------------------------------------------

box = get_unit_cell_dimensions_across_frames(
    xh,
    frames=FRAMES,
)

print("Box / thermodynamic data:")
print(box.head())
print()


# ---------------------------------------------------------
# Optional: export results
# ---------------------------------------------------------

outdir = Path("reaxkit_outputs/examples")
outdir.mkdir(parents=True, exist_ok=True)

traj.to_csv(outdir / "atom1_z_trajectory.csv", index=False)
msd.to_csv(outdir / "atom1_z_msd.csv", index=False)
box.to_csv(outdir / "box_dimensions.csv", index=False)

print(f"Results written to: {outdir.resolve()}")
