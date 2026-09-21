"""CSV tables and variable guides for three-folded wurtzite workflows."""

from __future__ import annotations

import re
from pathlib import Path

from reaxkit.analysis.ferroelectrics.four_folded_wurtzite.neighbors import time_after_iter
from reaxkit.analysis.ferroelectrics.three_folded_wurtzite.neighbors import (
    centers_csv_table,
    neighbors_csv_table,
)
from reaxkit.analysis.ferroelectrics.three_folded_wurtzite.polarity import polarity_csv_table
from reaxkit.workflows.ferroelectrics.four_folded_wurtzite.artifacts import (
    VARIABLE_DESCRIPTIONS as _FOUR_FOLD_DESCRIPTIONS,
)

VARIABLE_DESCRIPTIONS = {
    **_FOUR_FOLD_DESCRIPTIONS,
    "neighbor_count_within_cutoff": "Number of unique selected neighbor-species candidates inside the cutoff; candidates are not capped at four.",
    "has_three_basal_neighbors": "True when three candidates were available for the basal-plane assignment, which is sufficient for polarity.",
    "has_apical_neighbor": "True when a remaining candidate lies on the side required by the already-calculated polarity.",
    "neighbor_rank": "Distance rank among all unique candidates inside the cutoff, where 1 is nearest.",
    "neighbor_role": "Final role: basal, polarity-consistent apical, or ignored candidate.",
    "apical_neighbor_rank": "Distance rank of the selected polarity-consistent apical candidate; zero means none was selected.",
    "delta (angstrom)": "Three-folded geometric displacement, defined as the negative signed mean c-axis projection of the three basal bonds.",
    "delta_eff (angstrom)": "Signed basal-only displacement used to assign polarity; it is identical to delta.",
    "polarity": "Discrete polarity from basal-only delta: +1 above tolerance, -1 below negative tolerance, and 0 otherwise or without three basal neighbors.",
    "n_complete_sites": "Number of sites with three assigned basal neighbors.",
    "n_incomplete_sites": "Number of sites with fewer than three assigned basal neighbors.",
    "n_sites_with_apical_neighbor": "Number of complete sites with a polarity-consistent apical neighbor.",
}


def _neighbor_wide_description(column: str) -> str | None:
    match = re.fullmatch(r"neighbor_(\d+)_(.+)", column)
    if match is None:
        return None
    rank, field = match.groups()
    descriptions = {
        "atom_id": "persistent atom identifier",
        "element": "unmodified chemical symbol or atom label",
        "charge (e)": "charge in elementary-charge units",
        "distance (angstrom)": "minimum-image distance in angstrom",
        "bond_x (angstrom)": "center-to-neighbor x component in angstrom",
        "bond_y (angstrom)": "center-to-neighbor y component in angstrom",
        "bond_z (angstrom)": "center-to-neighbor z component in angstrom",
        "bond_c (angstrom)": "projection onto the normalized c-axis in angstrom",
        "role": "basal, apical, or ignored role",
    }
    description = descriptions.get(field)
    return None if description is None else f"Wide-column copy of the {description} for distance-ranked candidate {rank}."


def variable_description(column: str) -> str:
    description = VARIABLE_DESCRIPTIONS.get(column) or _neighbor_wide_description(column)
    if description is None:
        raise KeyError(f"No three-folded polarity variable description is defined for {column!r}.")
    return description


def write_polarity_variable_guide(result, output: Path) -> Path:
    frames = (
        centers_csv_table(result.centers), neighbors_csv_table(result.neighbor_geometry),
        polarity_csv_table(result.table), time_after_iter(result.summary),
        time_after_iter(result.proton_summary),
    )
    columns = list(dict.fromkeys(column for frame in frames for column in frame.columns))
    lines = [
        "Three-folded wurtzite polarity variable guide", "==============================================",
        "", "Polarity uses the negative signed mean projection of three basal bonds. Apical assignment is performed afterward and must agree with that polarity.",
        "", "Variables", "---------",
    ]
    for column in columns:
        lines.extend(("", column, f"  {variable_description(str(column))}"))
    path = output / "polarity_variables.txt"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_polarity_tables(
    result,
    output: Path,
    *,
    complete_only: bool = False,
    all_csvs_in_helpful_data: bool = False,
) -> dict[str, Path]:
    output.mkdir(parents=True, exist_ok=True)
    helpful = output / "other_helpful_data"
    helpful.mkdir(parents=True, exist_ok=True)
    primary = helpful if all_csvs_in_helpful_data else output
    paths = {
        "centers": helpful / "centers.csv", "neighbors": helpful / "neighbors.csv",
        "polarity": primary / "polarity.csv", "summary": primary / "polarity_summary.csv",
        "proton": primary / "proton_proximity_summary.csv",
        "variables": output / "polarity_variables.txt",
    }
    centers_csv_table(result.centers).to_csv(paths["centers"], index=False)
    neighbors_csv_table(result.neighbor_geometry).to_csv(paths["neighbors"], index=False)
    polarity = result.table[
        result.table["has_three_basal_neighbors"].astype(bool)
    ] if complete_only else result.table
    polarity_csv_table(polarity).to_csv(paths["polarity"], index=False)
    time_after_iter(result.summary).to_csv(paths["summary"], index=False)
    time_after_iter(result.proton_summary).to_csv(paths["proton"], index=False)
    write_polarity_variable_guide(result, output)
    return paths


__all__ = [
    "VARIABLE_DESCRIPTIONS", "variable_description", "write_polarity_tables",
    "write_polarity_variable_guide",
]
