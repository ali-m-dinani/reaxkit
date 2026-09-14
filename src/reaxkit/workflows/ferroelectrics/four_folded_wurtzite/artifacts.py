"""Shared CSV artifact writers for four-fold wurtzite workflows."""

from __future__ import annotations

import re
from pathlib import Path

from reaxkit.analysis.ferroelectrics.four_folded_wurtzite.neighbors import (
    centers_csv_table,
    neighbors_csv_table,
    time_after_iter,
)
from reaxkit.analysis.ferroelectrics.four_folded_wurtzite.polarity import (
    polarity_csv_table,
)

VARIABLE_DESCRIPTIONS = {
    "frame_index": "Zero-based source trajectory frame represented by this row.",
    "iter": "Simulation iteration associated with the source trajectory frame.",
    "time": "Simulation time corresponding to iter, calculated from trajectory metadata or the control-file time conversion when available.",
    "site_atom_index": "Zero-based array index of the center atom within the trajectory frame.",
    "site_atom_id": "Persistent atom identifier of the center site, normally the one-based atom number from the simulation.",
    "site_element": "Unmodified chemical symbol or atom label of the center site.",
    "site_x (angstrom)": "Cartesian x coordinate of the center atom in angstrom.",
    "site_y (angstrom)": "Cartesian y coordinate of the center atom in angstrom.",
    "site_z (angstrom)": "Cartesian z coordinate of the center atom in angstrom.",
    "site_charge (e)": "Dynamic or explicitly supplied formal charge of the center atom in elementary-charge units.",
    "neighbor_count_within_cutoff": "Number of selected neighbor-species atoms found within the configured distance cutoff, capped at four in the stored assignment.",
    "has_four_neighbors": "True when the center has a complete four-neighbor environment within the cutoff.",
    "has_proton_within_cutoff": "True when at least one selected proton-like atom lies within the proton cutoff of the center using the requested periodic boundaries.",
    "proton_cutoff (angstrom)": "Center-to-proton distance threshold used for the proton-proximity classification, in angstrom.",
    "charge_source": "Charge data used for the row: per-atom dynamic charges or explicit formal charges.",
    "neighbor_cutoff (angstrom)": "Maximum center-to-neighbor distance allowed during assignment, in angstrom; a missing value means no cutoff.",
    "c_axis_x": "Cartesian x component of the normalized polar c-axis used for bond projections.",
    "c_axis_y": "Cartesian y component of the normalized polar c-axis used for bond projections.",
    "c_axis_z": "Cartesian z component of the normalized polar c-axis used for bond projections.",
    "periodic_a": "Whether periodic minimum-image wrapping is enabled along the first lattice vector.",
    "periodic_b": "Whether periodic minimum-image wrapping is enabled along the second lattice vector.",
    "periodic_c": "Whether periodic minimum-image wrapping is enabled along the third lattice vector.",
    "neighbor_rank": "Distance rank of this neighbor around its center, where 1 is the nearest and 4 is the farthest assigned neighbor.",
    "neighbor_atom_index": "Zero-based array index of the assigned neighbor within the trajectory frame.",
    "neighbor_atom_id": "Persistent atom identifier of the assigned neighbor.",
    "neighbor_element": "Unmodified chemical symbol or atom label of the assigned neighbor.",
    "neighbor_charge (e)": "Dynamic or explicitly supplied formal charge of the assigned neighbor in elementary-charge units.",
    "neighbor_x (angstrom)": "Original Cartesian x coordinate of the neighbor in angstrom.",
    "neighbor_y (angstrom)": "Original Cartesian y coordinate of the neighbor in angstrom.",
    "neighbor_z (angstrom)": "Original Cartesian z coordinate of the neighbor in angstrom.",
    "neighbor_image_x (angstrom)": "Cartesian x coordinate of the periodic image used for the minimum-image center-neighbor bond.",
    "neighbor_image_y (angstrom)": "Cartesian y coordinate of the periodic image used for the minimum-image center-neighbor bond.",
    "neighbor_image_z (angstrom)": "Cartesian z coordinate of the periodic image used for the minimum-image center-neighbor bond.",
    "distance (angstrom)": "Minimum-image center-to-neighbor distance in angstrom.",
    "bond_x (angstrom)": "Cartesian x component of the vector from the center to the selected neighbor image.",
    "bond_y (angstrom)": "Cartesian y component of the vector from the center to the selected neighbor image.",
    "bond_z (angstrom)": "Cartesian z component of the vector from the center to the selected neighbor image.",
    "bond_c (angstrom)": "Projection of the center-to-neighbor bond vector onto the normalized polar c-axis; this equals bond_z only when the c-axis is Cartesian z.",
    "neighbor_role": "Geometric classification of the neighbor as apical, basal, or unassigned for an incomplete site.",
    "apical_neighbor_rank": "Distance rank of the neighbor selected as apical; the apical neighbor has the largest absolute bond_c projection.",
    "apical_neighbor_atom_id": "Persistent atom identifier of the neighbor selected as apical.",
    "apical_bond_c (angstrom)": "Signed c-axis projection of the apical center-neighbor bond in angstrom.",
    "mean_basal_bond_c (angstrom)": "Arithmetic mean of the signed c-axis projections of the three basal bonds.",
    "mean_basal_bond_c_change_from_frame_0 (angstrom)": "Mean basal bond projection for this site minus the value for the same site in the configured reference frame, which defaults to frame 0.",
    "delta (angstrom)": "Geometric polar displacement defined as apical_bond_c minus mean_basal_bond_c.",
    "delta_eff (angstrom)": "Signed effective geometric displacement used to assign polarity; for the current cation-centered algorithm it is identical to delta.",
    "polarity": "Discrete geometric polarity: +1 when delta_eff exceeds the tolerance, -1 when it is below the negative tolerance, and 0 otherwise or for an incomplete site.",
    "polarity_label": "Human-readable form of polarity: UP, DOWN, or UNASSIGNED.",
    "p_x (e*angstrom)": "Cartesian x component of the local neighbor-charge dipole sum, calculated as the sum of neighbor charge times its center-to-neighbor bond vector.",
    "p_y (e*angstrom)": "Cartesian y component of the local neighbor-charge dipole sum, calculated as the sum of neighbor charge times its center-to-neighbor bond vector.",
    "p_z (e*angstrom)": "Cartesian z component of the local neighbor-charge dipole sum, calculated as the sum of neighbor charge times its center-to-neighbor bond vector.",
    "p_x (debye)": "The local dipole x component converted from e*angstrom to debye.",
    "p_y (debye)": "The local dipole y component converted from e*angstrom to debye.",
    "p_z (debye)": "The local dipole z component converted from e*angstrom to debye.",
    "eta_c (e*angstrom)": "Charge-weighted local polar order obtained by projecting the local neighbor-charge dipole vector onto the normalized c-axis.",
    "eta_c (debye)": "The charge-weighted local c-axis polar order converted from e*angstrom to debye.",
    "n_sites": "Total number of selected center sites represented in the group or frame.",
    "n_complete_sites": "Number of selected centers having all four assigned neighbors.",
    "n_incomplete_sites": "Number of selected centers lacking a complete four-neighbor assignment.",
    "net_eta_c (e*angstrom)": "Mean finite eta_c over complete sites in the frame.",
    "positive_conditional_eta_c (e*angstrom)": "Mean eta_c over complete sites whose eta_c is strictly positive.",
    "negative_conditional_eta_c (e*angstrom)": "Mean eta_c over complete sites whose eta_c is strictly negative.",
    "n_positive_eta": "Number of complete sites with positive finite eta_c.",
    "n_negative_eta": "Number of complete sites with negative finite eta_c.",
    "n_zero_eta": "Number of complete sites with eta_c exactly equal to zero.",
    "n_polarity_up": "Number of complete sites assigned polarity +1.",
    "n_polarity_down": "Number of complete sites assigned polarity -1.",
    "n_polarity_zero": "Number of complete sites assigned polarity 0.",
    "fraction_polarity_up": "Fraction of complete sites assigned polarity +1.",
    "fraction_polarity_down": "Fraction of complete sites assigned polarity -1.",
    "mean_delta_eff (angstrom)": "Mean effective geometric displacement over complete sites in the frame.",
    "proton_group": "Readable proton-proximity category: with_proton or without_proton.",
    "n_basal_values": "Number of finite mean basal bond projections contributing to this proton-proximity group.",
    "n_basal_change_values": "Number of finite reference-relative basal changes contributing to this proton-proximity group.",
    "avg_mean_basal_bond_c_angstrom": "Group mean of mean_basal_bond_c values in angstrom.",
    "min_mean_basal_bond_c_angstrom": "Group minimum of mean_basal_bond_c values in angstrom.",
    "max_mean_basal_bond_c_angstrom": "Group maximum of mean_basal_bond_c values in angstrom.",
    "avg_mean_basal_bond_c_change_from_frame_0_angstrom": "Group mean of reference-relative mean basal bond changes in angstrom.",
    "min_mean_basal_bond_c_change_from_frame_0_angstrom": "Group minimum of reference-relative mean basal bond changes in angstrom.",
    "max_mean_basal_bond_c_change_from_frame_0_angstrom": "Group maximum of reference-relative mean basal bond changes in angstrom.",
}


def _neighbor_wide_description(column: str) -> str | None:
    match = re.fullmatch(r"neighbor_([1-4])_(.+)", column)
    if match is None:
        return None
    rank, field = match.groups()
    source = {
        "atom_id": "persistent atom identifier",
        "element": "unmodified chemical symbol or atom label",
        "charge (e)": "charge in elementary-charge units",
        "distance (angstrom)": "minimum-image center-neighbor distance in angstrom",
        "bond_x (angstrom)": "center-to-neighbor bond-vector x component in angstrom",
        "bond_y (angstrom)": "center-to-neighbor bond-vector y component in angstrom",
        "bond_z (angstrom)": "center-to-neighbor bond-vector z component in angstrom",
        "bond_c (angstrom)": "bond-vector projection onto the normalized c-axis in angstrom",
        "role": "apical, basal, or unassigned geometric role",
    }.get(field)
    if source is None:
        return None
    return f"Excel-friendly wide-column copy of the {source} for distance-ranked neighbor {rank}."


def variable_description(column: str) -> str:
    """Return the documented meaning of one generated CSV variable."""

    description = VARIABLE_DESCRIPTIONS.get(column) or _neighbor_wide_description(column)
    if description is None:
        raise KeyError(f"No wurtzite polarity variable description is defined for {column!r}.")
    return description


def write_polarity_variable_guide(result, output: Path) -> Path:
    """Write a plain-text glossary for every column in the generated polarity tables."""

    frames = (
        centers_csv_table(result.centers),
        neighbors_csv_table(result.neighbor_geometry),
        polarity_csv_table(result.table),
        time_after_iter(result.summary),
        time_after_iter(result.proton_summary),
    )
    columns = list(dict.fromkeys(column for frame in frames for column in frame.columns))
    lines = [
        "Wurtzite polarity variable guide",
        "=================================",
        "",
        "Geometry uses minimum-image center-to-neighbor vectors and projections onto the normalized --c-axis. Charge-weighted quantities require finite dynamic or explicit formal neighbor charges; geometric delta and polarity do not depend on charge.",
        "",
        "Variables",
        "---------",
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
    """Write the reusable neighbor, site-polarity, and summary tables."""

    output.mkdir(parents=True, exist_ok=True)
    helpful = output / "other_helpful_data"
    helpful.mkdir(parents=True, exist_ok=True)
    primary_csv_directory = helpful if all_csvs_in_helpful_data else output
    paths = {
        "centers": helpful / "centers.csv",
        "neighbors": helpful / "neighbors.csv",
        "polarity": primary_csv_directory / "polarity.csv",
        "summary": primary_csv_directory / "polarity_summary.csv",
        "proton": primary_csv_directory / "proton_proximity_summary.csv",
        "variables": output / "polarity_variables.txt",
    }
    centers_csv_table(result.centers).to_csv(paths["centers"], index=False)
    neighbors_csv_table(result.neighbor_geometry).to_csv(paths["neighbors"], index=False)
    polarity = (
        result.table[result.table["has_four_neighbors"].astype(bool)]
        if complete_only else result.table
    )
    polarity_csv_table(polarity).to_csv(paths["polarity"], index=False)
    time_after_iter(result.summary).to_csv(paths["summary"], index=False)
    time_after_iter(result.proton_summary).to_csv(paths["proton"], index=False)
    write_polarity_variable_guide(result, output)
    for obsolete in (
            "centers.csv",
            "neighbors.csv",
            "centers_and_neighbors.csv",
            "neighbor_geometry.csv",
            "table.csv",
            "apical_neighbors.csv",
            "basal_neighbors.csv",
    ):
        (output / obsolete).unlink(missing_ok=True)
    if all_csvs_in_helpful_data:
        for relocated in (
                "polarity.csv",
                "polarity_summary.csv",
                "proton_proximity_summary.csv",
        ):
            (output / relocated).unlink(missing_ok=True)
    return paths


__all__ = [
    "VARIABLE_DESCRIPTIONS",
    "variable_description",
    "write_polarity_tables",
    "write_polarity_variable_guide",
]
