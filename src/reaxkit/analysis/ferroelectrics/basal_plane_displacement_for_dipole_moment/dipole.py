"""Dipoles measured from the mean plane of three basal anions.

This local structural representation follows the reference-plane construction
used by Hayden et al., "Ferroelectricity in boron-substituted aluminum nitride
thin films," *Physical Review Materials* 5, 044412 (2021),
https://doi.org/10.1103/PhysRevMaterials.5.044412.

The three-folded wurtzite analysis identifies each Al/B center and its three
basal N neighbors. The N framework is held fixed as the local nonpolar
reference: every Al/B center is displaced from its basal-N mean plane, while N
and other reference-framework ions have zero displacement. Every ion is
present exactly once in the supercell table, but an ion is never measured from
several unrelated tetrahedral planes.

This is a classical local approximation to the paper's first-principles
expression. User-selected formal or ReaxFF ``fort.7`` charges replace the
longitudinal Born effective charges, and no electronic Berry-phase term is
available. The electron charge sign is explicitly negative, as requested by
the package convention.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from reaxkit.analysis.base import AnalysisTask
from reaxkit.analysis.ferroelectrics.four_folded_wurtzite.neighbors import (
    _trajectory_and_charges,
    required_wurtzite_data_type,
)
from reaxkit.analysis.ferroelectrics.poled_counts import directional_poled_counts
from reaxkit.analysis.ferroelectrics.three_folded_wurtzite.polarity import (
    EA_TO_DEBYE,
    WurtzitePolarityRequest,
    WurtzitePolarityResult,
    calculate_polarity_from_trajectory,
)
from reaxkit.core.registry.analysis_task_registry import register_task
from reaxkit.domain.base_result import BaseResult
from reaxkit.domain.data_models import TrajectoryData
from reaxkit.presentation.specs import PresentationSpec

ELECTRON_CHARGE_SIGN = -1.0


@dataclass
class BasalPlaneDipoleRequest(WurtzitePolarityRequest):
    """Configure the Hayden et al. basal-plane displacement approximation."""


@dataclass
class BasalPlaneDipoleResult(BaseResult):
    """Per-site basal references plus one contribution per supercell ion."""

    table: pd.DataFrame
    request: BasalPlaneDipoleRequest
    polarity_result: WurtzitePolarityResult
    ions: pd.DataFrame
    summary: pd.DataFrame
    poled_counts: pd.DataFrame
    frame_indices: np.ndarray
    iterations: np.ndarray

    @property
    def csv_tables(self) -> dict[str, pd.DataFrame]:
        return {
            "basal_plane_dipole": self.table,
            "basal_plane_ions": self.ions,
            "basal_plane_dipole_summary": self.summary,
            "basal_plane_dipole_poled_counts": self.poled_counts,
        }


def _empty_vector() -> np.ndarray:
    return np.full(3, np.nan, dtype=float)


def _iteration_for_frame(trajectory: TrajectoryData, frame: int) -> int:
    values = trajectory.iterations
    if values is None and trajectory.simulation is not None:
        values = trajectory.simulation.iterations
    return int(np.asarray(values)[frame]) if values is not None else int(frame)


def _charge_for_atom(
    *, frame: int, atom_index: int, element: str, charge_array: np.ndarray | None,
    formal_charges: dict[str, float],
) -> float:
    if charge_array is not None:
        return float(charge_array[frame, atom_index])
    return float(formal_charges.get(element.casefold(), np.nan))


def calculate_basal_plane_dipoles(
    data, request: BasalPlaneDipoleRequest
) -> BasalPlaneDipoleResult:
    """Calculate the local basal-plane approximation to ``sum(e Z_k du_k)``.

    For each Al/B center ``i`` identified by the three-folded analysis,

    ``r_basal,i = (r_b1 + r_b2 + r_b3) / 3``

    ``du_i = r_center,i - r_basal,i``

    ``mu_i = - Z_i du_i``.

    The minus sign is the configured negative electron-charge sign. All atoms
    are written once to ``basal_plane_ions.csv``. The basal-N framework is the
    displacement reference, so N and any non-center atoms have ``du = 0`` and
    ``mu = 0``; they are not assigned the height of an apical bond. This
    prevents double counting and avoids averaging incompatible local origins.

    A strict reproduction of ``P = e/Omega sum_k Z*_{k,33} du_{k,3}`` requires
    coordinates for both the polar wurtzite and nonpolar hexagonal structures
    plus path-averaged Born effective charges. Formal and ReaxFF charges used
    here are an approximation, not Born effective charges.
    """

    trajectory, raw_charges = _trajectory_and_charges(data)
    charge_array = (
        None
        if request.charge_source == "formal" or raw_charges is None
        else np.asarray(raw_charges, dtype=float)
    )
    formal_charges = {
        str(element).casefold(): float(charge)
        for element, charge in request.formal_charges.items()
    }
    polarity = calculate_polarity_from_trajectory(data, request)
    geometry = polarity.neighbor_geometry.reset_index(drop=True)
    geometry_groups = geometry.groupby(
        ["frame_index", "site_atom_id"], sort=False
    ).indices
    roles = geometry["neighbor_role"].to_numpy(dtype=object, copy=False)
    image_positions = geometry[[
        "neighbor_image_x (angstrom)",
        "neighbor_image_y (angstrom)",
        "neighbor_image_z (angstrom)",
    ]].to_numpy(float)
    neighbor_ids = geometry["neighbor_atom_id"].to_numpy(int)

    rows: list[dict[str, object]] = []
    center_values: dict[tuple[int, int], dict[str, object]] = {}
    for site in polarity.table.itertuples(index=False):
        source = dict(zip(polarity.table.columns, site, strict=False))
        frame = int(source["frame_index"])
        site_id = int(source["site_atom_id"])
        key = (frame, site_id)
        group_indices = np.asarray(
            geometry_groups.get(key, np.empty(0, dtype=int)), dtype=int
        )
        basal_indices = group_indices[roles[group_indices] == "basal"]
        apical_indices = group_indices[roles[group_indices] == "apical"]
        has_three_basal = len(basal_indices) == 3
        has_apical = len(apical_indices) == 1

        basal_mean = _empty_vector()
        center_displacement = _empty_vector()
        apical_displacement = _empty_vector()
        center_contribution = _empty_vector()
        dipole = _empty_vector()
        apical_id = -1
        apical_charge = np.nan
        if has_three_basal:
            basal_mean = np.mean(image_positions[basal_indices], axis=0)
            center_position = np.asarray([
                source["site_x (angstrom)"],
                source["site_y (angstrom)"],
                source["site_z (angstrom)"],
            ], dtype=float)
            center_displacement = center_position - basal_mean
            center_charge = float(source["site_charge (e)"])
            if np.isfinite(center_charge):
                center_contribution = (
                    ELECTRON_CHARGE_SIGN * center_charge * center_displacement
                )
                dipole = center_contribution.copy()
            center_values[key] = {
                "reference": basal_mean,
                "displacement": center_displacement,
                "mu": center_contribution,
                "has_dipole": bool(np.isfinite(center_contribution).all()),
            }
            if has_apical:
                apical_index = int(apical_indices[0])
                apical_displacement = image_positions[apical_index] - basal_mean
                apical_id = int(neighbor_ids[apical_index])
                apical_charge = float(
                    geometry.iloc[apical_index]["neighbor_charge (e)"]
                )

        row: dict[str, object] = {
            "frame_index": frame,
            "iter": int(source["iter"]),
            "site_atom_index": int(source["site_atom_index"]),
            "site_atom_id": site_id,
            "site_element": str(source["site_element"]),
            "site_x (angstrom)": float(source["site_x (angstrom)"]),
            "site_y (angstrom)": float(source["site_y (angstrom)"]),
            "site_z (angstrom)": float(source["site_z (angstrom)"]),
            "center_charge (e)": float(source["site_charge (e)"]),
            "apical_atom_id": apical_id,
            "apical_charge (e)": apical_charge,
            "polarity": int(source["polarity"]),
            "polarity_label": str(source["polarity_label"]),
            "has_three_basal_neighbors": bool(source["has_three_basal_neighbors"]),
            "has_apical_neighbor": bool(source["has_apical_neighbor"]),
            "has_basal_plane_dipole": bool(np.isfinite(dipole).all()),
            "included_ion_count": 1 if has_three_basal else 0,
            "electron_charge_sign": ELECTRON_CHARGE_SIGN,
            "charge_source": str(source["charge_source"]),
            "reference_convention": "fixed basal-anion framework",
        }
        zero = np.zeros(3, dtype=float) if has_three_basal else _empty_vector()
        for component, axis in enumerate("xyz"):
            row[f"basal_mean_{axis} (angstrom)"] = basal_mean[component]
            row[f"center_displacement_{axis} (angstrom)"] = center_displacement[component]
            row[f"apical_displacement_{axis} (angstrom)"] = apical_displacement[component]
            row[f"center_mu_{axis} (e*angstrom)"] = center_contribution[component]
            row[f"basal_mu_{axis} (e*angstrom)"] = zero[component]
            row[f"apical_mu_{axis} (e*angstrom)"] = zero[component]
            row[f"mu_{axis} (e*angstrom)"] = dipole[component]
            row[f"mu_{axis} (debye)"] = dipole[component] * EA_TO_DEBYE
        rows.append(row)

    table = pd.DataFrame(rows)
    positions = np.asarray(trajectory.positions, dtype=float)
    atom_ids = np.asarray(trajectory.atom_ids, dtype=int)
    elements = np.asarray(trajectory.elements, dtype=object)
    ion_rows: list[dict[str, object]] = []
    for frame in np.asarray(polarity.frame_indices, dtype=int):
        iteration = _iteration_for_frame(trajectory, int(frame))
        for atom_index, (atom_id, element) in enumerate(zip(atom_ids, elements, strict=True)):
            position = positions[int(frame), atom_index]
            if not np.isfinite(position).all():
                continue
            charge = _charge_for_atom(
                frame=int(frame), atom_index=atom_index, element=str(element),
                charge_array=charge_array, formal_charges=formal_charges,
            )
            values = center_values.get((int(frame), int(atom_id)))
            if values is None:
                reference = position.copy()
                displacement = np.zeros(3, dtype=float)
                contribution = np.zeros(3, dtype=float)
                role = "fixed_reference"
                has_dipole = True
            else:
                reference = np.asarray(values["reference"], dtype=float)
                displacement = np.asarray(values["displacement"], dtype=float)
                contribution = np.asarray(values["mu"], dtype=float)
                role = "displaced_center"
                has_dipole = bool(values["has_dipole"])
            ion: dict[str, object] = {
                "frame_index": int(frame), "iter": iteration,
                "atom_id": int(atom_id), "element": str(element), "role": role,
                "charge (e)": charge, "electron_charge_sign": ELECTRON_CHARGE_SIGN,
                "reference_occurrence_count": 1,
                "reference_convention": "fixed basal-anion framework",
                "x (angstrom)": float(position[0]),
                "y (angstrom)": float(position[1]),
                "z (angstrom)": float(position[2]),
                "has_ion_dipole": has_dipole,
            }
            for component, axis in enumerate("xyz"):
                ion[f"basal_reference_{axis} (angstrom)"] = float(reference[component])
                ion[f"displacement_{axis} (angstrom)"] = float(displacement[component])
                ion[f"mu_{axis} (e*angstrom)"] = float(contribution[component])
                ion[f"mu_{axis} (debye)"] = float(contribution[component] * EA_TO_DEBYE)
            ion_rows.append(ion)
    ions = pd.DataFrame(ion_rows)

    summary_rows: list[dict[str, object]] = []
    for (frame, iteration), group in ions.groupby(["frame_index", "iter"], sort=True):
        valid = group[group["has_ion_dipole"].astype(bool)]
        row: dict[str, object] = {
            "frame_index": int(frame), "iter": int(iteration),
            "ion_count": len(group), "valid_ion_count": len(valid),
            "displaced_ion_count": int((group["role"] == "displaced_center").sum()),
            "reference_convention": "fixed basal-anion framework",
        }
        for axis in "xyz":
            value = float(valid[f"mu_{axis} (e*angstrom)"].sum())
            row[f"mu_{axis} (e*angstrom)"] = value
            row[f"mu_{axis} (debye)"] = value * EA_TO_DEBYE
        summary_rows.append(row)
    poled_counts = directional_poled_counts(
        table,
        trajectory,
        {axis: f"mu_{axis} (e*angstrom)" for axis in "xyz"},
    )
    return BasalPlaneDipoleResult(
        table=table, request=request, polarity_result=polarity,
        ions=ions, summary=pd.DataFrame(summary_rows), poled_counts=poled_counts,
        frame_indices=polarity.frame_indices, iterations=polarity.iterations,
    )


@register_task(
    "get-basal-plane-displacement-dipole",
    label="Basal-plane Displacement Dipole",
)
class BasalPlaneDipoleTask(AnalysisTask):
    """Calculate local basal-plane dipoles for the fixed-anion reference."""

    required_data = TrajectoryData
    supports_selective_streaming = False
    VERSION = "3"

    def required_data_for(self, request: BasalPlaneDipoleRequest, args: dict | None = None):
        return required_wurtzite_data_type(request, args)

    @staticmethod
    def required_data_fields_for(request: BasalPlaneDipoleRequest, _args: dict) -> tuple[str, ...]:
        return ("trajectory", "charges") if request.charge_source != "formal" else ("trajectory",)

    @staticmethod
    def recommended_presentations(
        _result: BasalPlaneDipoleResult, _payload: dict[str, Any]
    ) -> list[PresentationSpec]:
        return [PresentationSpec(renderer="table", label="Basal-plane dipoles", view_type="table")]

    def run(self, data, request: BasalPlaneDipoleRequest, reporter=None):
        _ = reporter
        return calculate_basal_plane_dipoles(data, request)


__all__ = [
    "BasalPlaneDipoleRequest", "BasalPlaneDipoleResult", "BasalPlaneDipoleTask",
    "ELECTRON_CHARGE_SIGN", "calculate_basal_plane_dipoles",
]
