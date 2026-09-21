"""Polarity and polarity-aware apical assignment for three-folded wurtzite."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd

from reaxkit.analysis.base import AnalysisTask
from reaxkit.analysis.ferroelectrics.four_folded_wurtzite.neighbors import (
    _charges_for_request,
    _select_frames,
    _validate_request,
    required_wurtzite_data_type,
    time_after_iter,
)
from reaxkit.analysis.ferroelectrics.three_folded_wurtzite.neighbors import (
    WurtziteNeighborRequest,
    WurtziteNeighborResult,
    extract_wurtzite_neighbors,
)
from reaxkit.core.registry.analysis_task_registry import register_task
from reaxkit.domain.base_result import BaseResult
from reaxkit.domain.data_models import TrajectoryData
from reaxkit.presentation.specs import PresentationSpec

EA_TO_DEBYE = 4.80320427

POLARITY_COLUMNS = [
    "frame_index", "iter", "site_atom_index", "site_atom_id", "site_element",
    "site_x (angstrom)", "site_y (angstrom)", "site_z (angstrom)", "site_charge (e)",
    "neighbor_count_within_cutoff", "has_three_basal_neighbors", "has_apical_neighbor",
    "has_proton_within_cutoff", "proton_cutoff (angstrom)", "charge_source",
    "apical_neighbor_rank", "apical_neighbor_atom_id", "apical_bond_c (angstrom)",
    "mean_basal_bond_c (angstrom)",
    "mean_basal_bond_c_change_from_frame_0 (angstrom)", "delta (angstrom)",
    "delta_eff (angstrom)", "polarity", "polarity_label", "p_x (e*angstrom)",
    "p_y (e*angstrom)", "p_z (e*angstrom)", "p_x (debye)", "p_y (debye)",
    "p_z (debye)", "eta_c (e*angstrom)", "eta_c (debye)",
]

POLARITY_CSV_COLUMNS = [
    "frame_index", "iter", "time", "site_atom_id", "has_three_basal_neighbors",
    "has_apical_neighbor", "apical_neighbor_rank", "apical_neighbor_atom_id",
    "apical_bond_c (angstrom)", "mean_basal_bond_c (angstrom)",
    "mean_basal_bond_c_change_from_frame_0 (angstrom)", "delta (angstrom)",
    "delta_eff (angstrom)", "polarity", "polarity_label", "p_x (e*angstrom)",
    "p_y (e*angstrom)", "p_z (e*angstrom)", "p_x (debye)", "p_y (debye)",
    "p_z (debye)", "eta_c (e*angstrom)", "eta_c (debye)",
]


def polarity_csv_table(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [column for column in POLARITY_CSV_COLUMNS if column in frame.columns]
    return time_after_iter(frame.loc[:, columns])


@dataclass
class WurtzitePolarityRequest(WurtziteNeighborRequest):
    """Configure basal-only polarity and directional apical selection."""

    polarity_tolerance: float = 1.0e-10
    reference_frame: int = 0


@dataclass
class WurtzitePolarityResult(BaseResult):
    table: pd.DataFrame
    request: WurtzitePolarityRequest
    centers: pd.DataFrame
    neighbors: pd.DataFrame
    neighbor_geometry: pd.DataFrame
    apical_neighbors: pd.DataFrame
    basal_neighbors: pd.DataFrame
    ignored_neighbors: pd.DataFrame
    summary: pd.DataFrame
    proton_summary: pd.DataFrame
    frame_indices: np.ndarray
    iterations: np.ndarray

    @property
    def csv_tables(self) -> dict[str, pd.DataFrame]:
        return {
            "polarity": polarity_csv_table(self.table),
            "polarity_summary": time_after_iter(self.summary),
            "proton_proximity_summary": time_after_iter(self.proton_summary),
        }


def _summary_for_frame(table: pd.DataFrame) -> dict[str, object]:
    complete = table[table["has_three_basal_neighbors"].astype(bool)]
    polarity = complete["polarity"].to_numpy(dtype=int)
    eta = pd.to_numeric(complete["eta_c (e*angstrom)"], errors="coerce").to_numpy(float)
    eta = eta[np.isfinite(eta)]
    count = len(complete)
    return {
        "frame_index": int(table["frame_index"].iloc[0]),
        "iter": int(table["iter"].iloc[0]),
        "n_sites": len(table),
        "n_complete_sites": count,
        "n_incomplete_sites": len(table) - count,
        "n_sites_with_apical_neighbor": int(complete["has_apical_neighbor"].sum()),
        "net_eta_c (e*angstrom)": float(np.mean(eta)) if eta.size else np.nan,
        "n_polarity_up": int(np.count_nonzero(polarity > 0)),
        "n_polarity_down": int(np.count_nonzero(polarity < 0)),
        "n_polarity_zero": int(np.count_nonzero(polarity == 0)),
        "fraction_polarity_up": float(np.count_nonzero(polarity > 0) / count) if count else np.nan,
        "fraction_polarity_down": float(np.count_nonzero(polarity < 0) / count) if count else np.nan,
        "mean_delta_eff (angstrom)": float(complete["delta_eff (angstrom)"].mean()) if count else np.nan,
    }


def summarize_by_proton_proximity(table: pd.DataFrame) -> pd.DataFrame:
    if table.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for (frame, iteration), frame_table in table.groupby(["frame_index", "iter"], sort=True):
        for near_proton in (False, True):
            group = frame_table[frame_table["has_proton_within_cutoff"].astype(bool).eq(near_proton)]
            basal = pd.to_numeric(group["mean_basal_bond_c (angstrom)"], errors="coerce")
            change = pd.to_numeric(
                group["mean_basal_bond_c_change_from_frame_0 (angstrom)"], errors="coerce"
            )
            rows.append({
                "frame_index": int(frame), "iter": int(iteration),
                "has_proton_within_cutoff": near_proton,
                "proton_group": "with_proton" if near_proton else "without_proton",
                "proton_cutoff (angstrom)": float(frame_table["proton_cutoff (angstrom)"].iloc[0]),
                "n_sites": len(group),
                "n_complete_sites": int(group["has_three_basal_neighbors"].sum()),
                "n_basal_values": int(basal.count()),
                "avg_mean_basal_bond_c_angstrom": float(basal.mean()),
                "min_mean_basal_bond_c_angstrom": float(basal.min()),
                "max_mean_basal_bond_c_angstrom": float(basal.max()),
                "n_basal_change_values": int(change.count()),
                "avg_mean_basal_bond_c_change_from_frame_0_angstrom": float(change.mean()),
                "min_mean_basal_bond_c_change_from_frame_0_angstrom": float(change.min()),
                "max_mean_basal_bond_c_change_from_frame_0_angstrom": float(change.max()),
            })
    return pd.DataFrame(rows)


def calculate_wurtzite_polarity(
    neighbors: WurtziteNeighborResult,
    request: WurtzitePolarityRequest,
    *,
    reference_basal: Optional[dict[int, float]] = None,
) -> WurtzitePolarityResult:
    """Calculate polarity from basal bonds, then select an apical neighbor on that side."""

    tolerance = float(request.polarity_tolerance)
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("polarity_tolerance must be finite and nonnegative.")
    centers = neighbors.centers.copy()
    geometry = neighbors.neighbors.copy().reset_index(drop=True)
    site_rows: list[dict[str, object]] = []
    geometry_groups = geometry.groupby(
        ["frame_index", "site_atom_id"], sort=False
    ).indices
    roles = geometry["neighbor_role"].to_numpy(dtype=object, copy=True)
    bond_c = geometry["bond_c (angstrom)"].to_numpy(dtype=float, copy=False)
    neighbor_charges = geometry["neighbor_charge (e)"].to_numpy(dtype=float, copy=False)
    bond_vectors = geometry[
        ["bond_x (angstrom)", "bond_y (angstrom)", "bond_z (angstrom)"]
    ].to_numpy(dtype=float, copy=False)
    neighbor_ranks = geometry["neighbor_rank"].to_numpy(dtype=int, copy=False)
    neighbor_ids = geometry["neighbor_atom_id"].to_numpy(dtype=int, copy=False)
    neighbor_elements = geometry["neighbor_element"].to_numpy(dtype=object, copy=False)
    neighbor_distances = geometry["distance (angstrom)"].to_numpy(dtype=float, copy=False)

    for center in centers.itertuples(index=False):
        row = dict(zip(centers.columns, center, strict=False))
        frame, site_id = int(row["frame_index"]), int(row["site_atom_id"])
        group_indices = np.asarray(
            geometry_groups.get((frame, site_id), np.empty(0, dtype=int)), dtype=int
        )
        group_indices = group_indices[np.argsort(neighbor_ranks[group_indices], kind="stable")]
        basal_indices = group_indices[roles[group_indices] == "basal"]
        has_three = bool(row["has_three_basal_neighbors"]) and len(basal_indices) == 3
        basal_mean = float(np.mean(bond_c[basal_indices])) if has_three else np.nan
        # Bond vectors point from the center to its neighbors. Basal anions
        # below the center have negative c projections, whereas the physical
        # polarization points from those anions toward the center.
        delta = -basal_mean
        polarity = (
            int(delta > tolerance) - int(delta < -tolerance) if np.isfinite(delta) else 0
        )

        candidate_indices = group_indices[roles[group_indices] == "apical_candidate"]
        if polarity > 0:
            directional = candidate_indices[bond_c[candidate_indices] > tolerance]
        elif polarity < 0:
            directional = candidate_indices[bond_c[candidate_indices] < -tolerance]
        else:
            directional = np.empty(0, dtype=int)
        apical_index = None
        if directional.size:
            apical_index = int(directional[np.argmax(np.abs(bond_c[directional]))])
        roles[candidate_indices] = "ignored"
        if apical_index is not None:
            roles[apical_index] = "apical"
        row["has_apical_neighbor"] = apical_index is not None

        selected_indices = basal_indices
        if apical_index is not None:
            selected_indices = np.append(selected_indices, apical_index)
        charges = neighbor_charges[selected_indices]
        bonds = bond_vectors[selected_indices]
        dipole = np.full(3, np.nan)
        eta_c = np.nan
        if has_three and np.isfinite(charges).all() and np.isfinite(bonds).all():
            dipole = np.sum(charges[:, None] * bonds, axis=0)
            c_axis = np.asarray([row["c_axis_x"], row["c_axis_y"], row["c_axis_z"]], float)
            eta_c = float(dipole @ c_axis)

        row.update({
            "apical_neighbor_rank": neighbor_ranks[apical_index] if apical_index is not None else 0,
            "apical_neighbor_atom_id": neighbor_ids[apical_index] if apical_index is not None else -1,
            "apical_bond_c (angstrom)": bond_c[apical_index] if apical_index is not None else np.nan,
            "mean_basal_bond_c (angstrom)": basal_mean,
            "delta (angstrom)": delta,
            "delta_eff (angstrom)": delta,
            "polarity": polarity,
            "polarity_label": "UP" if polarity > 0 else "DOWN" if polarity < 0 else "UNASSIGNED",
            "p_x (e*angstrom)": float(dipole[0]), "p_y (e*angstrom)": float(dipole[1]),
            "p_z (e*angstrom)": float(dipole[2]),
            "p_x (debye)": float(dipole[0] * EA_TO_DEBYE),
            "p_y (debye)": float(dipole[1] * EA_TO_DEBYE),
            "p_z (debye)": float(dipole[2] * EA_TO_DEBYE),
            "eta_c (e*angstrom)": eta_c, "eta_c (debye)": eta_c * EA_TO_DEBYE,
        })
        for source_index in group_indices:
            rank = neighbor_ranks[source_index]
            prefix = f"neighbor_{rank}"
            row[f"{prefix}_atom_id"] = neighbor_ids[source_index]
            row[f"{prefix}_element"] = str(neighbor_elements[source_index])
            row[f"{prefix}_charge (e)"] = neighbor_charges[source_index]
            row[f"{prefix}_distance (angstrom)"] = neighbor_distances[source_index]
            row[f"{prefix}_bond_x (angstrom)"] = bond_vectors[source_index, 0]
            row[f"{prefix}_bond_y (angstrom)"] = bond_vectors[source_index, 1]
            row[f"{prefix}_bond_z (angstrom)"] = bond_vectors[source_index, 2]
            row[f"{prefix}_bond_c (angstrom)"] = bond_c[source_index]
            row[f"{prefix}_role"] = str(roles[source_index])
        site_rows.append(row)

    geometry["neighbor_role"] = roles
    table = pd.DataFrame(site_rows)
    reference = dict(reference_basal or {})
    reference_rows = table[table["frame_index"].astype(int).eq(int(request.reference_frame))]
    if not reference and not reference_rows.empty:
        reference = dict(zip(
            reference_rows["site_atom_id"].astype(int),
            reference_rows["mean_basal_bond_c (angstrom)"].astype(float), strict=False,
        ))
    table["mean_basal_bond_c_change_from_frame_0 (angstrom)"] = (
        table["mean_basal_bond_c (angstrom)"] - table["site_atom_id"].astype(int).map(reference)
    )
    ordered = [column for column in POLARITY_COLUMNS if column in table.columns]
    table = table[ordered + [column for column in table.columns if column not in ordered]]
    apical_by_site = {
        (int(row["frame_index"]), int(row["site_atom_id"])): bool(row["has_apical_neighbor"])
        for _, row in table.iterrows()
    }
    centers["has_apical_neighbor"] = [
        apical_by_site.get((int(row.frame_index), int(row.site_atom_id)), False)
        for row in centers.itertuples(index=False)
    ]
    summary = pd.DataFrame([
        _summary_for_frame(group) for _, group in table.groupby("frame_index", sort=True)
    ]) if not table.empty else pd.DataFrame()
    return WurtzitePolarityResult(
        table=table, request=request, centers=centers, neighbors=neighbors.neighbors,
        neighbor_geometry=geometry,
        apical_neighbors=geometry[geometry["neighbor_role"].eq("apical")].copy(),
        basal_neighbors=geometry[geometry["neighbor_role"].eq("basal")].copy(),
        ignored_neighbors=geometry[geometry["neighbor_role"].eq("ignored")].copy(),
        summary=summary, proton_summary=summarize_by_proton_proximity(table),
        frame_indices=neighbors.frame_indices, iterations=neighbors.iterations,
    )


def calculate_polarity_from_trajectory(data, request: WurtzitePolarityRequest):
    trajectory, charges = _charges_for_request(data, request)
    selected = _select_frames(np.asarray(trajectory.positions).shape[0], request)
    reference = int(request.reference_frame)
    if reference < 0 or reference >= np.asarray(trajectory.positions).shape[0]:
        raise ValueError(f"Reference frame {reference} is not present in the trajectory.")
    analysis_frames = list(dict.fromkeys([reference, *selected]))
    raw = calculate_wurtzite_polarity(
        extract_wurtzite_neighbors(
            trajectory, request, charges=charges, frame_indices=analysis_frames
        ),
        request,
    )
    if set(analysis_frames) == set(selected):
        return raw
    selected_set = set(selected)

    def keep(frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty or "frame_index" not in frame:
            return frame
        return frame[frame["frame_index"].astype(int).isin(selected_set)].reset_index(drop=True)

    table = keep(raw.table)
    return WurtzitePolarityResult(
        table=table, request=request, centers=keep(raw.centers), neighbors=keep(raw.neighbors),
        neighbor_geometry=keep(raw.neighbor_geometry), apical_neighbors=keep(raw.apical_neighbors),
        basal_neighbors=keep(raw.basal_neighbors), ignored_neighbors=keep(raw.ignored_neighbors),
        summary=pd.DataFrame([
            _summary_for_frame(group) for _, group in table.groupby("frame_index", sort=True)
        ]) if not table.empty else pd.DataFrame(),
        proton_summary=summarize_by_proton_proximity(table),
        frame_indices=np.asarray(selected, int),
        iterations=np.asarray([
            int(np.asarray(trajectory.iterations).reshape(-1)[frame])
            if trajectory.iterations is not None else frame for frame in selected
        ], int),
    )


def combine_wurtzite_polarity_results(
    results: Sequence[WurtzitePolarityResult], request: WurtzitePolarityRequest
) -> WurtzitePolarityResult:
    def combine(name: str) -> pd.DataFrame:
        frames = [getattr(result, name) for result in results if not getattr(result, name).empty]
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    table = combine("table")
    return WurtzitePolarityResult(
        table=table, request=request, centers=combine("centers"), neighbors=combine("neighbors"),
        neighbor_geometry=combine("neighbor_geometry"), apical_neighbors=combine("apical_neighbors"),
        basal_neighbors=combine("basal_neighbors"), ignored_neighbors=combine("ignored_neighbors"),
        summary=pd.DataFrame([
            _summary_for_frame(group) for _, group in table.groupby("frame_index", sort=True)
        ]) if not table.empty else pd.DataFrame(),
        proton_summary=summarize_by_proton_proximity(table),
        frame_indices=np.concatenate([value.frame_indices for value in results]) if results else np.empty(0, int),
        iterations=np.concatenate([value.iterations for value in results]) if results else np.empty(0, int),
    )


@register_task("get-three-folded-wurtzite-polarity", label="Three-folded Wurtzite Polarity")
class WurtzitePolarityTask(AnalysisTask):
    """Calculate basal-only polarity and polarity-aware apical assignment."""

    required_data = TrajectoryData
    supports_selective_streaming = False
    VERSION = "1"

    def required_data_for(self, request: WurtzitePolarityRequest, args: dict | None = None):
        return required_wurtzite_data_type(request, args)

    @staticmethod
    def required_data_fields_for(request: WurtzitePolarityRequest, _args: dict) -> tuple[str, ...]:
        return ("trajectory", "charges") if request.charge_source != "formal" else ("trajectory",)

    @staticmethod
    def recommended_presentations(
        _result: WurtzitePolarityResult, _payload: dict[str, Any]
    ) -> list[PresentationSpec]:
        return [PresentationSpec(renderer="table", label="Three-folded polarity", view_type="table")]

    def run(self, data, request: WurtzitePolarityRequest, reporter=None):
        _ = reporter
        _validate_request(request)
        return calculate_polarity_from_trajectory(data, request)


__all__ = [
    "EA_TO_DEBYE", "POLARITY_COLUMNS", "POLARITY_CSV_COLUMNS",
    "WurtzitePolarityRequest", "WurtzitePolarityResult", "WurtzitePolarityTask",
    "calculate_polarity_from_trajectory", "calculate_wurtzite_polarity",
    "combine_wurtzite_polarity_results", "polarity_csv_table",
    "summarize_by_proton_proximity",
]
