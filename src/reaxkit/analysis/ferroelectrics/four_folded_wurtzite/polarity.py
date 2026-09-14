"""Local polar-order descriptors built from four-fold neighbor assignments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd

from reaxkit.analysis.base import AnalysisTask
from reaxkit.analysis.ferroelectrics.four_folded_wurtzite.neighbors import (
    WurtziteNeighborRequest,
    WurtziteNeighborResult,
    _charges_for_request,
    _select_frames,
    _source_frame,
    _validate_request,
    extract_wurtzite_neighbors,
    required_wurtzite_data_type,
    time_after_iter,
)
from reaxkit.core.registry.analysis_task_registry import register_task
from reaxkit.domain.base_result import BaseResult
from reaxkit.domain.data_models import TrajectoryData
from reaxkit.presentation.specs import PresentationSpec

EA_TO_DEBYE = 4.80320427
SITE_KEY = ["frame_index", "site_atom_id"]

POLARITY_COLUMNS = [
    "frame_index",
    "iter",
    "site_atom_index",
    "site_atom_id",
    "site_element",
    "site_x (angstrom)",
    "site_y (angstrom)",
    "site_z (angstrom)",
    "site_charge (e)",
    "neighbor_count_within_cutoff",
    "has_four_neighbors",
    "has_proton_within_cutoff",
    "proton_cutoff (angstrom)",
    "charge_source",
    "apical_neighbor_rank",
    "apical_neighbor_atom_id",
    "apical_bond_c (angstrom)",
    "mean_basal_bond_c (angstrom)",
    "mean_basal_bond_c_change_from_frame_0 (angstrom)",
    "delta (angstrom)",
    "delta_eff (angstrom)",
    "polarity",
    "polarity_label",
    "p_x (e*angstrom)",
    "p_y (e*angstrom)",
    "p_z (e*angstrom)",
    "p_x (debye)",
    "p_y (debye)",
    "p_z (debye)",
    "eta_c (e*angstrom)",
    "eta_c (debye)",
]

POLARITY_CSV_COLUMNS = [
    "frame_index",
    "iter",
    "time",
    "site_atom_id",
    "apical_neighbor_rank",
    "apical_neighbor_atom_id",
    "apical_bond_c (angstrom)",
    "mean_basal_bond_c (angstrom)",
    "mean_basal_bond_c_change_from_frame_0 (angstrom)",
    "delta (angstrom)",
    "delta_eff (angstrom)",
    "polarity",
    "polarity_label",
    "p_x (e*angstrom)",
    "p_y (e*angstrom)",
    "p_z (e*angstrom)",
    "p_x (debye)",
    "p_y (debye)",
    "p_z (debye)",
    "eta_c (e*angstrom)",
    "eta_c (debye)",
]


def polarity_csv_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Return the compact public polarity table without repeated neighbor inputs."""

    columns = [column for column in POLARITY_CSV_COLUMNS if column in frame.columns]
    return time_after_iter(frame.loc[:, columns])


@dataclass
class WurtzitePolarityRequest(WurtziteNeighborRequest):
    """Configure polarity and local charge-weighted order calculations."""

    polarity_tolerance: float = 1.0e-10
    reference_frame: int = 0


@dataclass
class WurtzitePolarityResult(BaseResult):
    """Site polarity, role-annotated neighbors, and frame/group summaries."""

    table: pd.DataFrame
    request: WurtzitePolarityRequest
    centers: pd.DataFrame
    neighbors: pd.DataFrame
    neighbor_geometry: pd.DataFrame
    apical_neighbors: pd.DataFrame
    basal_neighbors: pd.DataFrame
    summary: pd.DataFrame
    proton_summary: pd.DataFrame
    frame_indices: np.ndarray
    iterations: np.ndarray

    @property
    def csv_tables(self) -> dict[str, pd.DataFrame]:
        """Expose primary polarity tables; raw neighbor inputs are written separately."""

        return {
            "polarity": polarity_csv_table(self.table),
            "polarity_summary": time_after_iter(self.summary),
            "proton_proximity_summary": time_after_iter(self.proton_summary),
        }


def _summary_for_frame(table: pd.DataFrame) -> dict[str, object]:
    complete = table[table["has_four_neighbors"].astype(bool)]
    eta = pd.to_numeric(complete["eta_c (e*angstrom)"], errors="coerce").to_numpy(dtype=float)
    eta = eta[np.isfinite(eta)]
    polarity = complete["polarity"].to_numpy(dtype=int)
    positive = eta[eta > 0.0]
    negative = eta[eta < 0.0]
    n_complete = int(len(complete))
    return {
        "frame_index": int(table["frame_index"].iloc[0]),
        "iter": int(table["iter"].iloc[0]),
        "n_sites": int(len(table)),
        "n_complete_sites": n_complete,
        "n_incomplete_sites": int(len(table) - n_complete),
        "net_eta_c (e*angstrom)": float(np.mean(eta)) if eta.size else np.nan,
        "positive_conditional_eta_c (e*angstrom)": (
            float(np.mean(positive)) if positive.size else np.nan
        ),
        "negative_conditional_eta_c (e*angstrom)": (
            float(np.mean(negative)) if negative.size else np.nan
        ),
        "n_positive_eta": int(np.count_nonzero(eta > 0.0)),
        "n_negative_eta": int(np.count_nonzero(eta < 0.0)),
        "n_zero_eta": int(np.count_nonzero(eta == 0.0)),
        "n_polarity_up": int(np.count_nonzero(polarity > 0)),
        "n_polarity_down": int(np.count_nonzero(polarity < 0)),
        "n_polarity_zero": int(np.count_nonzero(polarity == 0)),
        "fraction_polarity_up": (
            float(np.count_nonzero(polarity > 0) / n_complete) if n_complete else np.nan
        ),
        "fraction_polarity_down": (
            float(np.count_nonzero(polarity < 0) / n_complete) if n_complete else np.nan
        ),
        "mean_delta_eff (angstrom)": (
            float(complete["delta_eff (angstrom)"].mean()) if n_complete else np.nan
        ),
    }


def summarize_by_proton_proximity(table: pd.DataFrame) -> pd.DataFrame:
    """Summarize basal geometry and frame-zero changes by proton proximity."""

    if table.empty:
        return pd.DataFrame()
    basal = "mean_basal_bond_c (angstrom)"
    change = "mean_basal_bond_c_change_from_frame_0 (angstrom)"
    group_columns = ["frame_index", "iter", "has_proton_within_cutoff"]
    grouped = table.groupby(group_columns, sort=True, dropna=False).agg(
        n_sites=("site_atom_id", "size"),
        n_complete_sites=("has_four_neighbors", "sum"),
        n_basal_values=(basal, "count"),
        avg_mean_basal_bond_c_angstrom=(basal, "mean"),
        min_mean_basal_bond_c_angstrom=(basal, "min"),
        max_mean_basal_bond_c_angstrom=(basal, "max"),
        n_basal_change_values=(change, "count"),
        avg_mean_basal_bond_c_change_from_frame_0_angstrom=(change, "mean"),
        min_mean_basal_bond_c_change_from_frame_0_angstrom=(change, "min"),
        max_mean_basal_bond_c_change_from_frame_0_angstrom=(change, "max"),
    )
    frame_iterations = table[["frame_index", "iter"]].drop_duplicates().sort_values("frame_index")
    complete_index = pd.MultiIndex.from_tuples(
        [
            (int(row.frame_index), int(row.iter), has_proton)
            for row in frame_iterations.itertuples(index=False)
            for has_proton in (False, True)
        ],
        names=group_columns,
    )
    grouped = grouped.reindex(complete_index).reset_index()
    count_columns = [
        "n_sites", "n_complete_sites", "n_basal_values", "n_basal_change_values"
    ]
    grouped[count_columns] = grouped[count_columns].fillna(0).astype(int)
    grouped.insert(3, "proton_group", np.where(
        grouped["has_proton_within_cutoff"], "with_proton", "without_proton"
    ))
    cutoff_values = table["proton_cutoff (angstrom)"].dropna().unique()
    grouped.insert(4, "proton_cutoff (angstrom)", (
        float(cutoff_values[0]) if len(cutoff_values) == 1 else np.nan
    ))
    return grouped


def calculate_wurtzite_polarity(
        neighbors: WurtziteNeighborResult,
        request: WurtzitePolarityRequest,
        *,
        reference_basal: Optional[dict[int, float]] = None,
) -> WurtzitePolarityResult:
    """Calculate polarity from the normalized output of neighbor extraction."""

    if not np.isfinite(request.polarity_tolerance) or request.polarity_tolerance < 0.0:
        raise ValueError("polarity_tolerance must be finite and nonnegative.")
    centers = neighbors.centers.copy()
    neighbor_geometry = neighbors.neighbors.copy()
    neighbor_geometry["neighbor_role"] = "unassigned"
    site_rows: list[dict[str, object]] = []

    for center in centers.itertuples(index=False):
        frame = int(getattr(center, "frame_index"))
        site_id = int(getattr(center, "site_atom_id"))
        mask = (
                neighbor_geometry["frame_index"].astype(int).eq(frame)
                & neighbor_geometry["site_atom_id"].astype(int).eq(site_id)
        )
        group = neighbor_geometry.loc[mask].sort_values("neighbor_rank", kind="stable")
        has_four = bool(getattr(center, "has_four_neighbors")) and len(group) == 4
        row = dict(zip(centers.columns, center, strict=False))
        dipole = np.full(3, np.nan)
        eta_c = np.nan
        apical_rank = 0
        apical_atom_id = -1
        apical_projection = np.nan
        basal_mean = np.nan
        delta = np.nan
        polarity = 0

        if has_four:
            projections = group["bond_c (angstrom)"].to_numpy(dtype=float)
            apical_position = int(np.argmax(np.abs(projections)))
            apical_index = group.index[apical_position]
            neighbor_geometry.loc[mask, "neighbor_role"] = "basal"
            neighbor_geometry.loc[apical_index, "neighbor_role"] = "apical"
            apical = group.iloc[apical_position]
            basal_positions = [position for position in range(4) if position != apical_position]
            apical_rank = int(apical["neighbor_rank"])
            apical_atom_id = int(apical["neighbor_atom_id"])
            apical_projection = float(projections[apical_position])
            basal_mean = float(np.mean(projections[basal_positions]))
            delta = apical_projection - basal_mean
            polarity = int(delta > request.polarity_tolerance) - int(
                delta < -request.polarity_tolerance
            )
            charges = group["neighbor_charge (e)"].to_numpy(dtype=float)
            bonds = group[
                ["bond_x (angstrom)", "bond_y (angstrom)", "bond_z (angstrom)"]
            ].to_numpy(dtype=float)
            if np.isfinite(charges).all() and np.isfinite(bonds).all():
                dipole = np.sum(charges[:, np.newaxis] * bonds, axis=0)
                c_axis = np.asarray(
                    [row["c_axis_x"], row["c_axis_y"], row["c_axis_z"]], dtype=float
                )
                eta_c = float(dipole @ c_axis)

        row.update(
            {
                "apical_neighbor_rank": apical_rank,
                "apical_neighbor_atom_id": apical_atom_id,
                "apical_bond_c (angstrom)": apical_projection,
                "mean_basal_bond_c (angstrom)": basal_mean,
                "delta (angstrom)": delta,
                "delta_eff (angstrom)": delta,
                "polarity": polarity,
                "polarity_label": (
                    "UP" if polarity > 0 else "DOWN" if polarity < 0 else "UNASSIGNED"
                ),
                "p_x (e*angstrom)": float(dipole[0]),
                "p_y (e*angstrom)": float(dipole[1]),
                "p_z (e*angstrom)": float(dipole[2]),
                "p_x (debye)": float(dipole[0] * EA_TO_DEBYE),
                "p_y (debye)": float(dipole[1] * EA_TO_DEBYE),
                "p_z (debye)": float(dipole[2] * EA_TO_DEBYE),
                "eta_c (e*angstrom)": eta_c,
                "eta_c (debye)": eta_c * EA_TO_DEBYE,
            }
        )
        # Excel-friendly wide neighbor columns retain the original script's convenience.
        for _, source_row in group.iterrows():
            rank = int(source_row["neighbor_rank"])
            prefix = f"neighbor_{rank}"
            row[f"{prefix}_atom_id"] = int(source_row["neighbor_atom_id"])
            row[f"{prefix}_element"] = str(source_row["neighbor_element"])
            row[f"{prefix}_charge (e)"] = float(source_row["neighbor_charge (e)"])
            row[f"{prefix}_distance (angstrom)"] = float(source_row["distance (angstrom)"])
            for axis in "xyzc":
                row[f"{prefix}_bond_{axis} (angstrom)"] = float(
                    source_row[f"bond_{axis} (angstrom)"]
                )
            row[f"{prefix}_role"] = (
                "apical" if rank == apical_rank and has_four else "basal" if has_four else "unassigned"
            )
        site_rows.append(row)

    table = pd.DataFrame(site_rows)
    reference = dict(reference_basal or {})
    reference_rows = table[table["frame_index"].astype(int).eq(int(request.reference_frame))]
    if not reference and not reference_rows.empty:
        reference = dict(zip(
            reference_rows["site_atom_id"].astype(int),
            reference_rows["mean_basal_bond_c (angstrom)"].astype(float),
            strict=False,
        ))
    table["mean_basal_bond_c_change_from_frame_0 (angstrom)"] = (
            table["mean_basal_bond_c (angstrom)"]
            - table["site_atom_id"].astype(int).map(reference)
    )
    ordered = [column for column in POLARITY_COLUMNS if column in table.columns]
    ordered.extend(column for column in table.columns if column not in ordered)
    table = table[ordered]
    summary = (
        pd.DataFrame([_summary_for_frame(group) for _, group in table.groupby("frame_index", sort=True)])
        if not table.empty else pd.DataFrame()
    )
    return WurtzitePolarityResult(
        table=table,
        request=request,
        centers=centers,
        neighbors=neighbors.neighbors,
        neighbor_geometry=neighbor_geometry,
        apical_neighbors=neighbor_geometry[neighbor_geometry["neighbor_role"].eq("apical")].copy(),
        basal_neighbors=neighbor_geometry[neighbor_geometry["neighbor_role"].eq("basal")].copy(),
        summary=summary,
        proton_summary=summarize_by_proton_proximity(table),
        frame_indices=neighbors.frame_indices,
        iterations=neighbors.iterations,
    )


def calculate_polarity_from_trajectory(data, request: WurtzitePolarityRequest):
    trajectory, charges = _charges_for_request(data, request)
    selected = _select_frames(np.asarray(trajectory.positions).shape[0], request)
    reference = int(request.reference_frame)
    if reference < 0 or reference >= np.asarray(trajectory.positions).shape[0]:
        raise ValueError(f"Reference frame {reference} is not present in the trajectory.")
    analysis_frames = list(dict.fromkeys([reference, *selected]))
    neighbor_result = extract_wurtzite_neighbors(
        trajectory,
        request,
        charges=charges,
        frame_indices=analysis_frames,
    )
    result = calculate_wurtzite_polarity(neighbor_result, request)
    selected_set = set(selected)
    if set(analysis_frames) == selected_set:
        return result

    def selected_rows(frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty or "frame_index" not in frame:
            return frame
        return frame[frame["frame_index"].astype(int).isin(selected_set)].reset_index(drop=True)

    table = selected_rows(result.table)
    return WurtzitePolarityResult(
        table=table,
        request=request,
        centers=selected_rows(result.centers),
        neighbors=selected_rows(result.neighbors),
        neighbor_geometry=selected_rows(result.neighbor_geometry),
        apical_neighbors=selected_rows(result.apical_neighbors),
        basal_neighbors=selected_rows(result.basal_neighbors),
        summary=(
            pd.DataFrame([_summary_for_frame(group) for _, group in table.groupby("frame_index", sort=True)])
            if not table.empty else pd.DataFrame()
        ),
        proton_summary=summarize_by_proton_proximity(table),
        frame_indices=np.asarray(selected, dtype=int),
        iterations=np.asarray([
            int(np.asarray(trajectory.iterations).reshape(-1)[frame])
            if trajectory.iterations is not None else frame
            for frame in selected
        ], dtype=int),
    )


def combine_wurtzite_polarity_results(
        results: Sequence[WurtzitePolarityResult], request: WurtzitePolarityRequest
) -> WurtzitePolarityResult:
    def combine(name: str) -> pd.DataFrame:
        values = [getattr(result, name) for result in results if not getattr(result, name).empty]
        return pd.concat(values, ignore_index=True) if values else pd.DataFrame()

    table = combine("table")
    return WurtzitePolarityResult(
        table=table,
        request=request,
        centers=combine("centers"),
        neighbors=combine("neighbors"),
        neighbor_geometry=combine("neighbor_geometry"),
        apical_neighbors=combine("apical_neighbors"),
        basal_neighbors=combine("basal_neighbors"),
        summary=(
            pd.DataFrame([_summary_for_frame(group) for _, group in table.groupby("frame_index", sort=True)])
            if not table.empty else pd.DataFrame()
        ),
        proton_summary=summarize_by_proton_proximity(table),
        frame_indices=np.concatenate([result.frame_indices for result in results]) if results else np.empty(0,
                                                                                                            dtype=int),
        iterations=np.concatenate([result.iterations for result in results]) if results else np.empty(0, dtype=int),
    )


@register_task("get-wurtzite-polarity", label="Four-fold Wurtzite Polarity")
class WurtzitePolarityTask(AnalysisTask):
    """Calculate site polarity from the neighbor module's normalized tables."""

    required_data = TrajectoryData
    supports_selective_streaming = True
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
        return [PresentationSpec(renderer="table", label="Wurtzite polarity", view_type="table")]

    def run(self, data, request: WurtzitePolarityRequest, reporter=None):
        _ = reporter
        return calculate_polarity_from_trajectory(data, request)

    def run_stream(self, frames, request: WurtzitePolarityRequest, reporter=None):
        _validate_request(request)
        requested = None if request.frames is None else [int(v) for v in request.frames][:: int(request.every)]
        requested_set = set(requested or ())
        reference_basal: dict[int, float] = {}
        pending: list[tuple[int, WurtziteNeighborResult]] = []
        results: list[WurtzitePolarityResult] = []
        seen: set[int] = set()
        processed = 0
        for data in frames:
            processed += 1
            trajectory, charges = _charges_for_request(data, request)
            source_frame = _source_frame(trajectory, 0)
            seen.add(source_frame)
            keep = source_frame in requested_set if requested is not None else source_frame % int(request.every) == 0
            need_reference = source_frame == int(request.reference_frame)
            if keep or need_reference:
                frame_request = WurtzitePolarityRequest(**{
                    **vars(request), "frames": [0], "every": 1, "reference_frame": source_frame,
                })
                neighbor_result = extract_wurtzite_neighbors(
                    trajectory,
                    frame_request,
                    charges=charges,
                    frame_indices=[0],
                    preserve_source_frame_indices=True,
                )
                raw_result = calculate_wurtzite_polarity(neighbor_result, frame_request)
                if need_reference:
                    reference_basal = dict(zip(
                        raw_result.table["site_atom_id"].astype(int),
                        raw_result.table["mean_basal_bond_c (angstrom)"].astype(float),
                        strict=False,
                    ))
                    for pending_frame, pending_neighbors in pending:
                        pending_request = WurtzitePolarityRequest(**{
                            **vars(request), "reference_frame": int(request.reference_frame)
                        })
                        results.append(calculate_wurtzite_polarity(
                            pending_neighbors, pending_request, reference_basal=reference_basal
                        ))
                    pending.clear()
                if keep:
                    if reference_basal or source_frame == int(request.reference_frame):
                        results.append(calculate_wurtzite_polarity(
                            neighbor_result, request, reference_basal=reference_basal
                        ))
                    else:
                        pending.append((source_frame, neighbor_result))
            if callable(reporter):
                reporter("stream", processed, 0, "Calculating four-fold wurtzite polarity")
        if pending:
            raise ValueError(f"Reference frame {request.reference_frame} is required for basal changes.")
        if requested is not None:
            missing = [value for value in requested if value not in seen]
            if missing:
                raise ValueError(f"Requested frame(s) not found in trajectory: {missing}.")
        return combine_wurtzite_polarity_results(results, request)


__all__ = [
    "EA_TO_DEBYE",
    "POLARITY_COLUMNS",
    "POLARITY_CSV_COLUMNS",
    "WurtzitePolarityRequest",
    "WurtzitePolarityResult",
    "WurtzitePolarityTask",
    "calculate_polarity_from_trajectory",
    "calculate_wurtzite_polarity",
    "combine_wurtzite_polarity_results",
    "polarity_csv_table",
    "summarize_by_proton_proximity",
]
