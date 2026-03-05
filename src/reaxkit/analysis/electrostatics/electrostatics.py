"""Engine-agnostic electrostatics analysis tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull

from reaxkit.analysis.base import AnalysisTask
from reaxkit.core.analysis_task_registry import register_task
from reaxkit.domain.base_request import BaseRequest
from reaxkit.domain.base_result import BaseResult
from reaxkit.domain.data_models import ConnectivityData, ElectrostaticsData, ElectricFieldData
from reaxkit.engine.reaxff.adapter import (
    _charges_from_fort7_handler,
    _connectivity_from_fort7_handler,
    _trajectory_from_xmolout_handler,
)
from reaxkit.core.constants import const
from reaxkit.utils.numerical.numerical_calcs import find_zero_crossings

Scope = Literal["total", "local"]
Mode = Literal["dipole", "polarization"]
VolumeMethod = Literal["hull", "bbox", "cell"]
AggregateKind = Optional[Literal["mean", "max", "min", "last"]]


@dataclass
class DipoleRequest(BaseRequest):
    """Request for dipole analysis."""

    scope: Scope = "total"
    atom_ids: Optional[Sequence[int]] = None
    atom_types: Optional[Sequence[str]] = None
    frames: Optional[Sequence[int]] = None
    every: int = 1
    min_bo: float = 0.0
    scale_neighbor_charges: bool = True


@dataclass
class DipoleResult(BaseResult):
    """Result of dipole analysis."""

    table: pd.DataFrame


@dataclass
class PolarizationRequest(BaseRequest):
    """Request for polarization analysis."""

    scope: Scope = "total"
    atom_ids: Optional[Sequence[int]] = None
    atom_types: Optional[Sequence[str]] = None
    frames: Optional[Sequence[int]] = None
    every: int = 1
    volume_method: Optional[VolumeMethod] = None
    min_bo: float = 0.0
    scale_neighbor_charges: bool = True


@dataclass
class PolarizationResult(BaseResult):
    """Result of polarization analysis."""

    table: pd.DataFrame


@dataclass
class PolarizationFieldRequest(BaseRequest):
    """Request for polarization-electric field hysteresis analysis."""

    frames: Optional[Sequence[int]] = None
    every: int = 1
    aggregate: AggregateKind = None
    field_component: str = "field_z"
    x_variable: str = "field_z"
    y_variable: str = "P_z (uC/cm^2)"
    field_scale: float = const("electric_field_VA_to_MVcm")


@dataclass
class PolarizationFieldResult(BaseResult):
    """Result of hysteresis analysis."""

    full_table: pd.DataFrame
    aggregated_table: pd.DataFrame
    polarization_zero_crossings: list[float]
    field_zero_crossings: list[float]


def _minimum_image_delta(delta: np.ndarray, box_lengths: np.ndarray) -> np.ndarray:
    delta = np.asarray(delta, dtype=float)
    box_lengths = np.asarray(box_lengths, dtype=float)
    if box_lengths.shape != (3,):
        return delta
    if np.any(~np.isfinite(box_lengths)) or np.any(box_lengths <= 0):
        return delta
    return delta - box_lengths * np.round(delta / box_lengths)


def _convex_hull_volume(coords: np.ndarray) -> float:
    coords = np.asarray(coords, float)
    if coords.ndim != 2 or coords.shape[1] != 3 or coords.shape[0] < 4:
        return np.nan
    try:
        return float(ConvexHull(coords).volume)
    except Exception:
        return np.nan


def _bbox_volume(coords: np.ndarray) -> float:
    coords = np.asarray(coords, float)
    if coords.ndim != 2 or coords.shape[1] != 3 or coords.shape[0] == 0:
        return np.nan
    mn = np.nanmin(coords, axis=0)
    mx = np.nanmax(coords, axis=0)
    if np.any(~np.isfinite(mn)) or np.any(~np.isfinite(mx)):
        return np.nan
    d = mx - mn
    if np.any(d < 0):
        return np.nan
    return float(d[0] * d[1] * d[2])


def _cell_volume(cell_lengths: Optional[np.ndarray], frame_index: int) -> float:
    if cell_lengths is None:
        return np.nan
    v = np.asarray(cell_lengths[int(frame_index)], dtype=float)
    if v.shape != (3,) or np.any(~np.isfinite(v)) or np.any(v <= 0):
        return np.nan
    return float(v[0] * v[1] * v[2])


def _frame_indices(n_frames: int, frames: Optional[Sequence[int]], every: int) -> list[int]:
    idx = list(range(n_frames)) if frames is None else [int(i) for i in frames]
    return [i for i in idx if 0 <= i < n_frames][:: max(1, int(every))]


def _atom_selector(
    elements: Sequence[str],
    atom_ids: Sequence[int],
    *,
    selected_atom_ids: Optional[Sequence[int]],
    selected_atom_types: Optional[Sequence[str]],
) -> np.ndarray:
    if selected_atom_ids:
        chosen = {int(v) for v in selected_atom_ids}
        return np.array([int(aid) in chosen for aid in atom_ids], dtype=bool)
    if selected_atom_types:
        chosen = {str(v) for v in selected_atom_types}
        return np.array([str(e) in chosen for e in elements], dtype=bool)
    return np.ones(len(atom_ids), dtype=bool)


def _bo_frame_from_connectivity(connectivity: ConnectivityData, frame_index: int) -> np.ndarray:
    bo = connectivity.bond_orders
    if bo is None:
        raise ValueError("Local electrostatics requires ConnectivityData.bond_orders or connectivity.")
    if isinstance(bo, np.ndarray):
        if bo.ndim != 3:
            raise ValueError("bond_orders ndarray must have shape (n_frames, n_atoms, n_atoms).")
        return np.asarray(bo[int(frame_index)], dtype=float)
    if isinstance(bo, (list, tuple)):
        fr = bo[int(frame_index)]
        if hasattr(fr, "toarray"):
            return np.asarray(fr.toarray(), dtype=float)
        if hasattr(fr, "todense"):
            return np.asarray(fr.todense(), dtype=float)
        return np.asarray(fr, dtype=float)
    raise TypeError("Unsupported ConnectivityData.bond_orders type.")


def _neighbors_from_connectivity(
    connectivity: ConnectivityData,
    frame_index: int,
    *,
    n_atoms: int,
    min_bo: float,
) -> list[np.ndarray]:
    if connectivity.connectivity is not None:
        conn = connectivity.connectivity
        if isinstance(conn, np.ndarray):
            if conn.ndim == 2:
                mat = np.asarray(conn, dtype=float)
            elif conn.ndim == 3:
                mat = np.asarray(conn[int(frame_index)], dtype=float)
            else:
                raise ValueError("connectivity must be 2D or 3D when ndarray.")
        elif isinstance(conn, (list, tuple)):
            fr = conn[int(frame_index)]
            if hasattr(fr, "toarray"):
                mat = np.asarray(fr.toarray(), dtype=float)
            elif hasattr(fr, "todense"):
                mat = np.asarray(fr.todense(), dtype=float)
            else:
                mat = np.asarray(fr, dtype=float)
        else:
            mat = np.asarray(conn, dtype=float)
        if mat.shape != (n_atoms, n_atoms):
            raise ValueError("connectivity shape must match atom count.")
        return [np.where(mat[i] > 0)[0] for i in range(n_atoms)]

    mat = _bo_frame_from_connectivity(connectivity, frame_index)
    if mat.shape != (n_atoms, n_atoms):
        raise ValueError("bond-order frame must be square and match atom count.")
    thr = float(min_bo)
    return [np.where(mat[i] > thr)[0] for i in range(n_atoms)]


def _series_total(
    *,
    positions: np.ndarray,
    charges: np.ndarray,
    iterations: np.ndarray,
    frame_idx: Sequence[int],
    mode: Mode,
    volume_method: VolumeMethod,
    cell_lengths: Optional[np.ndarray],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for fi in frame_idx:
        coords = positions[int(fi)].astype(float)
        q = charges[int(fi)].astype(float)
        mu_ea = (coords * q[:, None]).sum(axis=0)
        mu_debye = mu_ea * const("ea_to_debye")

        row: dict[str, Any] = {
            "frame_index": int(fi),
            "iter": int(iterations[int(fi)]),
            "mu_x (debye)": float(mu_debye[0]),
            "mu_y (debye)": float(mu_debye[1]),
            "mu_z (debye)": float(mu_debye[2]),
        }

        if mode == "polarization":
            if volume_method == "cell":
                volume = _cell_volume(cell_lengths, int(fi))
            elif volume_method == "bbox":
                volume = _bbox_volume(coords)
            else:
                volume = _convex_hull_volume(coords)

            if np.isfinite(volume) and volume > 0:
                p_vec = mu_ea / volume * const("ea3_to_uC_cm2")
                row["P_x (uC/cm^2)"] = float(p_vec[0])
                row["P_y (uC/cm^2)"] = float(p_vec[1])
                row["P_z (uC/cm^2)"] = float(p_vec[2])
            else:
                row["P_x (uC/cm^2)"] = np.nan
                row["P_y (uC/cm^2)"] = np.nan
                row["P_z (uC/cm^2)"] = np.nan
            row["volume (angstrom^3)"] = float(volume)
        rows.append(row)
    return pd.DataFrame(rows)


def _series_local(
    *,
    positions: np.ndarray,
    charges: np.ndarray,
    elements: Sequence[str],
    atom_ids: Sequence[int],
    iterations: np.ndarray,
    frame_idx: Sequence[int],
    cell_lengths: Optional[np.ndarray],
    cell_angles: Optional[np.ndarray],
    connectivity: ConnectivityData,
    mode: Mode,
    volume_method: Optional[VolumeMethod],
    selected_atom_ids: Optional[Sequence[int]],
    selected_atom_types: Optional[Sequence[str]],
    min_bo: float,
    scale_neighbor_charges: bool,
    reporter=None,
) -> pd.DataFrame:
    mask = _atom_selector(
        elements,
        atom_ids,
        selected_atom_ids=selected_atom_ids,
        selected_atom_types=selected_atom_types,
    )
    core_idx = np.where(mask)[0]
    if core_idx.size == 0:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    total = len(frame_idx)
    for step_i, fi in enumerate(frame_idx, start=1):
        coords = positions[int(fi)].astype(float)
        q = charges[int(fi)].astype(float)
        neighbors = _neighbors_from_connectivity(
            connectivity,
            int(fi),
            n_atoms=coords.shape[0],
            min_bo=float(min_bo),
        )

        box_lengths = None
        if cell_lengths is not None:
            bl = np.asarray(cell_lengths[int(fi)], dtype=float)
            if bl.shape == (3,) and np.all(np.isfinite(bl)) and np.all(bl > 0):
                is_ortho = True
                if cell_angles is not None:
                    ang = np.asarray(cell_angles[int(fi)], dtype=float)
                    if ang.shape == (3,):
                        is_ortho = bool(np.all(np.isclose(ang, 90.0, atol=1.0e-3)))
                if is_ortho:
                    box_lengths = bl

        for ci in core_idx:
            neigh = np.asarray(neighbors[int(ci)], dtype=int)
            neigh = neigh[(neigh >= 0) & (neigh < coords.shape[0]) & (neigh != int(ci))]

            cluster = np.concatenate(([int(ci)], neigh))
            cluster_coords = coords[cluster]
            rel = cluster_coords - cluster_coords[0:1]
            if box_lengths is not None:
                rel = _minimum_image_delta(rel, box_lengths)

            cluster_q = q[cluster].copy()
            if scale_neighbor_charges and neigh.size > 0:
                cluster_q[1:] = cluster_q[1:] / float(neigh.size)

            mu_ea = (rel * cluster_q[:, None]).sum(axis=0)
            mu_debye = mu_ea * const("ea_to_debye")

            row: dict[str, Any] = {
                "frame_index": int(fi),
                "iter": int(iterations[int(fi)]),
                "core_atom_id": int(atom_ids[int(ci)]),
                "core_atom_type": str(elements[int(ci)]),
                "mu_x (debye)": float(mu_debye[0]),
                "mu_y (debye)": float(mu_debye[1]),
                "mu_z (debye)": float(mu_debye[2]),
            }

            if mode == "polarization":
                vm = volume_method or "bbox"
                if vm == "hull":
                    volume = _convex_hull_volume(rel)
                else:
                    volume = _bbox_volume(rel)

                if np.isfinite(volume) and volume > 0:
                    p_vec = mu_ea / volume * const("ea3_to_uC_cm2")
                    row["P_x (uC/cm^2)"] = float(p_vec[0])
                    row["P_y (uC/cm^2)"] = float(p_vec[1])
                    row["P_z (uC/cm^2)"] = float(p_vec[2])
                else:
                    row["P_x (uC/cm^2)"] = np.nan
                    row["P_y (uC/cm^2)"] = np.nan
                    row["P_z (uC/cm^2)"] = np.nan
                row["volume (angstrom^3)"] = float(volume)
            rows.append(row)
        if reporter:
            reporter("analyze", step_i, total, f"Computing local {mode}")
    return pd.DataFrame(rows)


def _field_component_series(
    field_data: ElectricFieldData,
    *,
    component: str,
    target_iters: np.ndarray,
) -> np.ndarray:
    def _extract(values: np.ndarray, components: Sequence[str], wanted: str) -> Optional[np.ndarray]:
        vals = np.asarray(values, dtype=float)
        names = [str(c) for c in components]
        if wanted not in names:
            return None
        if vals.ndim == 1:
            if len(names) != 1:
                raise ValueError("1D electric field values require exactly one component label.")
            return vals
        if vals.ndim == 2:
            j = names.index(wanted)
            if j >= vals.shape[1]:
                raise ValueError("Electric field values second dimension does not match component labels.")
            return vals[:, j]
        raise ValueError("Electric field values must be 1D or 2D.")

    series = _extract(field_data.applied_field_values, field_data.applied_field_components, component)
    if series is None:
        series = _extract(field_data.field_energy_values, field_data.field_energy_components, component)
    if series is None:
        all_components = [
            *[str(c) for c in field_data.applied_field_components],
            *[str(c) for c in field_data.field_energy_components],
        ]
        raise KeyError(f"Electric field component '{component}' not found in components={all_components}.")

    if field_data.sampled_field_iterations is None:
        if len(series) < len(target_iters):
            raise ValueError("Electric field series shorter than selected polarization frames.")
        return np.asarray(series[: len(target_iters)], dtype=float)

    field_iters = np.asarray(field_data.sampled_field_iterations, dtype=int).reshape(-1)
    if field_iters.shape[0] != series.shape[0]:
        raise ValueError("ElectricFieldData.sampled_field_iterations length must match electric field samples.")
    by_iter = pd.Series(np.asarray(series, dtype=float), index=field_iters)
    return by_iter.reindex(target_iters).to_numpy(dtype=float)


def _electrostatics_data_from_handlers(xh, f7) -> ElectrostaticsData:
    traj = _trajectory_from_xmolout_handler(xh)
    n_frames, n_atoms = traj.positions.shape[:2]
    iters = np.asarray(traj.iterations if traj.iterations is not None else np.arange(n_frames), dtype=int)

    charge_data = _charges_from_fort7_handler(f7, simulation=traj.simulation)
    charges = np.asarray(charge_data.charges, dtype=float)
    if charges.shape != (n_frames, n_atoms):
        raise ValueError(
            "ChargeData.charges from fort.7 must have shape (n_frames, n_atoms) matching trajectory."
        )
    charge_data.simulation = traj.simulation
    charge_data.iterations = iters

    conn_data = _connectivity_from_fort7_handler(f7)
    conn_data.atom_ids = np.asarray(traj.atom_ids, dtype=int)
    conn_data.elements = list(traj.elements)
    conn_data.simulation = traj.simulation
    conn_data.iterations = iters

    return ElectrostaticsData(
        trajectory=traj,
        charges=charge_data,
        connectivity=conn_data,
    )


def _run_electrostatics(
    data: ElectrostaticsData,
    *,
    mode: Mode,
    scope: Scope,
    atom_ids: Optional[Sequence[int]],
    atom_types: Optional[Sequence[str]],
    frames: Optional[Sequence[int]],
    every: int,
    volume_method: Optional[VolumeMethod],
    min_bo: float,
    scale_neighbor_charges: bool,
    reporter=None,
) -> pd.DataFrame:
    positions = np.asarray(data.trajectory.positions, dtype=float)
    charges = np.asarray(data.charges.charges, dtype=float)
    if positions.ndim != 3 or positions.shape[2] != 3:
        raise ValueError("Trajectory positions must have shape (n_frames, n_atoms, 3).")
    if charges.shape != positions.shape[:2]:
        raise ValueError("ChargeData.charges must have shape (n_frames, n_atoms) matching trajectory.")

    n_frames = positions.shape[0]
    frame_idx = _frame_indices(n_frames, frames, every)
    if not frame_idx:
        return pd.DataFrame()

    if data.trajectory.iterations is not None:
        iterations = np.asarray(data.trajectory.iterations, dtype=int).reshape(-1)
    elif data.charges.iterations is not None:
        iterations = np.asarray(data.charges.iterations, dtype=int).reshape(-1)
    else:
        iterations = np.arange(n_frames, dtype=int)
    if iterations.shape[0] != n_frames:
        raise ValueError("iterations length must match n_frames.")

    elements = list(data.trajectory.elements)
    if len(elements) != positions.shape[1]:
        raise ValueError("TrajectoryData.elements length must match n_atoms.")
    atom_id_list = list(data.trajectory.atom_ids)
    if len(atom_id_list) != positions.shape[1]:
        raise ValueError("TrajectoryData.atom_ids length must match n_atoms.")

    if scope == "total":
        vm = volume_method or ("hull" if mode == "polarization" else "cell")
        return _series_total(
            positions=positions,
            charges=charges,
            iterations=iterations,
            frame_idx=frame_idx,
            mode=mode,
            volume_method=vm,
            cell_lengths=(data.trajectory.simulation.cell_lengths if data.trajectory.simulation else None),
        ).reset_index(drop=True)

    if data.connectivity is None:
        raise ValueError("Local electrostatics requires ElectrostaticsData.connectivity.")

    table = _series_local(
        positions=positions,
        charges=charges,
        elements=elements,
        atom_ids=atom_id_list,
        iterations=iterations,
        frame_idx=frame_idx,
        cell_lengths=(data.trajectory.simulation.cell_lengths if data.trajectory.simulation else None),
        cell_angles=(data.trajectory.simulation.cell_angles if data.trajectory.simulation else None),
        connectivity=data.connectivity,
        mode=mode,
        volume_method=volume_method,
        selected_atom_ids=atom_ids,
        selected_atom_types=atom_types,
        min_bo=min_bo,
        scale_neighbor_charges=scale_neighbor_charges,
        reporter=reporter,
    )
    if table.empty:
        return table
    return table.sort_values(["frame_index", "core_atom_id"], kind="stable").reset_index(drop=True)


@register_task("dipole")
class DipoleTask(AnalysisTask):
    """Compute dipole series as total or local."""

    required_data = ElectrostaticsData

    def run(self, data: ElectrostaticsData, request: DipoleRequest, reporter=None) -> DipoleResult:
        out = _run_electrostatics(
            data,
            mode="dipole",
            scope=request.scope,
            atom_ids=request.atom_ids,
            atom_types=request.atom_types,
            frames=request.frames,
            every=request.every,
            volume_method=None,
            min_bo=request.min_bo,
            scale_neighbor_charges=request.scale_neighbor_charges,
            reporter=reporter,
        )
        return DipoleResult(table=out)


@register_task("polarization")
class PolarizationTask(AnalysisTask):
    """Compute polarization series as total or local."""

    required_data = ElectrostaticsData

    def run(self, data: ElectrostaticsData, request: PolarizationRequest, reporter=None) -> PolarizationResult:
        out = _run_electrostatics(
            data,
            mode="polarization",
            scope=request.scope,
            atom_ids=request.atom_ids,
            atom_types=request.atom_types,
            frames=request.frames,
            every=request.every,
            volume_method=request.volume_method,
            min_bo=request.min_bo,
            scale_neighbor_charges=request.scale_neighbor_charges,
            reporter=reporter,
        )
        return PolarizationResult(table=out)


@register_task("polarization_field")
class PolarizationFieldTask(AnalysisTask):
    """Compute polarization-field data and hysteresis roots."""

    required_data = ElectrostaticsData

    def run(self, data: ElectrostaticsData, request: PolarizationFieldRequest) -> PolarizationFieldResult:
        if data.electric_field is None:
            raise ValueError("Polarization field analysis requires ElectrostaticsData.electric_field.")

        pol = PolarizationTask().run(
            data,
            PolarizationRequest(
                scope="total",
                frames=request.frames,
                every=request.every,
                volume_method="hull",
            ),
        ).table
        if pol.empty:
            raise ValueError("No polarization data produced for selected frames.")

        pol = pol.sort_values("iter").reset_index(drop=True)
        iters = pol["iter"].to_numpy(dtype=int)
        field = _field_component_series(
            data.electric_field,
            component=str(request.field_component),
            target_iters=iters,
        )
        field = np.asarray(field, dtype=float) * float(request.field_scale)

        full = pol.copy()
        full[str(request.field_component)] = field

        if request.aggregate is None:
            agg = full.copy()
        else:
            if request.aggregate not in {"mean", "max", "min", "last"}:
                raise ValueError("aggregate must be one of: mean|max|min|last (or None).")
            group_col = str(request.field_component)
            if request.aggregate == "mean":
                agg = full.groupby(group_col, as_index=False).mean(numeric_only=True)
            elif request.aggregate == "max":
                agg = full.groupby(group_col, as_index=False).max(numeric_only=True)
            elif request.aggregate == "min":
                agg = full.groupby(group_col, as_index=False).min(numeric_only=True)
            else:
                agg = full.sort_values("iter").groupby(group_col, as_index=False).tail(1).reset_index(drop=True)

        if request.x_variable not in agg.columns or request.y_variable not in agg.columns:
            raise KeyError(
                f"Missing required columns '{request.x_variable}' or '{request.y_variable}' in aggregated data."
            )

        x = agg[request.x_variable].to_numpy(float)
        y = agg[request.y_variable].to_numpy(float)
        y_zeros = find_zero_crossings(x, y)
        x_zeros = find_zero_crossings(y, x)

        return PolarizationFieldResult(
            full_table=full.reset_index(drop=True),
            aggregated_table=agg.reset_index(drop=True),
            polarization_zero_crossings=y_zeros,
            field_zero_crossings=x_zeros,
        )


__all__ = [
    "DipoleRequest",
    "DipoleResult",
    "DipoleTask",
    "PolarizationRequest",
    "PolarizationResult",
    "PolarizationTask",
    "PolarizationFieldRequest",
    "PolarizationFieldResult",
    "PolarizationFieldTask",
]
