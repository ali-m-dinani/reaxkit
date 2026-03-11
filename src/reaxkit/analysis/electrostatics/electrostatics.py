"""Engine-agnostic electrostatics analysis tasks."""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field
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
from reaxkit.presentation.specs import PresentationSpec
from reaxkit.utils.numerical.numerical_calcs import find_zero_crossings

Scope = Literal["total", "local"]
Mode = Literal["dipole", "polarization"]
VolumeMethod = Literal["hull", "bbox", "cell"]
AggregateKind = Optional[Literal["mean", "max", "min", "last"]]
FieldDirection = Literal["x", "y", "z"]
DipoleOrPolarizationDirection = Literal["mu_x", "mu_y", "mu_z", "p_x", "p_y", "p_z"]


@dataclass
class DipoleRequest(BaseRequest):
    """Request for dipole analysis.

    Parameters
    ----------
    scope
        ``"total"`` computes one dipole vector per frame for the whole system.
        ``"local"`` computes one dipole vector per selected core atom and frame.
        Example: ``scope="local"``.
    atom_ids
        Optional core atom-id filter used in local mode.
        Example: ``[1, 5, 9]``.
    atom_types
        Optional core atom-type filter used in local mode when ``atom_ids`` is
        not provided. Example: ``["O", "H"]``.
    frames
        Optional frame indices to include. If omitted, all frames are included.
        Example: ``[0, 10, 20]``.
    every
        Frame stride after selection. Example: ``every=5`` keeps every fifth
        selected frame.
    """

    scope: Scope = dc_field(
        default="total",
        metadata={
            'label': 'Scope',
            'help': "Dipole evaluation mode: 'total' (whole-system) or 'local' (per core atom). Example: 'local'.",
            'choices': ['total', 'local'],
        },
    )
    atom_ids: Optional[Sequence[int]] = dc_field(
        default=None,
        metadata={
            'label': 'Atom Ids',
            'help': "Optional core atom-id filter used in local mode. Example: [1, 5, 9].",
            'units': 'index',
        },
    )
    atom_types: Optional[Sequence[str]] = dc_field(
        default=None,
        metadata={
            'label': 'Atom Types',
            'help': "Optional core atom-type filter in local mode (used when atom_ids is not set). Example: ['O', 'H'].",
        },
    )
    frames: Optional[Sequence[int]] = dc_field(
        default=None,
        metadata={
            'label': 'Frames',
            'help': "Optional frame indices to include. Example: [0, 10, 20].",
            'units': 'frame_index',
        },
    )
    every: int = dc_field(
        default=1,
        metadata={
            'label': 'Every',
            'help': "Stride for selected frames. Example: every=5.",
            'min': 1,
            'units': 'frames',
        },
    )


@dataclass
class DipoleResult(BaseResult):
    """Dipole analysis result.

    Output structure
    ----------------
    - ``request``: the :class:`DipoleRequest` used to compute this result.
    - ``table``: pandas.DataFrame with dipole components per selected frame
      (total mode) or per selected core atom and frame (local mode).

    Common columns
    --------------
    - ``frame_index``: source frame index.
    - ``iter``: simulation iteration mapped to the frame.
    - ``mu_x (debye)``, ``mu_y (debye)``, ``mu_z (debye)``: dipole components.

    Local-mode extra columns
    ------------------------
    - ``core_atom_id``: selected core atom id.
    - ``core_atom_type``: selected core atom type.

    Example
    -------
    A row like ``frame_index=12, iter=1200, mu_z (debye)=1.45`` means the
    z-component dipole at that frame is 1.45 Debye.
    """

    table: pd.DataFrame
    request: DipoleRequest


@dataclass
class PolarizationRequest(BaseRequest):
    """Request for polarization analysis.

    Parameters
    ----------
    scope
        ``"total"`` computes one polarization vector per frame for the whole
        system. ``"local"`` computes local polarization around selected core
        atoms. Example: ``scope="local"``.
    atom_ids
        Optional core atom-id filter used in local mode.
        Example: ``[1, 5, 9]``.
    atom_types
        Optional core atom-type filter used in local mode when ``atom_ids`` is
        not provided. Example: ``["O", "H"]``.
    frames
        Optional frame indices to include. If omitted, all frames are included.
        Example: ``[0, 10, 20]``.
    every
        Frame stride after selection. Example: ``every=5``.
    volume_method
        Volume estimator for polarization normalization.
        ``"hull"``: convex hull, ``"bbox"``: axis-aligned bounding box,
        ``"cell"``: simulation cell volume (total mode).
        Example: ``volume_method="hull"``.
    """

    scope: Scope = dc_field(
        default="total",
        metadata={
            'label': 'Scope',
            'help': "Polarization evaluation mode: 'total' (whole-system) or 'local' (per core atom). Example: 'local'.",
            'choices': ['total', 'local'],
        },
    )
    atom_ids: Optional[Sequence[int]] = dc_field(
        default=None,
        metadata={
            'label': 'Atom Ids',
            'help': "Optional core atom-id filter used in local mode. Example: [1, 5, 9].",
            'units': 'index',
        },
    )
    atom_types: Optional[Sequence[str]] = dc_field(
        default=None,
        metadata={
            'label': 'Atom Types',
            'help': "Optional core atom-type filter in local mode (used when atom_ids is not set). Example: ['O', 'H'].",
        },
    )
    frames: Optional[Sequence[int]] = dc_field(
        default=None,
        metadata={
            'label': 'Frames',
            'help': "Optional frame indices to include. Example: [0, 10, 20].",
            'units': 'frame_index',
        },
    )
    every: int = dc_field(
        default=1,
        metadata={
            'label': 'Every',
            'help': "Stride for selected frames. Example: every=5.",
            'min': 1,
            'units': 'frames',
        },
    )
    volume_method: Optional[VolumeMethod] = dc_field(
        default=None,
        metadata={
            'label': 'Volume Method',
            'help': "Volume estimator for polarization normalization. Examples: 'hull', 'bbox', 'cell'.",
            'choices': ['hull', 'bbox', 'cell'],
        },
    )


@dataclass
class PolarizationResult(BaseResult):
    """Polarization analysis result.

    Output structure
    ----------------
    - ``request``: the :class:`PolarizationRequest` used for computation.
    - ``table``: pandas.DataFrame with polarization components and volume terms
      per selected frame (total mode) or per selected core atom and frame
      (local mode).

    Common columns
    --------------
    - ``frame_index``: source frame index.
    - ``iter``: simulation iteration mapped to the frame.
    - ``P_x (uC/cm^2)``, ``P_y (uC/cm^2)``, ``P_z (uC/cm^2)``: polarization
      components.
    - ``volume (angstrom^3)``: normalization volume used for each row.

    Local-mode extra columns
    ------------------------
    - ``core_atom_id``: selected core atom id.
    - ``core_atom_type``: selected core atom type.

    Example
    -------
    A row like ``frame_index=12, iter=1200, P_z (uC/cm^2)=18.7`` means the
    z-component polarization at that frame is 18.7 uC/cm^2.
    """

    table: pd.DataFrame
    request: PolarizationRequest


@dataclass
class PolarizationFieldRequest(BaseRequest):
    """Request for polarization/dipole versus electric-field hysteresis analysis.

    Parameters
    ----------
    frames
        Optional frame indices to include. If omitted, all available frames are used.
        Example: ``[0, 10, 20]``.
    every
        Frame stride after selection. Example: ``every=5``.
    aggregate
        Optional aggregation applied after pairing field and response values.
        Choices:
        - ``mean``: average all rows sharing the same field value
        - ``max``: maximum over grouped rows
        - ``min``: minimum over grouped rows
        - ``last``: keep the last row by iteration per field value
        - ``None``: no aggregation
    field_direction
        Electric-field axis used for hysteresis x-axis, mapped to
        ``field_x`` / ``field_y`` / ``field_z``. Example: ``"z"``.
    dipole_or_polaization_direction
        Response axis used for hysteresis y-axis.
        - ``mu_x``/``mu_y``/``mu_z`` map to dipole components
        - ``p_x``/``p_y``/``p_z`` map to polarization components
        Example: ``"p_z"``.
    """

    frames: Optional[Sequence[int]] = dc_field(
        default=None,
        metadata={
            'label': 'Frames',
            'help': "Optional frame indices to include. Example: [0, 10, 20].",
            'units': 'frame_index',
        },
    )
    every: int = dc_field(
        default=1,
        metadata={
            'label': 'Every',
            'help': "Stride for selected frames. Example: every=5.",
            'min': 1,
            'units': 'frames',
        },
    )
    aggregate: AggregateKind = dc_field(
        default=None,
        metadata={
            'label': 'Aggregate',
            'help': "Optional aggregation over rows sharing the same field value. Example: 'mean'.",
            'choices': ['mean', 'max', 'min', 'last'],
        },
    )
    field_direction: FieldDirection = dc_field(
        default="z",
        metadata={
            'label': 'Field Direction',
            'help': "Field axis used as hysteresis x-axis. 'z' maps to field_z. Example: 'z'.",
            'choices': ['x', 'y', 'z'],
        },
    )
    dipole_or_polaization_direction: DipoleOrPolarizationDirection = dc_field(
        default="p_z",
        metadata={
            'label': 'Dipole Or Polaization Direction',
            'help': "Response axis for hysteresis. Example: 'mu_z' for dipole-z or 'p_z' for polarization-z.",
            'choices': ['mu_x', 'mu_y', 'mu_z', 'p_x', 'p_y', 'p_z'],
        },
    )


@dataclass
class PolarizationFieldResult(BaseResult):
    """Hysteresis analysis result for field-response curves.

    Output structure
    ----------------
    - ``request``: the :class:`PolarizationFieldRequest` used to build the result.
    - ``full_table``: per-frame joined table with field and response values.
    - ``aggregated_table``: grouped table after optional aggregation.
    - ``polarization_zero_crossings``: x-values where y crosses zero.
    - ``field_zero_crossings``: y-values where x crosses zero.

    Table semantics
    ---------------
    ``full_table`` always contains:
    - ``iter`` and ``frame_index`` from polarization series
    - dipole columns ``mu_x/y/z (debye)``
    - polarization columns ``P_x/y/z (uC/cm^2)``
    - one field column chosen by ``request.field_direction``:
      ``field_x`` or ``field_y`` or ``field_z`` (scaled to MV/cm)

    Example
    -------
    With ``field_direction='z'`` and ``dipole_or_polaization_direction='p_z'``,
    hysteresis is computed from ``x=field_z`` and ``y=P_z (uC/cm^2)``.
    """

    full_table: pd.DataFrame
    aggregated_table: pd.DataFrame
    polarization_zero_crossings: list[float]
    field_zero_crossings: list[float]
    request: PolarizationFieldRequest


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
    return [np.where(mat[i] > 0)[0] for i in range(n_atoms)]


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
            if neigh.size > 0:
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
        reporter=reporter,
    )
    if table.empty:
        return table
    return table.sort_values(["frame_index", "core_atom_id"], kind="stable").reset_index(drop=True)


@register_task("dipole")
class DipoleTask(AnalysisTask):
    """Compute dipole series as total or local."""

    required_data = ElectrostaticsData

    @staticmethod
    def recommended_presentations(
        _result: DipoleResult,
        payload: dict[str, Any],
    ) -> list[PresentationSpec]:
        rows = payload.get("table")
        if not isinstance(rows, list) or not rows:
            return [PresentationSpec(renderer="table", label="Table", view_type="table")]
        sample = rows[0] if isinstance(rows[0], dict) else {}
        if "iter" not in sample or "mu_z (debye)" not in sample:
            return [PresentationSpec(renderer="table", label="Table", view_type="table")]
        group_by = "core_atom_id" if "core_atom_id" in sample else ""
        return [
            PresentationSpec(renderer="table", label="Table", view_type="table"),
            PresentationSpec(
                renderer="single_plot",
                label="mu_z (debye) vs iter",
                mapping={"x_col": "iter", "y_col": "mu_z (debye)", "group_by_col": group_by},
                options={
                    "title": "Dipole",
                    "xlabel": "iter",
                    "ylabel": "mu_z (debye)",
                    "legend": True,
                },
                view_type="plot2d",
            ),
        ]

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
            reporter=reporter,
        )
        return DipoleResult(table=out, request=request)


@register_task("polarization")
class PolarizationTask(AnalysisTask):
    """Compute polarization series as total or local."""

    required_data = ElectrostaticsData

    @staticmethod
    def recommended_presentations(
        _result: PolarizationResult,
        payload: dict[str, Any],
    ) -> list[PresentationSpec]:
        rows = payload.get("table")
        if not isinstance(rows, list) or not rows:
            return [PresentationSpec(renderer="table", label="Table", view_type="table")]
        sample = rows[0] if isinstance(rows[0], dict) else {}
        if "iter" not in sample or "P_z (uC/cm^2)" not in sample:
            return [PresentationSpec(renderer="table", label="Table", view_type="table")]
        group_by = "core_atom_id" if "core_atom_id" in sample else ""
        return [
            PresentationSpec(renderer="table", label="Table", view_type="table"),
            PresentationSpec(
                renderer="single_plot",
                label="P_z (uC/cm^2) vs iter",
                mapping={"x_col": "iter", "y_col": "P_z (uC/cm^2)", "group_by_col": group_by},
                options={
                    "title": "Polarization",
                    "xlabel": "iter",
                    "ylabel": "P_z (uC/cm^2)",
                    "legend": True,
                },
                view_type="plot2d",
            ),
        ]

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
            reporter=reporter,
        )
        return PolarizationResult(table=out, request=request)


@register_task("polarization_field")
class PolarizationFieldTask(AnalysisTask):
    """Compute polarization-field data and hysteresis roots."""

    required_data = ElectrostaticsData

    @staticmethod
    def recommended_presentations(
        _result: PolarizationFieldResult,
        payload: dict[str, Any],
    ) -> list[PresentationSpec]:
        views: list[PresentationSpec] = [
            PresentationSpec(
                renderer="table",
                label="Full Table",
                options={"source_key": "full_table"},
                view_type="table",
            ),
            PresentationSpec(
                renderer="table",
                label="Aggregated Table",
                options={"source_key": "aggregated_table"},
                view_type="table",
            ),
        ]

        agg_rows = payload.get("aggregated_table")
        req_payload = payload.get("request") if isinstance(payload.get("request"), dict) else {}
        if not isinstance(agg_rows, list) or not agg_rows:
            return views

        sample = agg_rows[0] if isinstance(agg_rows[0], dict) else {}
        field_dir = str(req_payload.get("field_direction", "z")).strip().lower()
        y_key = str(req_payload.get("dipole_or_polaization_direction", "p_z")).strip().lower()
        x_col = f"field_{field_dir}" if field_dir in {"x", "y", "z"} else "field_z"
        y_map = {
            "mu_x": "mu_x (debye)",
            "mu_y": "mu_y (debye)",
            "mu_z": "mu_z (debye)",
            "p_x": "P_x (uC/cm^2)",
            "p_y": "P_y (uC/cm^2)",
            "p_z": "P_z (uC/cm^2)",
        }
        y_col = y_map.get(y_key, "P_z (uC/cm^2)")
        if x_col not in sample or y_col not in sample:
            return views

        views.append(
            PresentationSpec(
                renderer="single_plot",
                label=f"{y_col} vs {x_col}",
                mapping={"x_col": x_col, "y_col": y_col, "group_by_col": ""},
                options={
                    "title": "Hysteresis",
                    "xlabel": x_col,
                    "ylabel": y_col,
                    "legend": False,
                    "source_key": "aggregated_table",
                },
                view_type="plot2d",
            )
        )
        return views

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
        field_col = f"field_{str(request.field_direction).strip().lower()}"
        field = _field_component_series(
            data.electric_field,
            component=field_col,
            target_iters=iters,
        )
        field = np.asarray(field, dtype=float) * float(const("electric_field_VA_to_MVcm"))

        full = pol.copy()
        full[field_col] = field

        if request.aggregate is None:
            agg = full.copy()
        else:
            if request.aggregate not in {"mean", "max", "min", "last"}:
                raise ValueError("aggregate must be one of: mean|max|min|last (or None).")
            group_col = field_col
            if request.aggregate == "mean":
                agg = full.groupby(group_col, as_index=False).mean(numeric_only=True)
            elif request.aggregate == "max":
                agg = full.groupby(group_col, as_index=False).max(numeric_only=True)
            elif request.aggregate == "min":
                agg = full.groupby(group_col, as_index=False).min(numeric_only=True)
            else:
                agg = full.sort_values("iter").groupby(group_col, as_index=False).tail(1).reset_index(drop=True)
            # Keep aggregated table column order aligned with full table.
            preferred_cols = [c for c in full.columns if c in agg.columns]
            trailing_cols = [c for c in agg.columns if c not in preferred_cols]
            agg = agg.loc[:, preferred_cols + trailing_cols]

        y_map = {
            "mu_x": "mu_x (debye)",
            "mu_y": "mu_y (debye)",
            "mu_z": "mu_z (debye)",
            "p_x": "P_x (uC/cm^2)",
            "p_y": "P_y (uC/cm^2)",
            "p_z": "P_z (uC/cm^2)",
        }
        y_col = y_map.get(str(request.dipole_or_polaization_direction).strip().lower())
        if y_col is None:
            raise KeyError(
                f"Unsupported dipole_or_polaization_direction='{request.dipole_or_polaization_direction}'."
            )
        if field_col not in agg.columns or y_col not in agg.columns:
            raise KeyError(f"Missing required columns '{field_col}' or '{y_col}' in aggregated data.")

        x = agg[field_col].to_numpy(float)
        y = agg[y_col].to_numpy(float)
        y_zeros = find_zero_crossings(x, y)
        x_zeros = find_zero_crossings(y, x)

        return PolarizationFieldResult(
            full_table=full.reset_index(drop=True),
            aggregated_table=agg.reset_index(drop=True),
            polarization_zero_crossings=y_zeros,
            field_zero_crossings=x_zeros,
            request=request,
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
