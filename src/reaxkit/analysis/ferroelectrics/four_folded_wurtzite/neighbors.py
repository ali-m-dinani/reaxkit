"""Distance-based center/neighbor assignment for four-fold wurtzite sites."""

from __future__ import annotations

import itertools
import warnings
from dataclasses import dataclass, field as dc_field
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from reaxkit.analysis.base import AnalysisTask
from reaxkit.core.registry.analysis_task_registry import register_task
from reaxkit.domain.base_request import BaseRequest
from reaxkit.domain.base_result import BaseResult
from reaxkit.domain.data_models import ElectrostaticsData, TrajectoryData
from reaxkit.presentation.specs import PresentationSpec

CENTER_COLUMNS = [
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
    "neighbor_cutoff (angstrom)",
    "c_axis_x",
    "c_axis_y",
    "c_axis_z",
    "periodic_a",
    "periodic_b",
    "periodic_c",
]

NEIGHBOR_COLUMNS = [
    "frame_index",
    "iter",
    "site_atom_id",
    "site_element",
    "neighbor_rank",
    "neighbor_atom_index",
    "neighbor_atom_id",
    "neighbor_element",
    "neighbor_charge (e)",
    "neighbor_x (angstrom)",
    "neighbor_y (angstrom)",
    "neighbor_z (angstrom)",
    "neighbor_image_x (angstrom)",
    "neighbor_image_y (angstrom)",
    "neighbor_image_z (angstrom)",
    "distance (angstrom)",
    "bond_x (angstrom)",
    "bond_y (angstrom)",
    "bond_z (angstrom)",
    "bond_c (angstrom)",
    "neighbor_role",
]

CENTER_CSV_EXCLUDED_COLUMNS = {
    "proton_cutoff (angstrom)",
    "charge_source",
    "neighbor_cutoff (angstrom)",
    "c_axis_x",
    "c_axis_y",
    "c_axis_z",
    "periodic_a",
    "periodic_b",
    "periodic_c",
}


def time_after_iter(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with ``time`` immediately after ``iter`` when both exist."""

    output = frame.copy()
    if "time" not in output.columns or "iter" not in output.columns:
        return output
    columns = [column for column in output.columns if column != "time"]
    columns.insert(columns.index("iter") + 1, "time")
    return output.loc[:, columns]


def centers_csv_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Shape the compact, per-center public CSV table."""

    return time_after_iter(
        frame.drop(columns=list(CENTER_CSV_EXCLUDED_COLUMNS), errors="ignore")
    )


def neighbors_csv_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Shape the public neighbor CSV while retaining the c-axis projection."""

    return time_after_iter(frame)


@dataclass
class WurtziteNeighborRequest(BaseRequest):
    """Configure four-neighbor extraction around selected center species."""

    centers: Sequence[str] = ("Al",)
    neighbors: Sequence[str] = ("N",)
    protons: Sequence[str] = ("H",)
    charge_source: str = "auto"
    formal_charges: Mapping[str, float] = dc_field(default_factory=dict)
    neighbor_cutoff: Optional[float] = 3.0
    proton_cutoff: float = 2.0
    c_axis: Sequence[float] = (0.0, 0.0, 1.0)
    periodic: Sequence[bool] = (True, True, True)
    cell_lengths: Optional[Sequence[float]] = None
    cell_angles: Sequence[float] = (90.0, 90.0, 90.0)
    frames: Optional[Sequence[int]] = None
    every: int = 1


@dataclass
class WurtziteNeighborResult(BaseResult):
    """Center rows and normalized one-row-per-neighbor assignments."""

    table: pd.DataFrame
    request: WurtziteNeighborRequest
    centers: pd.DataFrame
    neighbors: pd.DataFrame
    frame_indices: np.ndarray
    iterations: np.ndarray

    @property
    def csv_tables(self) -> dict[str, pd.DataFrame]:
        """Expose only the two non-duplicate public CSV artifacts."""

        return {
            "centers": centers_csv_table(self.centers),
            "neighbors": neighbors_csv_table(self.neighbors),
        }


def unit_vector(values: Sequence[float], name: str = "vector") -> np.ndarray:
    vector = np.asarray(values, dtype=float)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain exactly three finite values.")
    magnitude = float(np.linalg.norm(vector))
    if magnitude <= 0.0:
        raise ValueError(f"{name} must not be the zero vector.")
    return vector / magnitude


def cell_matrix_from_lengths_angles(
        lengths: Sequence[float],
        angles_degrees: Sequence[float] = (90.0, 90.0, 90.0),
) -> np.ndarray:
    """Return a Cartesian cell matrix whose rows are lattice vectors."""

    a, b, c = np.asarray(lengths, dtype=float)
    alpha, beta, gamma = np.deg2rad(np.asarray(angles_degrees, dtype=float))
    if not np.isfinite([a, b, c, alpha, beta, gamma]).all() or min(a, b, c) <= 0.0:
        raise ValueError("Cell lengths and angles must be finite with positive lengths.")
    sin_gamma = float(np.sin(gamma))
    if abs(sin_gamma) < 1.0e-12:
        raise ValueError("The gamma cell angle produces a singular cell.")
    vector_a = np.asarray([a, 0.0, 0.0])
    vector_b = np.asarray([b * np.cos(gamma), b * sin_gamma, 0.0])
    vector_c_x = c * np.cos(beta)
    vector_c_y = c * (np.cos(alpha) - np.cos(beta) * np.cos(gamma)) / sin_gamma
    vector_c_z_sq = c * c - vector_c_x * vector_c_x - vector_c_y * vector_c_y
    if vector_c_z_sq < -1.0e-10:
        raise ValueError("Cell lengths and angles do not define a valid cell.")
    vector_c = np.asarray([vector_c_x, vector_c_y, np.sqrt(max(0.0, vector_c_z_sq))])
    cell = np.vstack((vector_a, vector_b, vector_c))
    if abs(float(np.linalg.det(cell))) < 1.0e-12:
        raise ValueError("Cell matrix is singular.")
    return cell


def _periodic_images(
        positions: np.ndarray,
        cell: np.ndarray | None,
        periodic: tuple[bool, bool, bool],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xyz = np.asarray(positions, dtype=float)
    source_indices = np.arange(xyz.shape[0], dtype=int)
    if cell is None or not any(periodic):
        return xyz, xyz, source_indices
    inverse_cell = np.linalg.inv(cell)
    fractional = xyz @ inverse_cell
    for axis, is_periodic in enumerate(periodic):
        if is_periodic:
            fractional[:, axis] -= np.floor(fractional[:, axis])
    wrapped = fractional @ cell
    image_ranges = [(-1, 0, 1) if value else (0,) for value in periodic]
    shifts = np.asarray(list(itertools.product(*image_ranges)), dtype=float) @ cell
    replicas = (wrapped[np.newaxis, :, :] + shifts[:, np.newaxis, :]).reshape(-1, 3)
    return wrapped, replicas, np.tile(source_indices, len(shifts))


def nearest_four_neighbors(
        center_positions: np.ndarray,
        neighbor_positions: np.ndarray,
        *,
        cell: np.ndarray | None,
        periodic: tuple[bool, bool, bool],
        cutoff: float | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Find four unique nearest atoms and their minimum-image bond vectors."""

    n_centers = center_positions.shape[0]
    selected_indices = np.full((n_centers, 4), -1, dtype=int)
    selected_distances = np.full((n_centers, 4), np.nan, dtype=float)
    selected_vectors = np.full((n_centers, 4, 3), np.nan, dtype=float)
    if n_centers == 0 or neighbor_positions.shape[0] == 0:
        return selected_indices, selected_distances, selected_vectors

    _, replicas, replica_sources = _periodic_images(neighbor_positions, cell, periodic)
    if cell is not None and any(periodic):
        fractional = center_positions @ np.linalg.inv(cell)
        for axis, is_periodic in enumerate(periodic):
            if is_periodic:
                fractional[:, axis] -= np.floor(fractional[:, axis])
        query_positions = fractional @ cell
        image_count = int(np.prod([3 if value else 1 for value in periodic]))
    else:
        query_positions = center_positions
        image_count = 1

    query_k = min(replicas.shape[0], max(4, 4 * image_count))
    tree = cKDTree(replicas)
    distances, replica_indices = tree.query(
        query_positions,
        k=query_k,
        distance_upper_bound=np.inf if cutoff is None else float(cutoff),
        workers=-1,
    )
    distances = np.atleast_2d(np.asarray(distances, dtype=float))
    replica_indices = np.atleast_2d(np.asarray(replica_indices, dtype=int))
    for center_row in range(n_centers):
        used_sources: set[int] = set()
        output_column = 0
        for distance, replica_index in zip(distances[center_row], replica_indices[center_row]):
            if not np.isfinite(distance) or int(replica_index) >= replicas.shape[0]:
                continue
            source_index = int(replica_sources[int(replica_index)])
            if source_index in used_sources:
                continue
            used_sources.add(source_index)
            selected_indices[center_row, output_column] = source_index
            selected_distances[center_row, output_column] = float(distance)
            selected_vectors[center_row, output_column] = (
                    replicas[int(replica_index)] - query_positions[center_row]
            )
            output_column += 1
            if output_column == 4:
                break
    return selected_indices, selected_distances, selected_vectors


def _has_neighbor_within_cutoff(
        centers: np.ndarray,
        neighbors: np.ndarray,
        *,
        cell: np.ndarray | None,
        periodic: tuple[bool, bool, bool],
        cutoff: float,
) -> np.ndarray:
    if centers.shape[0] == 0 or neighbors.shape[0] == 0:
        return np.zeros(centers.shape[0], dtype=bool)
    _, replicas, _ = _periodic_images(neighbors, cell, periodic)
    if cell is not None and any(periodic):
        fractional = centers @ np.linalg.inv(cell)
        for axis, is_periodic in enumerate(periodic):
            if is_periodic:
                fractional[:, axis] -= np.floor(fractional[:, axis])
        query = fractional @ cell
    else:
        query = centers
    distances, _ = cKDTree(replicas).query(query, k=1, workers=-1)
    distances = np.asarray(distances, dtype=float)
    return np.isfinite(distances) & (distances <= cutoff)


def _validate_request(request: WurtziteNeighborRequest) -> None:
    if int(request.every) < 1:
        raise ValueError("every must be at least 1.")
    if request.charge_source not in {"auto", "reaxff", "formal"}:
        raise ValueError("charge_source must be 'auto', 'reaxff', or 'formal'.")
    if request.neighbor_cutoff is not None and (
            not np.isfinite(request.neighbor_cutoff) or request.neighbor_cutoff <= 0.0
    ):
        raise ValueError("neighbor_cutoff must be positive and finite, or None.")
    if not np.isfinite(request.proton_cutoff) or request.proton_cutoff <= 0.0:
        raise ValueError("proton_cutoff must be positive and finite.")
    if not request.centers or not request.neighbors:
        raise ValueError("At least one center and neighbor label are required.")
    if {str(v).casefold() for v in request.centers} & {
        str(v).casefold() for v in request.neighbors
    }:
        raise ValueError("Center and neighbor label sets must not overlap.")
    if request.charge_source == "formal":
        configured = _formal_charge_map(request)
        required = {str(value).casefold(): str(value) for value in (*request.centers, *request.neighbors)}
        missing = [label for key, label in required.items() if key not in configured]
        if missing:
            joined = ", ".join(sorted(missing, key=str.casefold))
            raise ValueError(
                "--charge-source formal requires explicit --formal-charge values for "
                f"every center and neighbor species; missing: {joined}."
            )
    unit_vector(request.c_axis, "c_axis")
    if len(request.periodic) != 3:
        raise ValueError("periodic must contain exactly three booleans.")


def _formal_charge_map(request: WurtziteNeighborRequest) -> dict[str, float]:
    result = {str(label).casefold(): float(value) for label, value in request.formal_charges.items()}
    for value in result.values():
        if not np.isfinite(value):
            raise ValueError("Formal charges must be finite.")
    return result


def _frame_cell(trajectory: TrajectoryData, request: WurtziteNeighborRequest, frame: int):
    if request.cell_lengths is not None:
        return cell_matrix_from_lengths_angles(request.cell_lengths, request.cell_angles)
    simulation = trajectory.simulation
    if simulation is None or simulation.cell_lengths is None:
        return None
    lengths = np.asarray(simulation.cell_lengths, dtype=float)
    angles = (
        np.full((lengths.shape[0], 3), 90.0)
        if simulation.cell_angles is None
        else np.asarray(simulation.cell_angles, dtype=float)
    )
    return cell_matrix_from_lengths_angles(lengths[frame], angles[frame])


def _frame_labels(trajectory: TrajectoryData, frame: int) -> np.ndarray:
    if trajectory.atom_labels is not None:
        return np.asarray(trajectory.atom_labels[frame], dtype=object)
    return np.asarray(trajectory.elements, dtype=object)


def _frame_iteration(trajectory: TrajectoryData, frame: int) -> int:
    values = trajectory.iterations
    if values is None and trajectory.simulation is not None:
        values = trajectory.simulation.iterations
    return int(np.asarray(values).reshape(-1)[frame]) if values is not None else int(frame)


def _source_frame(trajectory: TrajectoryData, frame: int) -> int:
    values = trajectory.source_frame_indices
    return int(np.asarray(values).reshape(-1)[frame]) if values is not None else int(frame)


def _select_frames(n_frames: int, request: WurtziteNeighborRequest) -> list[int]:
    selected = list(range(n_frames)) if request.frames is None else [int(v) for v in request.frames]
    invalid = [value for value in selected if value < 0 or value >= n_frames]
    if invalid:
        raise ValueError(f"Frame(s) not found in trajectory: {invalid}.")
    return selected[:: int(request.every)]


def _trajectory_and_charges(
        data: TrajectoryData | ElectrostaticsData,
) -> tuple[TrajectoryData, np.ndarray | None]:
    if isinstance(data, ElectrostaticsData):
        return data.trajectory, np.asarray(data.charges.charges, dtype=float)
    return data, None


def _charges_for_request(data, request: WurtziteNeighborRequest):
    trajectory, charges = _trajectory_and_charges(data)
    return trajectory, None if request.charge_source == "formal" else charges


def extract_wurtzite_neighbors(
        trajectory: TrajectoryData,
        request: WurtziteNeighborRequest,
        *,
        charges: np.ndarray | None = None,
        frame_indices: Iterable[int] | None = None,
        preserve_source_frame_indices: bool = False,
) -> WurtziteNeighborResult:
    """Return center and nearest-neighbor tables without calculating polarity."""

    _validate_request(request)
    positions = np.asarray(trajectory.positions, dtype=float)
    n_frames, n_atoms = positions.shape[:2]
    if charges is not None and np.asarray(charges).shape != (n_frames, n_atoms):
        raise ValueError("charges must have shape (n_frames, n_atoms), matching trajectory positions.")
    charge_array = None if charges is None else np.asarray(charges, dtype=float)
    ids = np.asarray(trajectory.atom_ids, dtype=int)
    if frame_indices is None:
        selected_frames = _select_frames(n_frames, request)
    else:
        selected_frames = [int(value) for value in frame_indices]

    c_hat = unit_vector(request.c_axis, "c_axis")
    periodic = tuple(bool(value) for value in request.periodic)
    formal = _formal_charge_map(request)
    center_names = {str(value).casefold() for value in request.centers}
    neighbor_names = {str(value).casefold() for value in request.neighbors}
    proton_names = {str(value).casefold() for value in request.protons}
    center_rows: list[dict[str, object]] = []
    neighbor_rows: list[dict[str, object]] = []
    output_frames: list[int] = []
    output_iterations: list[int] = []

    for frame in selected_frames:
        xyz = positions[frame]
        labels = _frame_labels(trajectory, frame)
        folded = np.asarray([str(value).casefold() for value in labels], dtype=object)
        finite = np.isfinite(xyz).all(axis=1)
        center_indices = np.flatnonzero(finite & np.isin(folded, list(center_names)))
        neighbor_indices = np.flatnonzero(finite & np.isin(folded, list(neighbor_names)))
        proton_indices = np.flatnonzero(finite & np.isin(folded, list(proton_names)))
        output_frame = _source_frame(trajectory, frame) if preserve_source_frame_indices else frame
        if center_indices.size == 0:
            raise ValueError(f"Frame {output_frame} contains no selected centers.")
        if neighbor_indices.size < 4:
            raise ValueError(
                f"Frame {output_frame} contains only {neighbor_indices.size} "
                "selected neighbor atoms; at least four are required."
            )
        cell = _frame_cell(trajectory, request, frame)
        frame_periodic = periodic if cell is not None else (False, False, False)
        if any(periodic) and cell is None:
            warnings.warn(
                "Periodic directions were requested but no cell was supplied; using non-periodic distances.",
                RuntimeWarning,
                stacklevel=2,
            )
        local_indices, distances, vectors = nearest_four_neighbors(
            xyz[center_indices],
            xyz[neighbor_indices],
            cell=cell,
            periodic=frame_periodic,
            cutoff=request.neighbor_cutoff,
        )
        near_proton = _has_neighbor_within_cutoff(
            xyz[center_indices],
            xyz[proton_indices],
            cell=cell,
            periodic=frame_periodic,
            cutoff=float(request.proton_cutoff),
        )
        iteration = _frame_iteration(trajectory, frame)
        output_frames.append(output_frame)
        output_iterations.append(iteration)
        for center_row, center_index in enumerate(center_indices):
            valid = local_indices[center_row] >= 0
            count = int(np.count_nonzero(valid))
            center_label = str(labels[center_index])
            apical_rank = (
                int(np.argmax(np.abs(vectors[center_row, :, :] @ c_hat))) + 1
                if count == 4
                else 0
            )
            center_charge = (
                float(charge_array[frame, center_index])
                if charge_array is not None
                else float(formal.get(center_label.casefold(), np.nan))
            )
            center_rows.append(
                {
                    "frame_index": output_frame,
                    "iter": iteration,
                    "site_atom_index": int(center_index),
                    "site_atom_id": int(ids[center_index]),
                    "site_element": center_label,
                    "site_x (angstrom)": float(xyz[center_index, 0]),
                    "site_y (angstrom)": float(xyz[center_index, 1]),
                    "site_z (angstrom)": float(xyz[center_index, 2]),
                    "site_charge (e)": center_charge,
                    "neighbor_count_within_cutoff": count,
                    "has_four_neighbors": count == 4,
                    "has_proton_within_cutoff": bool(near_proton[center_row]),
                    "proton_cutoff (angstrom)": float(request.proton_cutoff),
                    "charge_source": "per-atom" if charge_array is not None else "formal",
                    "neighbor_cutoff (angstrom)": (
                        np.nan if request.neighbor_cutoff is None else float(request.neighbor_cutoff)
                    ),
                    "c_axis_x": float(c_hat[0]),
                    "c_axis_y": float(c_hat[1]),
                    "c_axis_z": float(c_hat[2]),
                    "periodic_a": periodic[0],
                    "periodic_b": periodic[1],
                    "periodic_c": periodic[2],
                }
            )
            for rank in range(4):
                local_index = int(local_indices[center_row, rank])
                if local_index < 0:
                    continue
                atom_index = int(neighbor_indices[local_index])
                label = str(labels[atom_index])
                neighbor_charge = (
                    float(charge_array[frame, atom_index])
                    if charge_array is not None
                    else float(formal.get(label.casefold(), np.nan))
                )
                vector = vectors[center_row, rank]
                image_xyz = xyz[center_index] + vector
                neighbor_rows.append(
                    {
                        "frame_index": output_frame,
                        "iter": iteration,
                        "site_atom_id": int(ids[center_index]),
                        "site_element": center_label,
                        "neighbor_rank": rank + 1,
                        "neighbor_atom_index": atom_index,
                        "neighbor_atom_id": int(ids[atom_index]),
                        "neighbor_element": label,
                        "neighbor_charge (e)": neighbor_charge,
                        "neighbor_x (angstrom)": float(xyz[atom_index, 0]),
                        "neighbor_y (angstrom)": float(xyz[atom_index, 1]),
                        "neighbor_z (angstrom)": float(xyz[atom_index, 2]),
                        "neighbor_image_x (angstrom)": float(image_xyz[0]),
                        "neighbor_image_y (angstrom)": float(image_xyz[1]),
                        "neighbor_image_z (angstrom)": float(image_xyz[2]),
                        "distance (angstrom)": float(distances[center_row, rank]),
                        "bond_x (angstrom)": float(vector[0]),
                        "bond_y (angstrom)": float(vector[1]),
                        "bond_z (angstrom)": float(vector[2]),
                        "bond_c (angstrom)": float(vector @ c_hat),
                        "neighbor_role": (
                            "apical"
                            if rank + 1 == apical_rank
                            else "basal"
                            if count == 4
                            else "unassigned"
                        ),
                    }
                )

    centers_table = pd.DataFrame(center_rows, columns=CENTER_COLUMNS)
    neighbors_table = pd.DataFrame(neighbor_rows, columns=NEIGHBOR_COLUMNS)
    return WurtziteNeighborResult(
        table=neighbors_table,
        request=request,
        centers=centers_table,
        neighbors=neighbors_table,
        frame_indices=np.asarray(output_frames, dtype=int),
        iterations=np.asarray(output_iterations, dtype=int),
    )


def _candidate_charge_file(args: dict) -> Path | None:
    configured = Path(str(args.get("fort7") or "fort.7"))
    candidates = [configured]
    for key in ("run_dir", "input", "xmolout"):
        raw = args.get(key)
        if not raw:
            continue
        path = Path(str(raw))
        base = path if path.is_dir() else path.parent
        candidates.append(configured if configured.is_absolute() else base / configured)
    return next((path for path in candidates if path.is_file()), None)


def required_wurtzite_data_type(request: WurtziteNeighborRequest, args: dict | None = None):
    """Choose coordinates-only formal analysis or coordinate+charge analysis."""

    if request.charge_source == "formal":
        return TrajectoryData
    if request.charge_source == "reaxff":
        return ElectrostaticsData
    return ElectrostaticsData if _candidate_charge_file(args or {}) is not None else TrajectoryData


@register_task("get-wurtzite-neighbors", label="Four-fold Wurtzite Neighbors")
class WurtziteNeighborTask(AnalysisTask):
    """Extract distance-defined neighbors and coordinates without polarity math."""

    required_data = TrajectoryData
    supports_selective_streaming = True
    VERSION = "1"

    def required_data_for(self, request: WurtziteNeighborRequest, args: dict | None = None):
        return required_wurtzite_data_type(request, args)

    @staticmethod
    def required_data_fields_for(request: WurtziteNeighborRequest, _args: dict) -> tuple[str, ...]:
        return ("trajectory", "charges") if request.charge_source != "formal" else ("trajectory",)

    @staticmethod
    def recommended_presentations(
            _result: WurtziteNeighborResult, _payload: dict[str, Any]
    ) -> list[PresentationSpec]:
        return [PresentationSpec(renderer="table", label="Center neighbors", view_type="table")]

    def run(self, data, request: WurtziteNeighborRequest, reporter=None) -> WurtziteNeighborResult:
        _ = reporter
        trajectory, charges = _charges_for_request(data, request)
        return extract_wurtzite_neighbors(trajectory, request, charges=charges)

    def run_stream(self, frames, request: WurtziteNeighborRequest, reporter=None):
        _validate_request(request)
        requested = None if request.frames is None else [int(v) for v in request.frames][:: int(request.every)]
        requested_set = set(requested or ())
        center_tables: list[pd.DataFrame] = []
        neighbor_tables: list[pd.DataFrame] = []
        output_frames: list[int] = []
        output_iterations: list[int] = []
        seen: set[int] = set()
        processed = 0
        for stream_index, data in enumerate(frames):
            processed += 1
            trajectory, charges = _charges_for_request(data, request)
            source_frame = _source_frame(trajectory, 0)
            seen.add(source_frame)
            keep = source_frame in requested_set if requested is not None else source_frame % int(request.every) == 0
            if keep:
                frame_request = WurtziteNeighborRequest(**{
                    **vars(request), "frames": [0], "every": 1,
                })
                result = extract_wurtzite_neighbors(
                    trajectory,
                    frame_request,
                    charges=charges,
                    frame_indices=[0],
                    preserve_source_frame_indices=True,
                )
                center_tables.append(result.centers)
                neighbor_tables.append(result.neighbors)
                output_frames.extend(result.frame_indices.tolist())
                output_iterations.extend(result.iterations.tolist())
            if callable(reporter):
                reporter("stream", processed, 0, "Finding four-fold wurtzite neighbors")
        if requested is not None:
            missing = [value for value in requested if value not in seen]
            if missing:
                raise ValueError(f"Requested frame(s) not found in trajectory: {missing}.")
        centers = pd.concat(center_tables, ignore_index=True) if center_tables else pd.DataFrame(columns=CENTER_COLUMNS)
        neighbors = pd.concat(neighbor_tables, ignore_index=True) if neighbor_tables else pd.DataFrame(
            columns=NEIGHBOR_COLUMNS)
        return WurtziteNeighborResult(
            table=neighbors,
            request=request,
            centers=centers,
            neighbors=neighbors,
            frame_indices=np.asarray(output_frames, dtype=int),
            iterations=np.asarray(output_iterations, dtype=int),
        )


__all__ = [
    "CENTER_COLUMNS",
    "NEIGHBOR_COLUMNS",
    "WurtziteNeighborRequest",
    "WurtziteNeighborResult",
    "WurtziteNeighborTask",
    "cell_matrix_from_lengths_angles",
    "centers_csv_table",
    "extract_wurtzite_neighbors",
    "nearest_four_neighbors",
    "neighbors_csv_table",
    "required_wurtzite_data_type",
    "time_after_iter",
    "unit_vector",
]
