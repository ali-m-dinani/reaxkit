"""Polarization from displacements relative to nonpolar hexagonal AlN.

The implementation follows the longitudinal displacement expression used by
Hayden et al., Phys. Rev. Materials 5, 044412 (2021):

``mu_k = -q_k * Delta u_k`` and ``P = e / Omega * sum_k(mu_k)``.

The Cartesian x, y, and z components are reported together with the projection
onto the user-selected c-axis.

The selected ``q_k`` values are ReaxFF partial charges or user-supplied formal
charges. They provide a classical approximation to the Born-effective-charge
expression and do not include an electronic Berry-phase contribution.

The reference is read from a CIF, optionally changed to ReaxKit's orthogonal
hexagonal supercell, and replicated by explicit user-supplied counts.  Small
box/reference length differences are treated as homogeneous strain, while a
large gap (such as slab vacuum) is retained.  A best-fit periodic translation
removes changes of cell origin before minimum-image displacements are
evaluated.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import numpy as np
import pandas as pd
from ase import Atoms
from ase.geometry import cell_to_cellpar
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import min_weight_full_bipartite_matching
from scipy.spatial import cKDTree

from reaxkit.analysis.base import AnalysisTask
from reaxkit.analysis.ferroelectrics.four_folded_wurtzite.neighbors import (
    _trajectory_and_charges,
    cell_matrix_from_lengths_angles,
    required_wurtzite_data_type,
)
from reaxkit.analysis.ferroelectrics.three_folded_wurtzite.polarization import (
    VolumeMethod,
    _bbox_volume,
    _hull_volume,
)
from reaxkit.core.platform.constants import const
from reaxkit.core.registry.analysis_task_registry import register_task
from reaxkit.domain.base_request import BaseRequest
from reaxkit.domain.base_result import BaseResult
from reaxkit.domain.data_models import ElectrostaticsData, TrajectoryData
from reaxkit.engine.common.generators.structure_transformers import (
    orthogonalize_hexagonal_cell,
)
from reaxkit.engine.common.io.geo_io import read_structure, write_structure
from reaxkit.presentation.specs import PresentationSpec

OrthogonalizeMode = Literal["auto", "always", "never"]
PeriodicAxes = tuple[bool, bool, bool]
ORDERED_MATCH_RMS_TOLERANCE_ANGSTROM = 2.0
ELECTRON_CHARGE_SIGN = -1.0

ChargeSource = Literal["auto", "reaxff", "formal"]
DEFAULT_FORMAL_CHARGES: dict[str, float] = {
    "Al": 3.0,
    "N": -3.0,
}
REFERENCE_STRUCTURE_PATH = Path(__file__).with_name("AlN_hbn.cif")


@dataclass
class HBNReferencePolarizationRequest(BaseRequest):
    """Configure reference construction, matching, and atomic charges."""

    reference_path: str | Path
    replication: Sequence[int]
    charge_source: ChargeSource = "auto"
    formal_charges: Mapping[str, float] = field(
        default_factory=lambda: dict(DEFAULT_FORMAL_CHARGES)
    )
    reference_species: Mapping[str, str] = field(
        default_factory=lambda: {"B": "Al"}
    )
    c_axis: Sequence[float] = (0.0, 0.0, 1.0)
    periodic: Sequence[bool] = (True, True, True)
    cell_lengths: Sequence[float] | None = None
    cell_angles: Sequence[float] = (90.0, 90.0, 90.0)
    frames: Sequence[int] | None = None
    every: int = 1
    reference_frame: int = 0
    orthogonalize: OrthogonalizeMode = "auto"
    angle_tolerance_degrees: float = 1.0
    max_reference_strain: float = 0.15
    max_alignment_candidates: int = 8
    volume_method: VolumeMethod = field(
        default="hull",
        metadata={"label": "Volume method", "choices": ["hull", "bbox", "cell"]},
    )


@dataclass
class PreparedHBNReference:
    """Replicated reference and its alignment to trajectory atom identities."""

    atoms: Atoms
    fractional_positions: np.ndarray
    simulation_to_reference: np.ndarray
    reference_to_simulation: np.ndarray
    translation_fractional: np.ndarray
    aligned_positions: np.ndarray
    repeats: tuple[int, int, int]
    orthogonalized: bool
    reference_angles: np.ndarray
    trajectory_angles: np.ndarray
    strain_ratios: np.ndarray
    applied_strain_ratios: np.ndarray
    local_cell_ids: np.ndarray
    local_cell_centers: np.ndarray
    local_layer_ids: np.ndarray
    local_layer_centers: np.ndarray


@dataclass
class HBNReferencePolarizationResult(BaseResult):
    """Per-atom displacements, frame polarization, and aligned reference."""

    table: pd.DataFrame
    request: HBNReferencePolarizationRequest
    displacements: pd.DataFrame
    mapping: pd.DataFrame
    reference: PreparedHBNReference
    frame_indices: np.ndarray
    iterations: np.ndarray

    @property
    def csv_tables(self) -> dict[str, pd.DataFrame]:
        return {
            "hbn_reference_polarization": self.table,
            "hbn_reference_displacements": self.displacements,
            "hbn_reference_mapping": self.mapping,
        }


def _unit_vector(values: Sequence[float], name: str) -> np.ndarray:
    vector = np.asarray(values, dtype=float)
    if vector.shape != (3,) or not np.isfinite(vector).all():
        raise ValueError(f"{name} must contain three finite values.")
    norm = float(np.linalg.norm(vector))
    if norm <= 0.0:
        raise ValueError(f"{name} must have nonzero length.")
    return vector / norm


def _frame_cell(
        trajectory: TrajectoryData,
        request: HBNReferencePolarizationRequest,
        frame_index: int,
) -> np.ndarray:
    if request.cell_lengths is not None:
        return cell_matrix_from_lengths_angles(request.cell_lengths, request.cell_angles)
    simulation = trajectory.simulation
    if simulation is None or simulation.cell_lengths is None:
        raise ValueError(
            "The trajectory has no simulation cell. Supply --cell-lengths and "
            "--cell-angles or use a trajectory containing cell metadata."
        )
    lengths = np.asarray(simulation.cell_lengths, dtype=float)
    if frame_index < 0 or frame_index >= len(lengths):
        raise ValueError(f"Cell metadata has no frame {frame_index}.")
    if simulation.cell_angles is None:
        angles = np.full(3, 90.0)
    else:
        angles = np.asarray(simulation.cell_angles, dtype=float)[frame_index]
    return cell_matrix_from_lengths_angles(lengths[frame_index], angles)


def _frame_labels(trajectory: TrajectoryData, frame_index: int) -> np.ndarray:
    if trajectory.atom_labels is not None:
        return np.asarray(trajectory.atom_labels[frame_index], dtype=object)
    return np.asarray(trajectory.elements, dtype=object)


def _iteration(trajectory: TrajectoryData, frame_index: int) -> int:
    values = trajectory.iterations
    if values is None and trajectory.simulation is not None:
        values = trajectory.simulation.iterations
    return int(np.asarray(values)[frame_index]) if values is not None else int(frame_index)


def _selected_frames(
        trajectory: TrajectoryData, request: HBNReferencePolarizationRequest
) -> list[int]:
    n_frames = int(np.asarray(trajectory.positions).shape[0])
    if int(request.every) < 1:
        raise ValueError("every must be at least 1.")
    selected = (
        list(range(n_frames))
        if request.frames is None
        else [int(value) for value in request.frames]
    )
    invalid = [value for value in selected if value < 0 or value >= n_frames]
    if invalid:
        raise ValueError(f"Frame(s) not found in trajectory: {invalid}.")
    return selected[:: int(request.every)]


def _angles(cell: np.ndarray) -> np.ndarray:
    return np.asarray(cell_to_cellpar(np.asarray(cell, dtype=float))[3:], dtype=float)


def _angle_error(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(left) - np.asarray(right)) ** 2)))


def _orient_reference(
        atoms: Atoms,
        target_angles: np.ndarray,
        mode: OrthogonalizeMode,
        tolerance: float,
) -> tuple[Atoms, bool]:
    if mode not in {"auto", "always", "never"}:
        raise ValueError("orthogonalize must be one of: auto, always, never.")
    original = atoms.copy()
    original.set_pbc(True)
    if mode == "never":
        return original, False
    converted = orthogonalize_hexagonal_cell(original)
    converted.wrap()
    if mode == "always":
        return converted, True
    original_error = _angle_error(_angles(original.cell.array), target_angles)
    converted_error = _angle_error(_angles(converted.cell.array), target_angles)
    if converted_error + float(tolerance) < original_error:
        return converted, True
    return original, False


def _primitive_ids_in_oriented_reference(source: Atoms, oriented: Atoms) -> np.ndarray:
    """Identify copies of the input crystallographic cell after orientation."""

    source_count = len(source)
    if source_count == 0 or len(oriented) % source_count:
        raise ValueError("The oriented reference is not an integer number of CIF cells.")
    # ASE make_supercell uses cell-major ordering by default: all basis atoms
    # from one translated source cell are contiguous. Coordinate-based recovery
    # is ambiguous after wrap() moves boundary atoms into adjacent images.
    return np.arange(len(oriented), dtype=int) // source_count


def _source_layer_ids(source: Atoms, *, tolerance: float = 1.0e-6) -> np.ndarray:
    """Assign one id to each distinct reference plane normal to the c axis."""

    fractional_z = np.mod(
        np.asarray(source.get_scaled_positions(wrap=False), dtype=float)[:, 2], 1.0
    )
    order = np.argsort(fractional_z, kind="stable")
    layer_ids = np.empty(len(source), dtype=int)
    layer_id = 0
    previous: float | None = None
    for atom_index in order:
        value = float(fractional_z[atom_index])
        if previous is not None and value - previous > float(tolerance):
            layer_id += 1
        layer_ids[atom_index] = layer_id
        previous = value
    return layer_ids


def _replicated_local_cell_ids(oriented: Atoms, replicated: Atoms) -> np.ndarray:
    """Combine oriented primitive-copy and repeat-image identities."""

    primitive_id = np.asarray(replicated.arrays["_rk_primitive_id"], dtype=int)
    oriented_count = len(oriented)
    if oriented_count == 0 or len(replicated) % oriented_count:
        raise ValueError("The replicated reference is not an integer oriented-cell repeat.")
    repeat_image = np.arange(len(replicated), dtype=int) // oriented_count
    primitive_count = int(np.max(np.asarray(oriented.arrays["_rk_primitive_id"]))) + 1
    return repeat_image * primitive_count + primitive_id


def _replicated_local_layer_ids(oriented: Atoms, replicated: Atoms) -> np.ndarray:
    """Combine source-layer, oriented-copy, and repeat-image identities."""

    primitive_id = np.asarray(replicated.arrays["_rk_primitive_id"], dtype=int)
    source_layer_id = np.asarray(
        replicated.arrays["_rk_source_layer_id"], dtype=int
    )
    oriented_count = len(oriented)
    if oriented_count == 0 or len(replicated) % oriented_count:
        raise ValueError("The replicated reference is not an integer oriented-cell repeat.")
    repeat_image = np.arange(len(replicated), dtype=int) // oriented_count
    primitive_count = int(np.max(np.asarray(oriented.arrays["_rk_primitive_id"]))) + 1
    source_layer_count = (
        int(np.max(np.asarray(oriented.arrays["_rk_source_layer_id"]))) + 1
    )
    oriented_layer_id = primitive_id * source_layer_count + source_layer_id
    return repeat_image * (primitive_count * source_layer_count) + oriented_layer_id


def _replication(values: Sequence[int]) -> tuple[int, int, int]:
    repeats = tuple(int(value) for value in values)
    if len(repeats) != 3 or any(value < 1 for value in repeats):
        raise ValueError("replication must contain three positive integers.")
    return repeats[0], repeats[1], repeats[2]


def _strain_ratios(
        reference: Atoms,
        target_cell: np.ndarray,
        repeats: tuple[int, int, int],
) -> np.ndarray:
    reference_lengths = np.linalg.norm(reference.cell.array, axis=1)
    target_lengths = np.linalg.norm(target_cell, axis=1)
    return target_lengths / (
            reference_lengths * np.asarray(repeats, dtype=float)
    )


def _periodic_axes(values: Sequence[bool]) -> PeriodicAxes:
    items = tuple(bool(value) for value in values)
    if len(items) != 3:
        raise ValueError("periodic must contain three booleans.")
    return items[0], items[1], items[2]


def _minimum_image(fractional: np.ndarray, periodic: PeriodicAxes) -> np.ndarray:
    values = np.asarray(fractional, dtype=float).copy()
    for axis, enabled in enumerate(periodic):
        if enabled:
            values[..., axis] -= np.round(values[..., axis])
    return values


def _wrap_fractional(
        fractional: np.ndarray, periodic: PeriodicAxes
) -> np.ndarray:
    values = np.asarray(fractional, dtype=float).copy()
    for axis, enabled in enumerate(periodic):
        if enabled:
            values[..., axis] -= np.floor(values[..., axis])
    return values


def _species_key_map(mapping: Mapping[str, str]) -> dict[str, str]:
    return {str(key).casefold(): str(value).casefold() for key, value in mapping.items()}


def _reference_species_for(
        labels: np.ndarray, mapping: Mapping[str, str]
) -> np.ndarray:
    aliases = _species_key_map(mapping)
    return np.asarray(
        [aliases.get(str(label).casefold(), str(label).casefold()) for label in labels],
        dtype=object,
    )


def _assignment_for_translation(
        simulation_fractional: np.ndarray,
        simulation_species: np.ndarray,
        reference_fractional: np.ndarray,
        reference_species: np.ndarray,
        translation: np.ndarray,
        cell: np.ndarray,
        periodic: PeriodicAxes,
) -> tuple[np.ndarray, float]:
    """Find a scalable one-to-one assignment for a trial origin shift.

    The common path is a periodic KD-tree lookup and needs O(N) memory.  If
    severe distortion causes two atoms to select the same reference site, a
    sparse k-nearest-neighbor bipartite match resolves only those alternatives.
    """

    assignment = np.full(len(simulation_fractional), -1, dtype=int)
    score = 0.0
    simulation_wrapped = _wrap_fractional(simulation_fractional, periodic)
    shifted_reference = _wrap_fractional(
        reference_fractional + translation, periodic
    )
    image_ranges = [(-1, 0, 1) if enabled else (0,) for enabled in periodic]
    shifts = np.asarray(list(itertools.product(*image_ranges)), dtype=float)
    for species in sorted(set(reference_species.tolist())):
        sim_indices = np.flatnonzero(simulation_species == species)
        ref_indices = np.flatnonzero(reference_species == species)
        if len(sim_indices) != len(ref_indices):
            raise ValueError(
                f"Species count mismatch for reference species {species!r}: "
                f"trajectory has {len(sim_indices)}, reference has {len(ref_indices)}. "
                "Use --reference-species for substitutions such as B=Al."
            )
        local_reference = shifted_reference[ref_indices]
        image_fractional = (
                local_reference[None, :, :] + shifts[:, None, :]
        ).reshape(-1, 3)
        image_to_local = np.tile(np.arange(len(ref_indices), dtype=int), len(shifts))
        tree = cKDTree(image_fractional @ cell)
        query_positions = simulation_wrapped[sim_indices] @ cell
        distances, image_indices = tree.query(query_positions, k=1)
        local_assignment = image_to_local[np.asarray(image_indices, dtype=int)]

        if len(np.unique(local_assignment)) != len(local_assignment):
            local_assignment, distances = _sparse_bijective_assignment(
                tree=tree,
                query_positions=query_positions,
                image_to_local=image_to_local,
                reference_count=len(ref_indices),
            )
        assignment[sim_indices] = ref_indices[local_assignment]
        score += float(np.sum(np.asarray(distances, dtype=float) ** 2))
    if np.any(assignment < 0):
        raise RuntimeError("Reference assignment did not cover every trajectory atom.")
    return assignment, score


def _sparse_bijective_assignment(
        *,
        tree: cKDTree,
        query_positions: np.ndarray,
        image_to_local: np.ndarray,
        reference_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Resolve KD-tree collisions without allocating a dense cost matrix."""

    maximum_neighbors = min(64, reference_count)
    neighbor_count = min(4, maximum_neighbors)
    while True:
        distances, image_indices = tree.query(
            query_positions, k=neighbor_count
        )
        distance_array = np.asarray(distances, dtype=float)
        image_array = np.asarray(image_indices, dtype=int)
        if distance_array.ndim == 1:
            distance_array = distance_array[:, None]
            image_array = image_array[:, None]

        edge_weights: dict[tuple[int, int], float] = {}
        for row in range(len(query_positions)):
            for rank in range(distance_array.shape[1]):
                column = int(image_to_local[image_array[row, rank]])
                weight = float(distance_array[row, rank] ** 2) + 1.0e-15
                key = (row, column)
                if weight < edge_weights.get(key, np.inf):
                    edge_weights[key] = weight
        rows = np.fromiter((key[0] for key in edge_weights), dtype=int)
        columns = np.fromiter((key[1] for key in edge_weights), dtype=int)
        weights = np.fromiter(edge_weights.values(), dtype=float)
        graph = coo_matrix(
            (weights, (rows, columns)),
            shape=(len(query_positions), reference_count),
        ).tocsr()
        try:
            # noinspection PyTypeChecker
            matched_rows, matched_columns = min_weight_full_bipartite_matching(
                graph
            )
            if len(matched_rows) == len(query_positions):
                assignment = np.empty(len(query_positions), dtype=int)
                assignment[matched_rows] = matched_columns
                matched_distances = np.sqrt(
                    np.asarray(graph[matched_rows, matched_columns]).reshape(-1)
                )
                return assignment, matched_distances
        except ValueError:
            pass
        if neighbor_count >= maximum_neighbors:
            raise ValueError(
                "Could not build a one-to-one atom/reference assignment from "
                f"the {maximum_neighbors} nearest sites. Check replication, "
                "orthogonalization, and reference-species settings."
            )
        neighbor_count = min(neighbor_count * 2, maximum_neighbors)


def _translation_from_assignment(
        simulation_fractional: np.ndarray,
        reference_fractional: np.ndarray,
        assignment: np.ndarray,
        cell: np.ndarray,
        periodic: PeriodicAxes,
) -> np.ndarray:
    difference = _minimum_image(
        simulation_fractional - reference_fractional[assignment], periodic
    )
    mean_cartesian = np.mean(difference @ cell, axis=0)
    return mean_cartesian @ np.linalg.inv(cell)


def _match_reference(
        simulation_positions: np.ndarray,
        simulation_labels: np.ndarray,
        reference: Atoms,
        cell: np.ndarray,
        request: HBNReferencePolarizationRequest,
) -> tuple[np.ndarray, np.ndarray, float]:
    periodic = _periodic_axes(request.periodic)
    simulation_fractional = np.asarray(simulation_positions, dtype=float) @ np.linalg.inv(cell)
    reference_fractional = np.asarray(reference.get_scaled_positions(wrap=False), dtype=float)
    simulation_species = _reference_species_for(simulation_labels, request.reference_species)
    reference_species = np.asarray(
        [str(value).casefold() for value in reference.get_chemical_symbols()], dtype=object
    )
    # ReaxFF trajectories retain GEO atom order. Explicit replication normally
    # recreates that order exactly, making atom identity an O(N) check. Keep the
    # spatial matcher as a fallback for reordered trajectory formats.
    if np.array_equal(simulation_species, reference_species):
        ordered_assignment = np.arange(len(simulation_fractional), dtype=int)
        ordered_translation = _translation_from_assignment(
            simulation_fractional,
            reference_fractional,
            ordered_assignment,
            cell,
            periodic,
        )
        ordered_delta = _minimum_image(
            simulation_fractional
            - reference_fractional
            - ordered_translation,
            periodic,
        ) @ cell
        ordered_score = float(np.sum(ordered_delta * ordered_delta))
        ordered_rms = float(np.sqrt(ordered_score / len(ordered_delta)))
        if ordered_rms <= ORDERED_MATCH_RMS_TOLERANCE_ANGSTROM:
            return ordered_assignment, ordered_translation, ordered_score
    anchor_species = simulation_species[0]
    compatible = np.flatnonzero(reference_species == anchor_species)
    if compatible.size == 0:
        raise ValueError(
            f"Trajectory species {simulation_labels[0]!r} has no compatible reference site."
        )
    # The correct origin puts the first simulation atom close to one of the
    # compatible reference sites. Rank those sites in O(N) time instead of
    # trying every replica or constructing an O(N^2) cost matrix.
    maximum = max(1, int(request.max_alignment_candidates))
    anchor_differences = _minimum_image(
        simulation_fractional[0] - reference_fractional[compatible], periodic
    )
    anchor_distances = np.sum((anchor_differences @ cell) ** 2, axis=1)
    compatible = compatible[np.argsort(anchor_distances)[:maximum]]
    candidates = [
        simulation_fractional[0] - reference_fractional[index]
        for index in compatible
    ]

    best: tuple[np.ndarray, np.ndarray, float] | None = None
    last_error: ValueError | None = None
    for initial in candidates:
        try:
            translation = np.asarray(initial, dtype=float)
            for _ in range(2):
                assignment, _ = _assignment_for_translation(
                    simulation_fractional, simulation_species,
                    reference_fractional, reference_species,
                    translation, cell, periodic,
                )
                updated = _translation_from_assignment(
                    simulation_fractional, reference_fractional, assignment, cell, periodic
                )
                if np.linalg.norm(
                        _minimum_image(updated - translation, periodic) @ cell
                ) < 1.0e-10:
                    translation = updated
                    break
                translation = updated
            assignment, score = _assignment_for_translation(
                simulation_fractional, simulation_species,
                reference_fractional, reference_species,
                translation, cell, periodic,
            )
        except ValueError as error:
            last_error = error
            continue
        candidate = (assignment, translation, score)
        if best is None or score < best[2]:
            best = candidate
    if best is None:
        detail = f" {last_error}" if last_error is not None else ""
        raise ValueError(f"Could not align the reference structure.{detail}")
    return best


def _validate_request(
        trajectory: TrajectoryData, request: HBNReferencePolarizationRequest
) -> None:
    positions = np.asarray(trajectory.positions, dtype=float)
    if positions.shape[1] == 0:
        raise ValueError("The trajectory contains no atoms.")
    if request.reference_frame < 0 or request.reference_frame >= positions.shape[0]:
        raise ValueError(f"reference_frame {request.reference_frame} is outside the trajectory.")
    _periodic_axes(request.periodic)
    if float(request.angle_tolerance_degrees) < 0.0:
        raise ValueError("angle_tolerance_degrees cannot be negative.")
    if not 0.0 <= float(request.max_reference_strain) < 1.0:
        raise ValueError("max_reference_strain must be at least 0 and less than 1.")
    if int(request.max_alignment_candidates) < 1:
        raise ValueError("max_alignment_candidates must be at least 1.")
    if request.volume_method not in {"hull", "bbox", "cell"}:
        raise ValueError("volume_method must be one of: hull, bbox, cell.")
    _replication(request.replication)
    _unit_vector(request.c_axis, "c_axis")


def prepare_hbn_reference(
        trajectory: TrajectoryData,
        request: HBNReferencePolarizationRequest,
) -> PreparedHBNReference:
    """Read, orient, replicate, and align the h-BN-like AlN reference."""

    _validate_request(trajectory, request)
    reference_path = Path(request.reference_path).expanduser().resolve()
    if not reference_path.is_file():
        raise FileNotFoundError(f"h-BN reference structure not found: {reference_path}")
    frame = int(request.reference_frame)
    target_cell = _frame_cell(trajectory, request, frame)
    target_angles = _angles(target_cell)
    source = read_structure(reference_path)
    if len(source) == 0 or abs(float(np.linalg.det(source.cell.array))) < 1.0e-12:
        raise ValueError("The reference CIF must contain atoms and a nonsingular cell.")
    source.new_array("_rk_source_basis_index", np.arange(len(source), dtype=int))
    source.new_array("_rk_source_layer_id", _source_layer_ids(source))
    oriented, was_orthogonalized = _orient_reference(
        source, target_angles, request.orthogonalize,
        float(request.angle_tolerance_degrees),
    )
    primitive_ids = _primitive_ids_in_oriented_reference(source, oriented)
    oriented.new_array("_rk_primitive_id", primitive_ids)
    oriented.new_array("_rk_oriented_index", np.arange(len(oriented), dtype=int))
    atom_count = int(np.asarray(trajectory.positions).shape[1])
    repeats = _replication(request.replication)
    strain = _strain_ratios(oriented, target_cell, repeats)
    replicated = oriented.repeat(repeats)
    local_cell_ids = _replicated_local_cell_ids(oriented, replicated)
    local_layer_ids = _replicated_local_layer_ids(oriented, replicated)
    replicated.wrap()
    if len(replicated) != atom_count:
        raise ValueError(
            f"--replication {' '.join(map(str, repeats))} produces "
            f"{len(replicated)} reference atoms after "
            f"{'orthogonalization' if was_orthogonalized else 'no orthogonalization'}, "
            f"but the trajectory contains {atom_count}."
        )
    natural_fractional = np.asarray(
        replicated.get_scaled_positions(wrap=False), dtype=float
    )
    reference_cell = np.asarray(replicated.cell.array, dtype=float)
    # Match ordinary homogeneous strain along axes whose lengths are close.
    # A large mismatch is normally vacuum in a slab box, so preserve the
    # reference thickness on that axis instead of stretching atoms into it.
    applied_strain = np.where(
        np.abs(strain - 1.0) <= float(request.max_reference_strain),
        strain,
        1.0,
    )
    reference_lengths = np.linalg.norm(reference_cell, axis=1)
    target_lengths = np.linalg.norm(target_cell, axis=1)
    target_directions = target_cell / target_lengths[:, None]
    strained_reference_cell = (
            target_directions * (reference_lengths * applied_strain)[:, None]
    )
    replicated.set_cell(strained_reference_cell, scale_atoms=False)
    replicated.set_scaled_positions(natural_fractional)
    # Embed the finite reference lattice in the simulation box without
    # rescaling it. Its fractional coordinates can then follow later changes
    # of the simulation cell while retaining the frame-zero slab/vacuum split.
    replicated.set_cell(target_cell, scale_atoms=False)
    reference_fractional = np.asarray(replicated.positions, dtype=float) @ np.linalg.inv(
        target_cell
    )
    reference_positions = np.asarray(trajectory.positions[frame], dtype=float)
    if not np.isfinite(reference_positions).all():
        raise ValueError("The reference trajectory frame contains non-finite coordinates.")
    labels = _frame_labels(trajectory, frame)
    assignment, translation, _ = _match_reference(
        reference_positions, labels, replicated, target_cell, request
    )
    inverse = np.empty_like(assignment)
    inverse[assignment] = np.arange(len(assignment), dtype=int)
    aligned_fractional = reference_fractional + translation
    aligned_positions = aligned_fractional @ target_cell
    local_cell_count = int(np.max(local_cell_ids)) + 1
    local_cell_centers = np.vstack([
        np.mean(aligned_positions[local_cell_ids == cell_id], axis=0)
        for cell_id in range(local_cell_count)
    ])
    local_layer_count = int(np.max(local_layer_ids)) + 1
    local_layer_centers = np.vstack([
        np.mean(aligned_positions[local_layer_ids == layer_id], axis=0)
        for layer_id in range(local_layer_count)
    ])
    replicated.set_positions(aligned_positions)
    replicated.set_pbc(_periodic_axes(request.periodic))
    return PreparedHBNReference(
        atoms=replicated,
        fractional_positions=reference_fractional,
        simulation_to_reference=assignment,
        reference_to_simulation=inverse,
        translation_fractional=translation,
        aligned_positions=aligned_positions,
        repeats=repeats,
        orthogonalized=was_orthogonalized,
        reference_angles=_angles(reference_cell),
        trajectory_angles=target_angles,
        strain_ratios=strain,
        applied_strain_ratios=applied_strain,
        local_cell_ids=local_cell_ids,
        local_cell_centers=local_cell_centers,
        local_layer_ids=local_layer_ids,
        local_layer_centers=local_layer_centers,
    )


def _formal_charge_map(values: Mapping[str, float]) -> dict[str, float]:
    result = {str(key).casefold(): float(value) for key, value in values.items()}
    invalid = [key for key, value in result.items() if not np.isfinite(value)]
    if invalid:
        raise ValueError(f"Formal charges must be finite; invalid: {invalid}.")
    return result


def calculate_hbn_reference_polarization(
        data: TrajectoryData | ElectrostaticsData,
        request: HBNReferencePolarizationRequest,
) -> HBNReferencePolarizationResult:
    """Calculate longitudinal dipole and polarization for selected frames."""

    if request.charge_source not in {"auto", "reaxff", "formal"}:
        raise ValueError("charge_source must be 'auto', 'reaxff', or 'formal'.")
    trajectory, dynamic_charges = _trajectory_and_charges(data)
    resolved_charge_source = (
        "reaxff"
        if request.charge_source == "reaxff"
           or (request.charge_source == "auto" and dynamic_charges is not None)
        else "formal"
    )
    if resolved_charge_source == "reaxff" and dynamic_charges is None:
        raise ValueError("--charge-source reaxff requires per-atom charges from fort.7.")
    if (
            dynamic_charges is not None
            and dynamic_charges.shape != np.asarray(trajectory.positions).shape[:2]
    ):
        raise ValueError("Per-atom charges must match the trajectory frame and atom counts.")

    prepared = prepare_hbn_reference(trajectory, request)
    selected = _selected_frames(trajectory, request)
    periodic = _periodic_axes(request.periodic)
    c_hat = _unit_vector(request.c_axis, "c_axis")
    formal = _formal_charge_map(request.formal_charges)
    positions = np.asarray(trajectory.positions, dtype=float)
    atom_ids = np.asarray(trajectory.atom_ids, dtype=int)
    assignment = prepared.simulation_to_reference
    reference_symbols = np.asarray(prepared.atoms.get_chemical_symbols(), dtype=object)
    factor_value = const("ea3_to_uC_cm2")
    debye_value = const("ea_to_debye")
    if factor_value is None or debye_value is None:  # pragma: no cover
        raise RuntimeError("Required dipole/polarization conversion constants are missing.")
    factor = float(factor_value)
    debye_factor = float(debye_value)
    displacement_tables: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = []
    output_iterations: list[int] = []

    reference_labels = _frame_labels(trajectory, int(request.reference_frame))
    if resolved_charge_source == "formal":
        missing = sorted(
            {str(value) for value in reference_labels if str(value).casefold() not in formal},
            key=str.casefold,
        )
        if missing:
            raise ValueError(
                "Missing formal charge(s) for trajectory species: "
                f"{', '.join(missing)}. Supply each value with --formal-charge ELEMENT=CHARGE."
            )

    for frame in selected:
        xyz = positions[frame]
        if not np.isfinite(xyz).all():
            raise ValueError(f"Frame {frame} contains non-finite coordinates.")
        labels = _frame_labels(trajectory, frame)
        if len(labels) != len(reference_labels):
            raise ValueError(f"Frame {frame} has a different atom count.")
        cell = _frame_cell(trajectory, request, frame)
        inverse_cell = np.linalg.inv(cell)
        simulation_fractional = xyz @ inverse_cell
        translation = _translation_from_assignment(
            simulation_fractional,
            prepared.fractional_positions,
            assignment,
            cell,
            periodic,
        )
        reference_for_atoms = prepared.fractional_positions[assignment] + translation
        delta_fractional = _minimum_image(
            simulation_fractional - reference_for_atoms, periodic
        )
        displacement = delta_fractional @ cell
        displacement_c = displacement @ c_hat
        charges = (
            np.asarray(dynamic_charges[frame], dtype=float)
            if resolved_charge_source == "reaxff"
            else np.asarray([formal.get(str(label).casefold(), np.nan) for label in labels])
        )
        if not np.isfinite(charges).all():
            bad = sorted({str(labels[index]) for index in np.flatnonzero(~np.isfinite(charges))})
            raise ValueError(
                f"Frame {frame} has missing or non-finite atomic charge(s): {', '.join(bad)}."
            )
        dipole = ELECTRON_CHARGE_SIGN * charges[:, None] * displacement
        dipole_c = dipole @ c_hat
        iteration = _iteration(trajectory, frame)
        output_iterations.append(iteration)
        simulation_volume = abs(float(np.linalg.det(cell)))
        if request.volume_method == "cell":
            volume = simulation_volume
        else:
            estimator = _hull_volume if request.volume_method == "hull" else _bbox_volume
            volume = float(estimator(xyz))
        total_dipole = np.sum(dipole, axis=0)
        total_dipole_c = float(np.sum(dipole_c))
        rms = float(np.sqrt(np.mean(np.sum(displacement * displacement, axis=1))))
        maximum = float(np.max(np.linalg.norm(displacement, axis=1)))
        summary_rows.append({
            "frame_index": int(frame),
            "iter": iteration,
            "atom_count": len(xyz),
            "reference_repeat_a": prepared.repeats[0],
            "reference_repeat_b": prepared.repeats[1],
            "reference_repeat_c": prepared.repeats[2],
            "reference_orthogonalized": prepared.orthogonalized,
            "alignment_shift_x (angstrom)": float((translation @ cell)[0]),
            "alignment_shift_y (angstrom)": float((translation @ cell)[1]),
            "alignment_shift_z (angstrom)": float((translation @ cell)[2]),
            "rms_displacement (angstrom)": rms,
            "max_displacement (angstrom)": maximum,
            "volume_method": request.volume_method,
            "charge_source": resolved_charge_source,
            "electron_charge_sign": ELECTRON_CHARGE_SIGN,
            "volume (angstrom^3)": volume,
            "simulation_cell_volume (angstrom^3)": simulation_volume,
            **{
                quantity: value
                for component, axis in enumerate("xyz")
                for quantity, value in (
                    (f"dipole_{axis} (e*angstrom)", float(total_dipole[component])),
                    (f"dipole_{axis} (debye)", float(total_dipole[component] * debye_factor)),
                    (
                        f"P_{axis} (uC/cm^2)",
                        float(total_dipole[component] / volume * factor)
                        if np.isfinite(volume) and volume > 0.0
                        else np.nan,
                    ),
                )
            },
            "dipole_c (e*angstrom)": total_dipole_c,
            "dipole_c (debye)": total_dipole_c * debye_factor,
            "P_c (uC/cm^2)": (
                total_dipole_c / volume * factor
                if np.isfinite(volume) and volume > 0.0
                else np.nan
            ),
        })
        reference_positions = reference_for_atoms @ cell
        frame_columns: dict[str, object] = {
            "frame_index": np.full(len(xyz), int(frame), dtype=int),
            "iter": np.full(len(xyz), iteration, dtype=int),
            "atom_index": np.arange(len(xyz), dtype=int),
            "atom_id": atom_ids,
            "element": labels.astype(str),
            "reference_atom_index": assignment,
            "reference_element": reference_symbols[assignment].astype(str),
            "local_cell_id": prepared.local_cell_ids[assignment],
            "local_layer_id": prepared.local_layer_ids[assignment],
            "charge_source": np.full(len(xyz), resolved_charge_source, dtype=object),
            "charge (e)": charges,
            "electron_charge_sign": np.full(len(xyz), ELECTRON_CHARGE_SIGN),
            "displacement_c (angstrom)": displacement_c,
            "dipole_c (e*angstrom)": dipole_c,
            "dipole_c (debye)": dipole_c * debye_factor,
        }
        for component, axis in enumerate("xyz"):
            frame_columns[f"{axis} (angstrom)"] = xyz[:, component]
            frame_columns[f"reference_{axis} (angstrom)"] = reference_positions[:, component]
            frame_columns[f"displacement_{axis} (angstrom)"] = displacement[:, component]
            frame_columns[f"dipole_{axis} (e*angstrom)"] = dipole[:, component]
            frame_columns[f"dipole_{axis} (debye)"] = dipole[:, component] * debye_factor
        displacement_tables.append(pd.DataFrame(frame_columns))

    base_labels = _frame_labels(trajectory, int(request.reference_frame))
    mapping_columns: dict[str, object] = {
        "atom_index": np.arange(len(assignment), dtype=int),
        "atom_id": atom_ids,
        "element": base_labels.astype(str),
        "reference_atom_index": assignment,
        "reference_element": reference_symbols[assignment].astype(str),
        "local_cell_id": prepared.local_cell_ids[assignment],
        "local_layer_id": prepared.local_layer_ids[assignment],
    }
    for component, axis in enumerate("xyz"):
        mapping_columns[f"reference_{axis} (angstrom)"] = prepared.aligned_positions[
            assignment, component
        ]
    return HBNReferencePolarizationResult(
        table=pd.DataFrame(summary_rows),
        request=request,
        displacements=pd.concat(displacement_tables, ignore_index=True),
        mapping=pd.DataFrame(mapping_columns),
        reference=prepared,
        frame_indices=np.asarray(selected, dtype=int),
        iterations=np.asarray(output_iterations, dtype=int),
    )


def write_aligned_reference_xyz(
        result: HBNReferencePolarizationResult, path: str | Path
) -> Path:
    """Write the replicated, strained, translated reference as extended XYZ."""

    atoms = result.reference.atoms.copy()
    atoms.info["reference_repeats"] = " ".join(map(str, result.reference.repeats))
    atoms.info["orthogonalized"] = bool(result.reference.orthogonalized)
    return write_structure(atoms, path, format="extxyz")


@register_task(
    "get-hbn-reference-polarization",
    label="h-BN-reference Polarization",
)
class HBNReferencePolarizationTask(AnalysisTask):
    """Calculate vector polarization relative to replicated h-AlN."""

    required_data = TrajectoryData
    supports_selective_streaming = False
    VERSION = "6"

    def required_data_for(
            self, request: HBNReferencePolarizationRequest, args: dict | None = None
    ):
        return required_wurtzite_data_type(request, args)

    @staticmethod
    def required_data_fields_for(
            request: HBNReferencePolarizationRequest, args: dict
    ) -> tuple[str, ...]:
        return (
            ("trajectory",)
            if required_wurtzite_data_type(request, args) is TrajectoryData
            else ("trajectory", "charges")
        )

    @classmethod
    def recommended_presentations(
            cls, _result: object, payload: dict[str, Any]
    ) -> list[PresentationSpec]:
        _ = (cls, payload)
        return [
            PresentationSpec(
                renderer="table", label="h-BN-reference polarization", view_type="table"
            )
        ]

    def run(
            self,
            data: TrajectoryData | ElectrostaticsData,
            request: HBNReferencePolarizationRequest,
            reporter=None,
    ) -> HBNReferencePolarizationResult:
        _ = reporter
        return calculate_hbn_reference_polarization(data, request)


__all__ = [
    "ChargeSource",
    "DEFAULT_FORMAL_CHARGES",
    "ELECTRON_CHARGE_SIGN",
    "HBNReferencePolarizationRequest",
    "HBNReferencePolarizationResult",
    "HBNReferencePolarizationTask",
    "OrthogonalizeMode",
    "PreparedHBNReference",
    "REFERENCE_STRUCTURE_PATH",
    "VolumeMethod",
    "calculate_hbn_reference_polarization",
    "prepare_hbn_reference",
    "write_aligned_reference_xyz",
]
