"""Detect coarse molecular isomer representatives from canonical data.

This analyzer selects same-formula molecular representatives from a
``ConnectivityTrajectoryData`` bundle. It uses per-frame molecule assignments,
element labels, and connectivity matrices already normalized by engine adapters;
it does not read engine files such as ReaxFF ``fort.7`` or ``xmolout`` directly.

The representative signature is intentionally coarser than full
graph-isomorphism based isomer enumeration. For training pipelines this is a
faster and cheaper screening stage because it can reduce the number of
downstream Jaguar (DFT) jobs while still sampling distinct same-formula bonding
environments.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field as dc_field
from typing import Any, Sequence

import numpy as np
import pandas as pd

from reaxkit.analysis.base import AnalysisTask
from reaxkit.core.registry.analysis_task_registry import register_task
from reaxkit.domain.base_request import BaseRequest
from reaxkit.domain.base_result import BaseResult
from reaxkit.domain.data_models import ConnectivityTrajectoryData
from reaxkit.presentation.specs import PresentationSpec


def _normalize_element_symbol(value: str) -> str:
    """Normalize an element token to standard symbol capitalization."""
    token = str(value).strip()
    if not token:
        return ""
    if len(token) == 1:
        return token.upper()
    return token[0].upper() + token[1:].lower()


def _normalize_formula_counts(formula: dict[str, int]) -> dict[str, int]:
    """Return normalized non-empty formula counts."""
    out: dict[str, int] = {}
    for raw_symbol, raw_count in dict(formula).items():
        symbol = _normalize_element_symbol(str(raw_symbol))
        if not symbol:
            raise ValueError("target_formula contains an empty element symbol.")
        count = int(raw_count)
        if count < 1:
            raise ValueError("target_formula counts must be positive integers.")
        out[symbol] = out.get(symbol, 0) + count
    if not out:
        raise ValueError("target_formula cannot be empty.")
    return out


def _formula_label(formula: dict[str, int]) -> str:
    """Build a stable formula label from normalized counts."""
    return "".join(f"{symbol}{int(formula[symbol])}" for symbol in sorted(formula))


@dataclass
class IsomerRepresentativeDetectionRequest(BaseRequest):
    """Request for coarse molecular isomer representative detection.

    Fields
    -----
    target_formula : dict[str, int]
        Formula that selected molecules must match exactly.
    structure_prefix : str
        Prefix for representative structure names. If empty, a formula label is
        derived from ``target_formula``.
    max_representatives : int | None
        Optional positive cap on selected representatives.
    frame_indices : Sequence[int] | None
        Optional frame indices to scan. ``None`` means all frames.
    """

    target_formula: dict[str, int] = dc_field(
        default_factory=dict,
        metadata={
            "label": "Target Formula",
            "help": "Exact molecular formula to select, e.g. {'C': 8, 'H': 13, 'O': 3, 'B': 5}.",
        },
    )
    structure_prefix: str = dc_field(
        default="",
        metadata={
            "label": "Structure Prefix",
            "help": "Prefix for representative structure names. Empty derives a formula label.",
        },
    )
    max_representatives: int | None = dc_field(
        default=None,
        metadata={
            "label": "Max Representatives",
            "help": "Optional cap on representative structures.",
            "min": 1,
        },
    )
    frame_indices: Sequence[int] | None = dc_field(
        default=None,
        metadata={
            "label": "Frame Indices",
            "help": "Optional frame indices to scan. Empty means all frames.",
            "units": "frame_index",
        },
    )


@dataclass(frozen=True)
class IsomerRepresentativeRecord:
    """One coarse isomer representative."""

    isomer_index: int
    structure_name: str
    frame_index: int
    iteration: int
    molecule_id: int
    atom_count: int
    atom_ids: tuple[int, ...]
    bond_type_counts: dict[tuple[str, str], int] = dc_field(default_factory=dict)
    bond_label_counts: dict[str, int] = dc_field(default_factory=dict)


@dataclass
class IsomerRepresentativeDetectionResult(BaseResult):
    """Result for coarse molecular isomer representative detection."""

    table: pd.DataFrame
    records: list[IsomerRepresentativeRecord]
    request: IsomerRepresentativeDetectionRequest


def _empty_representative_table() -> pd.DataFrame:
    """Return an empty representative table with stable columns."""
    return pd.DataFrame(
        columns=[
            "isomer_index",
            "structure_name",
            "frame_index",
            "iteration",
            "molecule_id",
            "atom_count",
            "atom_ids",
            "bond_signature",
        ]
    )


def _frame_matrix(frames: Any, frame_index: int, *, n_atoms: int, field_name: str) -> np.ndarray:
    """Return one dense frame matrix from dense, stacked, or sparse-frame data."""
    if frames is None:
        raise ValueError(f"ConnectivityData.{field_name} is required for representative detection.")
    if isinstance(frames, list | tuple):
        frame = frames[frame_index]
    else:
        arr = np.asarray(frames)
        if arr.ndim == 3:
            frame = arr[frame_index]
        elif arr.ndim == 2 and frame_index == 0:
            frame = arr
        else:
            raise ValueError(f"ConnectivityData.{field_name} has unsupported shape {arr.shape}.")
    if hasattr(frame, "toarray"):
        matrix = np.asarray(frame.toarray(), dtype=float)
    else:
        matrix = np.asarray(frame, dtype=float)
    if matrix.shape != (n_atoms, n_atoms):
        raise ValueError(
            f"ConnectivityData.{field_name} frame shape must be {(n_atoms, n_atoms)}, got {matrix.shape}."
        )
    return matrix


def _iteration_for_frame(data: ConnectivityTrajectoryData, frame_index: int) -> int:
    """Return iteration number for a frame."""
    for source in (data.trajectory.iterations, data.connectivity.iterations):
        if source is not None:
            values = np.asarray(source, dtype=int)
            if frame_index < values.shape[0]:
                return int(values[frame_index])
    return int(frame_index)


def _molecule_nums(data: ConnectivityTrajectoryData) -> np.ndarray:
    """Return per-frame/per-atom molecule ids."""
    sim = data.connectivity.simulation or data.trajectory.simulation
    molecule_nums = getattr(sim, "molecule_nums", None) if sim is not None else None
    if molecule_nums is None:
        raise ValueError(
            "ConnectivityTrajectoryData requires simulation.molecule_nums for representative detection."
        )
    arr = np.asarray(molecule_nums, dtype=int)
    if arr.ndim != 2:
        raise ValueError("simulation.molecule_nums must have shape (n_frames, n_atoms).")
    return arr


def _selected_frame_indices(n_frames: int, requested: Sequence[int] | None) -> list[int]:
    """Validate and return selected frame indices."""
    if requested is None:
        return list(range(n_frames))
    out = []
    for raw in requested:
        frame_index = int(raw)
        if frame_index < 0 or frame_index >= n_frames:
            raise ValueError(f"frame index {frame_index} is outside [0, {n_frames - 1}].")
        out.append(frame_index)
    return out


def _element_counts(atom_indices: Sequence[int], elements: Sequence[str]) -> dict[str, int]:
    """Count elements for selected atom indices."""
    counts: dict[str, int] = defaultdict(int)
    for atom_index in atom_indices:
        symbol = _normalize_element_symbol(str(elements[int(atom_index)]))
        counts[symbol] += 1
    return dict(counts)


def _bond_type_counts(
    atom_indices: Sequence[int],
    elements: Sequence[str],
    connectivity_matrix: np.ndarray,
) -> dict[tuple[str, str], int]:
    """Count doubled directed bonds by sorted element pair."""
    component = set(int(i) for i in atom_indices)
    counts: dict[tuple[str, str], int] = defaultdict(int)
    for src in atom_indices:
        src_symbol = _normalize_element_symbol(str(elements[int(src)]))
        neighbors = np.nonzero(connectivity_matrix[int(src)] > 0.0)[0]
        for dst in neighbors.tolist():
            if int(dst) not in component:
                continue
            dst_symbol = _normalize_element_symbol(str(elements[int(dst)]))
            key = tuple(sorted((src_symbol, dst_symbol)))
            counts[key] += 1
    return dict(counts)


def _bond_label_counts(bond_type_counts: dict[tuple[str, str], int]) -> dict[str, int]:
    """Convert doubled directed bond counts to undirected label counts."""
    labels: dict[str, int] = {}
    for (first, second), count in sorted(bond_type_counts.items()):
        labels[f"{first}-{second}"] = int(count / 2)
    return labels


def _records_table(records: list[IsomerRepresentativeRecord]) -> pd.DataFrame:
    """Build output table from representative records."""
    if not records:
        return _empty_representative_table()
    return pd.DataFrame(
        [
            {
                "isomer_index": record.isomer_index,
                "structure_name": record.structure_name,
                "frame_index": record.frame_index,
                "iteration": record.iteration,
                "molecule_id": record.molecule_id,
                "atom_count": record.atom_count,
                "atom_ids": list(record.atom_ids),
                "bond_signature": ";".join(
                    f"{label}:{count}" for label, count in record.bond_label_counts.items()
                ),
            }
            for record in records
        ],
        columns=list(_empty_representative_table().columns),
    )


@register_task("isomer_representative_detection", label="Isomer Representative Detection")
class IsomerRepresentativeDetectionTask(AnalysisTask):
    """Detect coarse same-formula isomer representatives from canonical data."""

    required_data = ConnectivityTrajectoryData

    @staticmethod
    def recommended_presentations(
        _result: IsomerRepresentativeDetectionResult,
        _payload: dict[str, Any],
    ) -> list[PresentationSpec]:
        """Recommend the representative table as the default presentation."""
        return [PresentationSpec(renderer="table", label="Representatives", view_type="table")]

    def run(
        self,
        data: ConnectivityTrajectoryData,
        request: IsomerRepresentativeDetectionRequest,
        reporter=None,
    ) -> IsomerRepresentativeDetectionResult:
        """Detect coarse same-formula representatives.

        Works on
        -----
        ``ConnectivityTrajectoryData`` plus representative-detection settings.

        Parameters
        -----
        data : ConnectivityTrajectoryData
            Canonical trajectory/connectivity bundle from an engine adapter.
        request : IsomerRepresentativeDetectionRequest
            Target formula, structure prefix, and optional frame/count limits.
        reporter : Any, optional
            Optional progress callback.
        """
        if not isinstance(data, ConnectivityTrajectoryData):
            raise TypeError("IsomerRepresentativeDetectionTask expects ConnectivityTrajectoryData.")

        formula = _normalize_formula_counts(request.target_formula)
        total_atoms = int(sum(formula.values()))
        prefix = str(request.structure_prefix or _formula_label(formula)).strip()
        if not prefix:
            prefix = _formula_label(formula)
        if request.max_representatives is not None and int(request.max_representatives) < 1:
            raise ValueError("max_representatives must be a positive integer when provided.")
        max_count = int(request.max_representatives) if request.max_representatives is not None else None

        trajectory = data.trajectory
        positions = np.asarray(trajectory.positions, dtype=float)
        if positions.ndim != 3:
            raise ValueError("TrajectoryData.positions must have shape (n_frames, n_atoms, 3).")
        n_frames, n_atoms, _ = positions.shape
        if len(trajectory.atom_ids) != n_atoms:
            raise ValueError("TrajectoryData.atom_ids length must match trajectory atom count.")
        if len(trajectory.elements) != n_atoms:
            raise ValueError("TrajectoryData.elements length must match trajectory atom count.")

        molecule_nums = _molecule_nums(data)
        if molecule_nums.shape != (n_frames, n_atoms):
            raise ValueError("simulation.molecule_nums shape must match trajectory frame and atom dimensions.")

        frame_indices = _selected_frame_indices(n_frames, request.frame_indices)
        records: list[IsomerRepresentativeRecord] = []
        seen_signatures: set[tuple[tuple[tuple[str, str], int], ...]] = set()

        for step_i, frame_index in enumerate(frame_indices, start=1):
            if reporter:
                reporter("analyze", step_i, len(frame_indices), "Detecting isomer representatives")
            connectivity_matrix = _frame_matrix(
                data.connectivity.connectivity,
                frame_index,
                n_atoms=n_atoms,
                field_name="connectivity",
            )
            valid = np.isfinite(positions[frame_index]).all(axis=1)
            mol_ids = molecule_nums[frame_index]
            molecule_to_indices: dict[int, list[int]] = defaultdict(list)
            for atom_index, molecule_id in enumerate(mol_ids.tolist()):
                if not bool(valid[atom_index]):
                    continue
                molecule_id = int(molecule_id)
                if molecule_id <= 0:
                    continue
                molecule_to_indices[molecule_id].append(atom_index)

            for molecule_id, atom_indices in molecule_to_indices.items():
                if len(atom_indices) != total_atoms:
                    continue
                if _element_counts(atom_indices, trajectory.elements) != formula:
                    continue
                bond_counts = _bond_type_counts(atom_indices, trajectory.elements, connectivity_matrix)
                signature = tuple(sorted(bond_counts.items()))
                if signature in seen_signatures:
                    continue
                seen_signatures.add(signature)
                isomer_index = len(records)
                atom_ids = tuple(int(trajectory.atom_ids[int(i)]) for i in atom_indices)
                records.append(
                    IsomerRepresentativeRecord(
                        isomer_index=isomer_index,
                        structure_name=f"{prefix}_{isomer_index}",
                        frame_index=int(frame_index),
                        iteration=_iteration_for_frame(data, frame_index),
                        molecule_id=int(molecule_id),
                        atom_count=total_atoms,
                        atom_ids=atom_ids,
                        bond_type_counts=bond_counts,
                        bond_label_counts=_bond_label_counts(bond_counts),
                    )
                )
                if max_count is not None and len(records) >= max_count:
                    table = _records_table(records)
                    return IsomerRepresentativeDetectionResult(table=table, records=records, request=request)

        table = _records_table(records)
        return IsomerRepresentativeDetectionResult(table=table, records=records, request=request)


__all__ = [
    "IsomerRepresentativeDetectionRequest",
    "IsomerRepresentativeDetectionResult",
    "IsomerRepresentativeDetectionTask",
    "IsomerRepresentativeRecord",
]
