"""Trajectory relabeling tasks."""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from reaxkit.analysis.base import AnalysisTask
from reaxkit.analysis.connectivity.coordination import CoordinationStatusRequest, CoordinationStatusTask
from reaxkit.core.analysis_task_registry import register_task
from reaxkit.domain.base_request import BaseRequest
from reaxkit.domain.base_result import BaseResult
from reaxkit.domain.data_models import (
    ConnectivityTrajectoryData,
    CoordinationStatusBundleData,
    ForceFieldParametersData,
    SimulationData,
    TrajectoryData,
)
from reaxkit.presentation.specs import PresentationSpec


def _subset_optional_array(values, frame_idx: np.ndarray):
    if values is None:
        return None
    arr = np.asarray(values)
    if arr.ndim == 0:
        return values
    return arr[frame_idx]


def _subset_simulation(simulation: SimulationData | None, frame_idx: np.ndarray, elements: list[str]) -> SimulationData | None:
    if simulation is None:
        return None
    return SimulationData(
        atom_ids=list(simulation.atom_ids),
        iterations=_subset_optional_array(simulation.iterations, frame_idx),
        time=_subset_optional_array(simulation.time, frame_idx),
        elements=list(elements),
        num_of_atoms=_subset_optional_array(simulation.num_of_atoms, frame_idx),
        potential_energy=_subset_optional_array(simulation.potential_energy, frame_idx),
        volume=_subset_optional_array(simulation.volume, frame_idx),
        temperature=_subset_optional_array(simulation.temperature, frame_idx),
        pressure=_subset_optional_array(simulation.pressure, frame_idx),
        density=_subset_optional_array(simulation.density, frame_idx),
        elapsed_time=_subset_optional_array(simulation.elapsed_time, frame_idx),
        atom_type_nums=_subset_optional_array(simulation.atom_type_nums, frame_idx),
        molecule_nums=_subset_optional_array(simulation.molecule_nums, frame_idx),
        cell_lengths=_subset_optional_array(simulation.cell_lengths, frame_idx),
        cell_angles=_subset_optional_array(simulation.cell_angles, frame_idx),
    )


@dataclass
class TrajectoryRelabelByCoordinationRequest(BaseRequest):
    """Request to relabel trajectory atom labels from coordination status.

    Parameters
    ----------
    labels
        Optional mapping from status code to tag string.
        Defaults are ``{-1: "U", 0: "C", 1: "O"}``.
        Example: ``{-1: "UN", 0: "OK", 1: "OV"}``.
    mode
        Relabeling style:
        - ``global``: replace labels with status tags only
        - ``by_type``: append status tag to original atom type
        Example: ``mode="by_type"``.
    keep_coord_original
        Used only with ``mode="by_type"``. If ``True``, atoms with status ``0``
        keep original labels instead of adding the coordinated tag.
    frames
        Optional frame indices to relabel. If omitted, all frames are used.
        Example: ``[0, 10, 20]``.
    every
        Frame stride after selection. Example: ``every=5``.
    valences
        Optional element-to-valence map passed to coordination classification.
        Example: ``{"C": 4.0, "O": 2.0, "H": 1.0}``.
    threshold
        Absolute tolerance around valence used for coordination classification.
        Example: ``0.9``.
    require_all_valences
        If ``True``, fail when selected atom types have no valence mapping.
    """

    labels: Optional[Mapping[int, str]] = dc_field(
        default=None,
        metadata={
            'label': 'Labels',
            'help': "Optional status->tag mapping. Example: {-1:'UN', 0:'OK', 1:'OV'}.",
        },
    )
    mode: str = dc_field(
        default="global",
        metadata={
            'label': 'Mode',
            'help': "Relabeling mode. 'global' uses only status tags, 'by_type' appends tags to original atom types.",
            'choices': ['global', 'by_type'],
        },
    )
    keep_coord_original: bool = dc_field(
        default=False,
        metadata={
            'label': 'Keep Coord Original',
            'help': "In by_type mode, keep original label for coordinated atoms (status=0).",
            'choices': [True, False],
        },
    )
    frames: Optional[Sequence[int]] = dc_field(
        default=None,
        metadata={
            'label': 'Frames',
            'help': "Optional frame indices to relabel. Example: [0, 10, 20].",
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
    valences: Optional[Mapping[str, float]] = dc_field(
        default=None,
        metadata={
            'label': 'Valences',
            'help': "Optional element->valence map for coordination classification. Example: {'C': 4, 'O': 2, 'H': 1}.",
        },
    )
    threshold: float = dc_field(
        default=0.9,
        metadata={
            'label': 'Threshold',
            'help': "Absolute tolerance used to classify under/coord/over status. Example: 0.9.",
            'min': 0.0,
        },
    )
    require_all_valences: bool = dc_field(
        default=True,
        metadata={
            'label': 'Require All Valences',
            'help': "If true, fail when any selected atom type has no valence mapping.",
            'choices': [True, False],
        },
    )


@dataclass
class TrajectoryRelabelByCoordinationResult(BaseResult):
    """Relabeled trajectory generated from computed coordination status.

    Output structure
    ----------------
    - ``request``: the :class:`TrajectoryRelabelByCoordinationRequest` used.
    - ``trajectory``: trajectory subset with relabeled ``atom_labels`` per frame.
    - ``table``: coordination-status table used for relabeling.

    Table columns
    -------------
    Typical columns include:
    ``frame_index``, ``iter``, ``atom_id``, ``atom_type``, ``sum_BOs``,
    ``valence``, ``delta``, ``status``, ``status_label``.

    Example
    -------
    With ``mode='by_type'`` and default labels, an oxygen atom with status ``1``
    is relabeled from ``O`` to ``OO`` in that frame.
    """

    trajectory: TrajectoryData
    table: pd.DataFrame
    request: TrajectoryRelabelByCoordinationRequest


def _status_labels(mapping: Optional[Mapping[int, str]]) -> dict[int, str]:
    out = {-1: "U", 0: "C", 1: "O"}
    if mapping:
        for key, value in mapping.items():
            out[int(key)] = str(value)
    return out


def _selected_frames(data: TrajectoryData, request: TrajectoryRelabelByCoordinationRequest) -> np.ndarray:
    n_frames = int(np.asarray(data.positions).shape[0])
    if request.frames is not None:
        idx = [int(i) for i in request.frames if 0 <= int(i) < n_frames]
    else:
        idx = list(range(n_frames))
    stride = max(1, int(request.every))
    return np.asarray(idx[::stride], dtype=int)


def _empty_force_field_parameters() -> ForceFieldParametersData:
    empty = pd.DataFrame()
    return ForceFieldParametersData(
        general_parameters=empty.copy(),
        atom_parameters=empty.copy(),
        bond_parameters=empty.copy(),
        off_diagonal_parameters=empty.copy(),
        angle_parameters=empty.copy(),
        torsion_parameters=empty.copy(),
        hydrogen_bond_parameters=empty.copy(),
    )


@register_task("trajectory_relabel_by_coordination")
class TrajectoryRelabelByCoordinationTask(AnalysisTask):
    """Build a relabeled trajectory from coordination-status output."""

    required_data = ConnectivityTrajectoryData

    @staticmethod
    def recommended_presentations(
        _result: TrajectoryRelabelByCoordinationResult,
        payload: dict[str, Any],
    ) -> list[PresentationSpec]:
        return [PresentationSpec(renderer="table", label="Table", view_type="table")]

    def run(
        self,
        data: ConnectivityTrajectoryData,
        request: TrajectoryRelabelByCoordinationRequest,
        reporter=None,
    ) -> TrajectoryRelabelByCoordinationResult:
        trajectory = data.trajectory
        positions = np.asarray(trajectory.positions, dtype=float)
        if positions.ndim != 3:
            raise ValueError("TrajectoryData.positions must have shape (n_frames, n_atoms, 3).")

        n_frames, n_atoms, _ = positions.shape
        atom_ids = [int(atom_id) for atom_id in trajectory.atom_ids]
        if len(atom_ids) != n_atoms:
            raise ValueError("TrajectoryData.atom_ids length must match positions atom count.")

        base_elements = [str(element) for element in trajectory.elements]
        if len(base_elements) != n_atoms:
            raise ValueError("TrajectoryData.elements length must match positions atom count.")

        force_field = data.force_field_parameters
        if force_field is None and request.valences is None:
            raise ValueError(
                "Relabel-by-coordination requires force-field valences (via ConnectivityTrajectoryData.force_field_parameters) "
                "or explicit request.valences."
            )
        if force_field is None:
            force_field = _empty_force_field_parameters()

        coordination_result = CoordinationStatusTask().run(
            CoordinationStatusBundleData(
                connectivity=data.connectivity,
                force_field_parameters=force_field,
            ),
            CoordinationStatusRequest(
                valences=request.valences,
                threshold=float(request.threshold),
                frames=request.frames,
                every=int(request.every),
                require_all_valences=bool(request.require_all_valences),
            ),
        )
        table = coordination_result.table.copy()
        frame_idx = _selected_frames(trajectory, request)
        labels_by_frame = np.tile(np.asarray(base_elements, dtype=object), (len(frame_idx), 1))
        label_map = _status_labels(request.labels)
        atom_to_index = {atom_id: i for i, atom_id in enumerate(atom_ids)}

        if not table.empty:
            required = {"frame_index", "atom_id", "status"}
            missing = required.difference(table.columns)
            if missing:
                raise ValueError(f"coordination_table is missing required columns: {sorted(missing)}")

            frame_to_local = {int(fi): i for i, fi in enumerate(frame_idx.tolist())}
            table = table[table["frame_index"].isin(frame_to_local)].copy()
            table["atom_id"] = pd.to_numeric(table["atom_id"], errors="coerce").astype("Int64")
            table["status"] = pd.to_numeric(table["status"], errors="coerce")

            for row in table.itertuples(index=False):
                if pd.isna(row.atom_id) or pd.isna(row.status):
                    continue
                atom_index = atom_to_index.get(int(row.atom_id))
                local_frame = frame_to_local.get(int(row.frame_index))
                if atom_index is None or local_frame is None:
                    continue
                original = str(base_elements[atom_index])
                status = int(row.status)
                tag = label_map.get(status, label_map[0])
                if request.mode == "global":
                    labels_by_frame[local_frame, atom_index] = str(tag)
                elif request.mode == "by_type":
                    if status == 0 and request.keep_coord_original:
                        labels_by_frame[local_frame, atom_index] = original
                    else:
                        labels_by_frame[local_frame, atom_index] = f"{original}{tag}"
                else:
                    raise ValueError("mode must be 'global' or 'by_type'.")

        relabeled = TrajectoryData(
            positions=positions[frame_idx],
            elements=list(base_elements),
            atom_ids=list(atom_ids),
            simulation=_subset_simulation(trajectory.simulation, frame_idx, list(base_elements)),
            iterations=_subset_optional_array(trajectory.iterations, frame_idx),
            atom_labels=labels_by_frame,
        )
        return TrajectoryRelabelByCoordinationResult(
            trajectory=relabeled,
            table=table.reset_index(drop=True),
            request=request,
        )


__all__ = [
    "TrajectoryRelabelByCoordinationRequest",
    "TrajectoryRelabelByCoordinationResult",
    "TrajectoryRelabelByCoordinationTask",
]
