"""Relabel trajectory atom types using analyzer-derived classification signals.

This module rewrites trajectory-facing atom labels based on coordination status
or related analyzer outputs while preserving framewise coordinate alignment.
It is scoped to relabeling transformations and does not recompute raw dynamics.

**Usage context**

- State-aware relabeling: Convert raw atom labels into context-specific labels.
- Pipeline preparation: Produce relabeled trajectories for downstream analyses.
- Comparative workflows: Align labeling conventions across simulation runs.
"""

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

    Fields
    -----
    labels : Optional[Mapping[int, str]]
        Mapping from coordination status code to tag text. Defaults to
        `{-1: "U", 0: "C", 1: "O"}` when not provided.
    mode : str
        Relabeling mode:
        - `"global"`: replace labels with status tags only
        - `"by_type"`: append status tag to original atom type
    keep_coord_original : bool
        Only used with `"by_type"`. If `True`, status `0` keeps the original
        atom type label (no suffix).
    frames : Optional[Sequence[int]]
        Frame indices to relabel. `None` means all frames.
    every : int
        Stride over selected frames. Must be `>= 1`.
    valences : Optional[Mapping[str, float]]
        Optional element-to-valence map passed to coordination classification.
    threshold : float
        Absolute tolerance used in coordination-status classification.
    require_all_valences : bool
        If `True`, fail when selected atom types have no valence mapping.

    Examples
    -----
    ```python
    req = TrajectoryRelabelByCoordinationRequest(
        mode="by_type",
        labels={-1: "UN", 0: "OK", 1: "OV"},
        frames=[0, 10, 20],
    )
    ```
    Sample output:
    `TrajectoryRelabelByCoordinationRequest(...)`
    Meaning:
    The request configures coordination-driven relabeling on sampled frames.
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

    Fields
    -----
    trajectory : TrajectoryData
        Relabeled trajectory subset whose `atom_labels` contains frame-wise
        relabel outputs.
    table : pd.DataFrame
        Coordination-status table used as the relabel source.
    request : TrajectoryRelabelByCoordinationRequest
        Request object used for this analysis execution.

    Notes
    -----
    The `table` usually includes columns such as `frame_index`, `iter`,
    `atom_id`, `atom_type`, `sum_BOs`, `valence`, `delta`, `status`,
    and `status_label`.

    Examples
    -----
    ```python
    result = TrajectoryRelabelByCoordinationTask().run(data, req)
    result.trajectory.atom_labels.shape
    ```
    Sample output:
    `(n_selected_frames, n_atoms)`
    Meaning:
    Relabeled atom labels are stored per selected frame and atom.
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


@register_task("trajectory_relabel_by_coordination", label="Trajectory Relabel by Coordination")
class TrajectoryRelabelByCoordinationTask(AnalysisTask):
    """Build a relabeled trajectory from coordination-status output."""

    required_data = ConnectivityTrajectoryData

    @staticmethod
    def recommended_presentations(
        _result: TrajectoryRelabelByCoordinationResult,
        payload: dict[str, Any],
    ) -> list[PresentationSpec]:
        """Build default presentations for relabel-by-coordination outputs.

        Works on
        -----
        Analyzer task output payloads

        Parameters
        -----
        _result : TrajectoryRelabelByCoordinationResult
            Analysis result object for the executed task.
        payload : dict[str, Any]
            Serialized result payload.

        Returns
        -----
        list[PresentationSpec]
            Table-only presentation configuration.

        Examples
        -----
        ```python
        specs = TrajectoryRelabelByCoordinationTask.recommended_presentations(result, payload)
        ```
        Sample output:
        `[PresentationSpec(renderer="table", ...)]`
        Meaning:
        The primary default representation is the coordination/relabel table.
        """
        return [PresentationSpec(renderer="table", label="Table", view_type="table")]

    def run(
        self,
        data: ConnectivityTrajectoryData,
        request: TrajectoryRelabelByCoordinationRequest,
        reporter=None,
    ) -> TrajectoryRelabelByCoordinationResult:
        """Relabel trajectory atom labels using coordination-status analysis.

        Computes coordination status from connectivity + valence information and
        applies requested relabeling mode to produce a relabeled trajectory
        subset aligned to selected frames.

        Works on
        -----
        `ConnectivityTrajectoryData` plus `TrajectoryRelabelByCoordinationRequest`

        Parameters
        -----
        data : ConnectivityTrajectoryData
            Bundle containing trajectory coordinates, connectivity, and optional
            force-field parameter data for valence lookup.
        request : TrajectoryRelabelByCoordinationRequest
            Relabeling configuration, frame sampling, and coordination options.
        reporter : Any, optional
            Optional progress callback passed through downstream operations.

        Returns
        -----
        TrajectoryRelabelByCoordinationResult
            Result containing relabeled trajectory subset and coordination table.

        Examples
        -----
        ```python
        req = TrajectoryRelabelByCoordinationRequest(mode="global")
        result = TrajectoryRelabelByCoordinationTask().run(bundle, req)
        ```
        Sample output:
        A result with `trajectory.atom_labels` and a coordination `table`.
        Meaning:
        Labels are transformed according to coordination status on sampled frames.
        """
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
