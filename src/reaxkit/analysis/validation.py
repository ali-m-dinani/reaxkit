"""Shared input validation for analysis tasks."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from reaxkit.core.exceptions import AnalysisError
from reaxkit.domain.base_request import BaseRequest
from reaxkit.domain.data_models import TrajectoryData


def _raise(task_name: str, message: str) -> None:
    raise AnalysisError(f"{task_name}: {message}")


def _is_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


def _validate_frames(task_name: str, frames: Any) -> None:
    if frames is None:
        return
    if not _is_sequence(frames):
        _raise(task_name, "frames must be a sequence of integers.")
    for v in frames:
        try:
            int(v)
        except Exception:
            _raise(task_name, f"invalid frame index '{v}'.")


def _validate_every(task_name: str, every: Any) -> None:
    if every is None:
        return
    try:
        ev = int(every)
    except Exception:
        _raise(task_name, "every must be an integer >= 1.")
    if ev < 1:
        _raise(task_name, "every must be >= 1.")


def _validate_numeric_positive(task_name: str, field_name: str, value: Any, *, allow_none: bool = True) -> None:
    if value is None and allow_none:
        return
    try:
        fv = float(value)
    except Exception:
        _raise(task_name, f"{field_name} must be numeric.")
    if not np.isfinite(fv):
        _raise(task_name, f"{field_name} must be finite.")
    if fv <= 0:
        _raise(task_name, f"{field_name} must be > 0.")


def _validate_ids(task_name: str, field_name: str, values: Any, *, allow_empty: bool = True) -> None:
    if values is None:
        return
    if not _is_sequence(values):
        _raise(task_name, f"{field_name} must be a sequence of integers.")
    vals = list(values)
    if not allow_empty and not vals:
        _raise(task_name, f"{field_name} cannot be empty.")
    for v in vals:
        try:
            int(v)
        except Exception:
            _raise(task_name, f"{field_name} contains non-integer value '{v}'.")


def _validate_str_list(task_name: str, field_name: str, values: Any, *, allow_empty: bool = True) -> None:
    if values is None:
        return
    if not _is_sequence(values):
        _raise(task_name, f"{field_name} must be a sequence.")
    vals = [str(v) for v in values]
    if not allow_empty and not vals:
        _raise(task_name, f"{field_name} cannot be empty.")


def _validate_msd(task_name: str, data: Any, request: Any) -> None:
    if not isinstance(data, TrajectoryData):
        return
    dims = tuple(str(d).lower() for d in getattr(request, "dims", ()) if str(d).lower() in {"x", "y", "z"})
    if not dims:
        _raise(task_name, "dims must include at least one of x, y, z.")
    _validate_ids(task_name, "atom_ids", getattr(request, "atom_ids", None))
    _validate_str_list(task_name, "atom_types", getattr(request, "atom_types", None))
    origin = getattr(request, "origin", "first")
    if not (origin == "first" or isinstance(origin, (int, np.integer))):
        _raise(task_name, "origin must be 'first' or an integer frame index.")


def _trajectory_selection_indices(data: TrajectoryData, request: Any) -> tuple[np.ndarray, np.ndarray]:
    positions = np.asarray(data.positions, dtype=float)
    n_frames, n_atoms = positions.shape[:2]

    frames = getattr(request, "frames", None)
    every = max(1, int(getattr(request, "every", 1)))
    frame_idx = list(range(n_frames)) if frames is None else [int(i) for i in frames if 0 <= int(i) < n_frames]
    frame_idx = frame_idx[::every]

    atom_ids = getattr(request, "atom_ids", None)
    atom_types = getattr(request, "atom_types", None)
    atom_ids_a = getattr(request, "atom_ids_a", None)
    atom_ids_b = getattr(request, "atom_ids_b", None)
    atom_types_a = getattr(request, "atom_types_a", None)
    atom_types_b = getattr(request, "atom_types_b", None)

    if atom_ids is not None:
        chosen = {int(v) for v in atom_ids}
        atom_idx = np.array([i for i, aid in enumerate(data.atom_ids) if int(aid) in chosen], dtype=int)
    elif atom_types is not None:
        chosen = {str(v) for v in atom_types}
        atom_idx = np.array([i for i, elem in enumerate(data.elements) if str(elem) in chosen], dtype=int)
    elif atom_ids_a is not None or atom_ids_b is not None or atom_types_a is not None or atom_types_b is not None:
        chosen_idx = set(range(n_atoms))
        if atom_ids_a is not None or atom_ids_b is not None:
            ids = set(int(v) for v in (list(atom_ids_a or []) + list(atom_ids_b or [])))
            chosen_idx = {i for i, aid in enumerate(data.atom_ids) if int(aid) in ids}
        if atom_types_a is not None or atom_types_b is not None:
            types = set(str(v) for v in (list(atom_types_a or []) + list(atom_types_b or [])))
            chosen_idx = {i for i, elem in enumerate(data.elements) if str(elem) in types}
        atom_idx = np.array(sorted(chosen_idx), dtype=int)
    else:
        atom_idx = np.arange(n_atoms, dtype=int)

    return np.asarray(frame_idx, dtype=int), atom_idx


def _validate_trajectory_selection_finite(task_name: str, data: TrajectoryData, request: Any) -> None:
    frame_idx, atom_idx = _trajectory_selection_indices(data, request)
    if frame_idx.size == 0 or atom_idx.size == 0:
        return
    coords = np.asarray(data.positions, dtype=float)[frame_idx][:, atom_idx, :]
    finite_mask = np.isfinite(coords).all(axis=2)
    if not finite_mask.all():
        bad = np.argwhere(~finite_mask)
        first = bad[0]
        fi = int(frame_idx[first[0]])
        ai = int(atom_idx[first[1]])
        atom_id = int(data.atom_ids[ai]) if ai < len(data.atom_ids) else ai + 1
        _raise(
            task_name,
            f"selected coordinates contain NaN/inf at frame_index={fi}, atom_id={atom_id}; "
            "filter atom_ids/atom_types or prefilter padded atoms.",
        )


def _validate_trajectory_like(task_name: str, request: Any) -> None:
    _validate_frames(task_name, getattr(request, "frames", None))
    _validate_every(task_name, getattr(request, "every", None))
    _validate_ids(task_name, "atom_ids", getattr(request, "atom_ids", None))
    _validate_ids(task_name, "atom_ids_a", getattr(request, "atom_ids_a", None))
    _validate_ids(task_name, "atom_ids_b", getattr(request, "atom_ids_b", None))
    _validate_str_list(task_name, "atom_types", getattr(request, "atom_types", None))
    _validate_str_list(task_name, "atom_types_a", getattr(request, "atom_types_a", None))
    _validate_str_list(task_name, "atom_types_b", getattr(request, "atom_types_b", None))


def _validate_task_specific(task: Any, data: Any, request: Any) -> None:
    task_name = type(task).__name__

    if hasattr(request, "frames"):
        _validate_frames(task_name, request.frames)
    if hasattr(request, "every"):
        _validate_every(task_name, request.every)

    if task_name == "MSDTask":
        _validate_msd(task_name, data, request)
        _validate_trajectory_selection_finite(task_name, data, request)
        return

    if task_name in {"RDFTask", "RDFPropertyTask"}:
        _validate_trajectory_like(task_name, request)
        _validate_numeric_positive(task_name, "bins", getattr(request, "bins", 0), allow_none=False)
        r_max = getattr(request, "r_max", None)
        if r_max is not None:
            _validate_numeric_positive(task_name, "r_max", r_max, allow_none=False)
        if isinstance(data, TrajectoryData):
            _validate_trajectory_selection_finite(task_name, data, request)
        return

    if task_name in {"TrajectoryCoordinateSeriesTask"} and isinstance(data, TrajectoryData):
        _validate_trajectory_like(task_name, request)
        _validate_trajectory_selection_finite(task_name, data, request)
        return

    if task_name in {"ChargeSeriesTask"}:
        _validate_ids(task_name, "atom_ids", getattr(request, "atom_ids", None), allow_empty=False)
        return

    if task_name in {"ElectricFieldSeriesTask"}:
        _validate_str_list(task_name, "components", getattr(request, "components", None), allow_empty=False)
        return

    if task_name in {"MolecularFrequencySeriesTask"}:
        _validate_str_list(task_name, "molecules", getattr(request, "molecules", None), allow_empty=False)
        return

    if task_name in {"MolecularTotalsSeriesTask"}:
        _validate_str_list(task_name, "quantities", getattr(request, "quantities", None), allow_empty=False)
        return

    if task_name == "ControlParametersTask":
        if not str(getattr(request, "key", "")).strip():
            _raise(task_name, "key cannot be empty.")
        return

    if task_name == "ForceFieldOptimizationParameterTask":
        return

    if task_name == "TrajectoryRelabelByCoordinationTask":
        mode = str(getattr(request, "mode", ""))
        if mode not in {"global", "by_type"}:
            _raise(task_name, "mode must be 'global' or 'by_type'.")
        return


def validate_task_inputs(task: Any, data: Any, request: Any) -> None:
    """Validate task data/request before entering task logic."""
    task_name = type(task).__name__

    required = getattr(task, "required_data", None)
    if required is not None and not isinstance(data, required):
        _raise(task_name, f"expected data type {required.__name__}, got {type(data).__name__}.")

    if not isinstance(request, BaseRequest):
        _raise(task_name, f"expected request type BaseRequest, got {type(request).__name__}.")

    data_validate = getattr(data, "validate", None)
    if callable(data_validate):
        try:
            data_validate()
        except Exception as exc:
            _raise(task_name, f"invalid data model: {exc}")

    _validate_task_specific(task, data, request)
