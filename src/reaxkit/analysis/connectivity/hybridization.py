"""Assign per-atom hybridization status from connectivity-derived metrics.

This module classifies atoms into configured hybridization states using
bond-order-sum targets and tolerances, optionally with element-specific rules.
It focuses on hybridization labels and summary tables rather than coordination
or event-detection logic.

**Usage context**

- Hybridization tagging: Label atoms as `sp`, `sp2`, `sp3`, or custom states.
- Force-field validation: Compare expected hybridization against trajectories.
- Chemistry post-processing: Feed state labels into reporting or filtering flows.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from reaxkit.analysis.base import AnalysisTask
from reaxkit.core.registry.analysis_task_registry import register_task
from reaxkit.domain.base_request import BaseRequest
from reaxkit.domain.base_result import BaseResult
from reaxkit.domain.data_models import ConnectivityData
from reaxkit.presentation.specs import PresentationSpec


@dataclass
class HybridizationStatusRequest(BaseRequest):
    """Request for per-atom hybridization-status classification.

    Fields
    -----
    hybridizations : Optional[Mapping[str, float]]
        Global hybridization targets used when element-specific mappings are not
        provided. Keys are state labels and values are target BO sums.
    element_hybridizations : Optional[Mapping[str, Mapping[str, float]]]
        Per-element hybridization map overriding global targets.
    target_elements : Optional[Sequence[str]]
        Optional element filter; only matching atom types are classified.
    target_atom_ids : Optional[Sequence[int]]
        Optional atom-id filter (1-based IDs).
    threshold : float
        Absolute tolerance used to classify closest target as matched/unmatched.
    frames : Optional[Sequence[int]]
        Optional frame indices to evaluate. `None` means all frames.
    every : int
        Frame stride after selection. Must be `>= 1`.
    require_defined_hybridization : bool
        If `True`, missing element mappings raise; otherwise rows are emitted as
        `status_label="undefined"`.

    Examples
    -----
    ```python
    req = HybridizationStatusRequest(
        hybridizations={"sp": 1.0, "sp2": 2.0, "sp3": 3.0},
        target_elements=["C"],
        threshold=0.2,
    )
    ```
    Sample output:
    `HybridizationStatusRequest(...)`
    Meaning:
    The request configures atom selection and matching tolerance for
    hybridization classification.
    """

    hybridizations: Optional[Mapping[str, float]] = dc_field(
        default=None,
        metadata={
            'label': 'Hybridizations',
            'help': (
                "Global hybridization mapping as state->target sum_BOs. "
                "Example: {'sp': 1, 'sp2': 2, 'sp3': 3}."
            ),
        },
    )
    element_hybridizations: Optional[Mapping[str, Mapping[str, float]]] = dc_field(
        default=None,
        metadata={
            'label': 'Element Hybridizations',
            'help': (
                "Per-element hybridization map overriding global values. "
                "Example: {'C': {'sp': 1, 'sp2': 2, 'sp3': 3}}."
            ),
        },
    )
    target_elements: Optional[Sequence[str]] = dc_field(
        default=None,
        metadata={
            'label': 'Target Elements',
            'help': "Optional element filter. Example: ['C', 'O'].",
        },
    )
    target_atom_ids: Optional[Sequence[int]] = dc_field(
        default=None,
        metadata={
            'label': 'Target Atom Ids',
            'help': "Optional atom-id filter (1-based). Example: [1, 2, 5].",
        },
    )
    threshold: float = dc_field(
        default=0.3,
        metadata={
            'label': 'Threshold',
            'help': "Absolute tolerance for match classification. Example: 0.2.",
            'min': 0.0,
        },
    )
    frames: Optional[Sequence[int]] = dc_field(
        default=None,
        metadata={
            'label': 'Frames',
            'help': "Optional frame indices to evaluate. Example: [0, 10, 20].",
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
    require_defined_hybridization: bool = dc_field(
        default=True,
        metadata={
            'label': 'Require Defined Hybridization',
            'help': (
                "If true, fail when an element has no mapping. If false, emit "
                "undefined rows instead."
            ),
            'choices': [True, False],
        },
    )


@dataclass
class HybridizationStatusResult(BaseResult):
    """Hybridization-status analysis result.

    Fields
    -----
    table : pd.DataFrame
        Output table with one row per selected atom per analyzed frame.
        Typical columns include `frame_index`, `iter`, `atom_id`, `atom_type`,
        `sum_BOs`, `hybridization`, `expected_sum_BOs`, `delta`,
        `within_threshold`, and `status_label`.
    request : HybridizationStatusRequest
        Request object used for this analysis run.

    Examples
    -----
    ```python
    result = HybridizationStatusTask().run(data, req)
    result.table.head()
    ```
    Sample output:
    DataFrame rows with matched/unmatched hybridization assignments.
    Meaning:
    Each row captures one atom-frame classification decision.
    """

    table: pd.DataFrame
    request: HybridizationStatusRequest


def _sum_bond_orders_matrix(data: ConnectivityData) -> np.ndarray:
    if data.sum_bond_orders is not None:
        arr = np.asarray(data.sum_bond_orders, dtype=float)
        if arr.ndim != 2:
            raise ValueError("ConnectivityData.sum_bond_orders must have shape (n_frames, n_atoms).")
        return arr

    bo = data.bond_orders
    if bo is None:
        raise ValueError("ConnectivityData requires either sum_bond_orders or bond_orders.")

    if isinstance(bo, np.ndarray):
        if bo.ndim != 3:
            raise ValueError("ConnectivityData.bond_orders ndarray must have shape (n_frames, n_atoms, n_atoms).")
        return np.sum(bo.astype(float), axis=2)

    if isinstance(bo, (list, tuple)):
        frames = []
        for fr in bo:
            if hasattr(fr, "sum"):
                s = np.asarray(fr.sum(axis=1)).reshape(-1).astype(float)
            else:
                s = np.sum(np.asarray(fr, dtype=float), axis=1)
            frames.append(s)
        if not frames:
            return np.empty((0, 0), dtype=float)
        return np.vstack(frames)

    raise TypeError("Unsupported bond_orders type; use ndarray, sparse-per-frame list, or provide sum_bond_orders.")


def _hyb_map_for_element(elem: str, req: HybridizationStatusRequest) -> Optional[dict[str, float]]:
    out: dict[str, float] = {}
    if req.element_hybridizations and elem in req.element_hybridizations:
        out = {str(k): float(v) for k, v in req.element_hybridizations[elem].items()}
    elif req.hybridizations:
        out = {str(k): float(v) for k, v in req.hybridizations.items()}
    return out or None


def _selected_atom_indices(
    *,
    atom_ids: np.ndarray,
    elements: list[str],
    target_atom_ids: Optional[Sequence[int]],
    target_elements: Optional[Sequence[str]],
) -> list[int]:
    idx = list(range(len(atom_ids)))
    if target_atom_ids is not None:
        wanted_ids = {int(a) for a in target_atom_ids}
        idx = [i for i in idx if int(atom_ids[i]) in wanted_ids]
    if target_elements is not None:
        wanted_elems = {str(e) for e in target_elements}
        idx = [i for i in idx if str(elements[i]) in wanted_elems]
    return idx


@register_task("get_hybridization", label="Hybridization")
class HybridizationStatusTask(AnalysisTask):
    """Per-atom hybridization status over selected frames."""

    required_data = ConnectivityData

    @staticmethod
    def recommended_presentations(
        _result: HybridizationStatusResult,
        _payload: dict[str, Any],
    ) -> list[PresentationSpec]:
        """Return default table presentation for hybridization outputs.

        Works on
        -----
        Analyzer task output payloads

        Parameters
        -----
        _result : HybridizationStatusResult
            Analysis result object for the executed task.
        _payload : dict[str, Any]
            Serialized result payload.

        Returns
        -----
        list[PresentationSpec]
            Table presentation specification.

        Examples
        -----
        ```python
        specs = HybridizationStatusTask.recommended_presentations(result, payload)
        ```
        Sample output:
        `[PresentationSpec(renderer="table", ...)]`
        Meaning:
        Hybridization results default to tabular rendering.
        """
        return [PresentationSpec(renderer="table", label="Table", view_type="table")]

    def run(
        self,
        data: ConnectivityData,
        request: HybridizationStatusRequest,
        reporter=None,
    ) -> HybridizationStatusResult:
        """Classify per-atom hybridization status across selected frames.

        Works on
        -----
        `ConnectivityData` plus `HybridizationStatusRequest` analyzer inputs

        Parameters
        -----
        data : ConnectivityData
            Connectivity input containing sum bond orders or bond-order matrices.
        request : HybridizationStatusRequest
            Selection and hybridization-target configuration.
        reporter : Any, optional
            Optional progress callback invoked during frame processing.

        Returns
        -----
        HybridizationStatusResult
            Result table with matched/unmatched hybridization assignments.

        Examples
        -----
        ```python
        result = HybridizationStatusTask().run(data, req)
        ```
        Sample output:
        `result.table` with `status_label` and `hybridization` columns.
        Meaning:
        One row is produced per selected atom per selected frame.
        """
        sum_bos_m = _sum_bond_orders_matrix(data)
        if sum_bos_m.size == 0:
            return HybridizationStatusResult(table=pd.DataFrame(), request=request)

        n_frames, n_atoms = sum_bos_m.shape
        elements = data.elements if data.elements is not None else ["X"] * n_atoms
        if len(elements) != n_atoms:
            raise ValueError(f"elements length ({len(elements)}) must match n_atoms ({n_atoms}).")

        if data.atom_ids is not None:
            atom_ids = np.asarray(data.atom_ids, dtype=int).reshape(-1)
            if atom_ids.shape[0] != n_atoms:
                raise ValueError(f"atom_ids length ({atom_ids.shape[0]}) must match n_atoms ({n_atoms}).")
        else:
            atom_ids = np.arange(1, n_atoms + 1, dtype=int)

        sel_idx = _selected_atom_indices(
            atom_ids=atom_ids,
            elements=elements,
            target_atom_ids=request.target_atom_ids,
            target_elements=request.target_elements,
        )
        if not sel_idx:
            return HybridizationStatusResult(table=pd.DataFrame(), request=request)

        frame_idx = list(range(n_frames)) if request.frames is None else [int(i) for i in request.frames]
        frame_idx = [i for i in frame_idx if 0 <= i < n_frames][:: max(1, int(request.every))]
        if not frame_idx:
            return HybridizationStatusResult(table=pd.DataFrame(), request=request)

        rows: list[dict] = []
        total = max(1, len(frame_idx))
        if reporter:
            reporter("analyze", 0, total, "Preparing hybridization status")
        for step_i, fi in enumerate(frame_idx, start=1):
            iter_val = int(data.iterations[fi]) if data.iterations is not None else int(fi)
            for ai in sel_idx:
                elem = str(elements[ai])
                hyb_map = _hyb_map_for_element(elem, request)
                sum_bo = float(sum_bos_m[fi, ai])

                if not hyb_map:
                    if request.require_defined_hybridization:
                        raise KeyError(f"Missing hybridization mapping for element '{elem}'.")
                    rows.append(
                        {
                            "frame_index": int(fi),
                            "iter": iter_val,
                            "atom_id": int(atom_ids[ai]),
                            "atom_type": elem,
                            "sum_BOs": sum_bo,
                            "hybridization": None,
                            "expected_sum_BOs": np.nan,
                            "delta": np.nan,
                            "within_threshold": False,
                            "status_label": "undefined",
                        }
                    )
                    continue

                best_state = None
                best_target = np.nan
                best_abs = np.inf
                for state, target in hyb_map.items():
                    diff_abs = abs(sum_bo - float(target))
                    if diff_abs < best_abs:
                        best_abs = diff_abs
                        best_state = str(state)
                        best_target = float(target)

                delta = float(sum_bo - best_target)
                within = bool(abs(delta) <= float(request.threshold))
                rows.append(
                    {
                        "frame_index": int(fi),
                        "iter": iter_val,
                        "atom_id": int(atom_ids[ai]),
                        "atom_type": elem,
                        "sum_BOs": sum_bo,
                        "hybridization": best_state if within else None,
                        "expected_sum_BOs": best_target,
                        "delta": delta,
                        "within_threshold": within,
                        "status_label": "matched" if within else "unmatched",
                    }
                )
            if reporter:
                reporter("analyze", step_i, total, "Computing hybridization status")

        out = pd.DataFrame(rows).sort_values(["frame_index", "atom_id"], kind="mergesort").reset_index(drop=True)
        if reporter:
            reporter("analyze", total, total, "Finished hybridization status")
        return HybridizationStatusResult(table=out, request=request)


__all__ = [
    "HybridizationStatusRequest",
    "HybridizationStatusResult",
    "HybridizationStatusTask",
]
