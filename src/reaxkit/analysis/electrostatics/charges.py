"""Charge extraction tasks built on top of ``ChargeData``."""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd

from reaxkit.analysis.base import AnalysisTask
from reaxkit.core.analysis_task_registry import register_task
from reaxkit.domain.base_request import BaseRequest
from reaxkit.domain.base_result import BaseResult
from reaxkit.domain.data_models import ChargeData
from reaxkit.presentation.specs import PresentationSpec


def _frame_indices(n_frames: int, frames: Optional[Sequence[int]], every: int) -> list[int]:
    idx = list(range(n_frames)) if frames is None else [int(i) for i in frames]
    return [i for i in idx if 0 <= i < n_frames][:: max(1, int(every))]


@dataclass
class ChargeTableRequest(BaseRequest):
    """Request for per-atom charge extraction across frames.

    Parameters
    ----------
    atom_ids
        Optional atom-id filter. If provided, only these atoms are included.
        Example: ``[1, 5, 9]``.
    atom_types
        Optional element/type filter. Used only when ``atom_ids`` is not set.
        Example: ``["O", "H"]``.
    frames
        Optional frame indices to include. If omitted, all frames are included.
        Example: ``[0, 10, 20]``.
    every
        Frame stride after selection. Example: ``every=5`` keeps every fifth
        selected frame.
    """

    atom_ids: Optional[Sequence[int]] = dc_field(
        default=None,
        metadata={
            'label': 'Atom Ids',
            'help': "Optional atom-id filter. Example: [1, 5, 9].",
            'units': 'index',
        },
    )
    atom_types: Optional[Sequence[str]] = dc_field(
        default=None,
        metadata={
            'label': 'Atom Types',
            'help': "Optional atom-type filter used when atom_ids is not provided. Example: ['O', 'H'].",
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
class ChargeTableResult(BaseResult):
    """Charge-table extraction result.

    Output structure
    ----------------
    - ``request``: the :class:`ChargeTableRequest` used to generate this result.
    - ``table``: pandas.DataFrame with columns:
      ``['frame_index', 'iter', 'atom_id', 'atom_type', 'charge']``.

    Column meanings
    ---------------
    - ``frame_index``: source frame index in the charge trajectory.
    - ``iter``: simulation iteration mapped to the frame.
    - ``atom_id``: atom identifier.
    - ``atom_type``: atom element/type label when available.
    - ``charge``: per-atom charge value at that frame.

    Example
    -------
    A row ``frame_index=12, iter=1200, atom_id=7, atom_type='O', charge=-0.52``
    means atom 7 had charge -0.52 at frame 12.
    """

    table: pd.DataFrame
    request: ChargeTableRequest


@register_task("charge_table", label="Charge Table")
class ChargeTableTask(AnalysisTask):
    """Return per-atom charges across selected frames as a tidy table."""

    required_data = ChargeData

    @staticmethod
    def recommended_presentations(
        _result: ChargeTableResult,
        payload: dict[str, Any],
    ) -> list[PresentationSpec]:
        rows = payload.get("table")
        if not isinstance(rows, list) or not rows:
            return [PresentationSpec(renderer="table", label="Table", view_type="table")]
        sample = rows[0] if isinstance(rows[0], dict) else {}
        if "iter" not in sample or "charge" not in sample:
            return [PresentationSpec(renderer="table", label="Table", view_type="table")]
        return [
            PresentationSpec(renderer="table", label="Table", view_type="table"),
            PresentationSpec(
                renderer="single_plot",
                label="charge vs iter",
                mapping={"x_col": "iter", "y_col": "charge", "group_by_col": "atom_id"},
                options={
                    "title": "Charge Table",
                    "xlabel": "iter",
                    "ylabel": "charge",
                    "legend": True,
                },
                view_type="plot2d",
            ),
        ]

    def run(self, data: ChargeData, request: ChargeTableRequest, reporter=None) -> ChargeTableResult:
        charges = np.asarray(data.charges, dtype=float)
        if charges.ndim != 2:
            raise ValueError("ChargeData.charges must have shape (n_frames, n_atoms).")

        n_frames, n_atoms = charges.shape
        frame_idx = _frame_indices(n_frames, request.frames, request.every)
        if not frame_idx:
            return ChargeTableResult(
                table=pd.DataFrame(columns=["frame_index", "iter", "atom_id", "atom_type", "charge"]),
                request=request,
            )

        iterations = (
            np.asarray(data.iterations, dtype=int).reshape(-1)
            if data.iterations is not None
            else np.arange(n_frames, dtype=int)
        )
        if iterations.shape[0] != n_frames:
            raise ValueError("ChargeData.iterations length must match number of frames.")

        atom_ids = None
        if data.simulation is not None and data.simulation.atom_ids is not None:
            atom_ids = [int(a) for a in data.simulation.atom_ids]
        if atom_ids is None:
            atom_ids = list(range(1, n_atoms + 1))
        if len(atom_ids) != n_atoms:
            raise ValueError("ChargeData atom_ids length must match number of atoms.")

        elements: list[str | None]
        if data.simulation is not None and data.simulation.elements is not None:
            elements = [str(e) for e in data.simulation.elements]
            if len(elements) != n_atoms:
                raise ValueError("ChargeData simulation elements length must match number of atoms.")
        else:
            elements = [None] * n_atoms

        if request.atom_ids is not None:
            chosen_ids = {int(a) for a in request.atom_ids}
            atom_indices = [i for i, atom_id in enumerate(atom_ids) if atom_id in chosen_ids]
        elif request.atom_types:
            chosen_types = {str(t) for t in request.atom_types}
            atom_indices = [i for i, elem in enumerate(elements) if elem in chosen_types]
        else:
            atom_indices = list(range(n_atoms))

        rows: list[dict[str, object]] = []
        for fi in frame_idx:
            for atom_idx in atom_indices:
                rows.append(
                    {
                        "frame_index": int(fi),
                        "iter": int(iterations[fi]),
                        "atom_id": int(atom_ids[atom_idx]),
                        "atom_type": elements[atom_idx],
                        "charge": float(charges[fi, atom_idx]),
                    }
                )

        table = pd.DataFrame(rows)
        if table.empty:
            table = pd.DataFrame(columns=["frame_index", "iter", "atom_id", "atom_type", "charge"])
        else:
            table = table.sort_values(["frame_index", "atom_id"], kind="stable").reset_index(drop=True)
        return ChargeTableResult(table=table, request=request)


__all__ = [
    "ChargeTableRequest",
    "ChargeTableResult",
    "ChargeTableTask",
]
