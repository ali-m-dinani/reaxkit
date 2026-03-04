"""Charge extraction tasks built on top of ``ChargeData``."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from reaxkit.analysis.base import AnalysisTask
from reaxkit.core.analysis_task_registry import register_task
from reaxkit.domain.base_request import BaseRequest
from reaxkit.domain.base_result import BaseResult
from reaxkit.domain.data_models import ChargeData


def _frame_indices(n_frames: int, frames: Optional[Sequence[int]], every: int) -> list[int]:
    idx = list(range(n_frames)) if frames is None else [int(i) for i in frames]
    return [i for i in idx if 0 <= i < n_frames][:: max(1, int(every))]


@dataclass
class ChargeTableRequest(BaseRequest):
    atom_ids: Optional[Sequence[int]] = None
    atom_types: Optional[Sequence[str]] = None
    frames: Optional[Sequence[int]] = None
    every: int = 1


@dataclass
class ChargeTableResult(BaseResult):
    table: pd.DataFrame


@register_task("charge_table")
class ChargeTableTask(AnalysisTask):
    """Return per-atom charges across selected frames as a tidy table."""

    required_data = ChargeData

    def run(self, data: ChargeData, request: ChargeTableRequest, reporter=None) -> ChargeTableResult:
        charges = np.asarray(data.charges, dtype=float)
        if charges.ndim != 2:
            raise ValueError("ChargeData.charges must have shape (n_frames, n_atoms).")

        n_frames, n_atoms = charges.shape
        frame_idx = _frame_indices(n_frames, request.frames, request.every)
        if not frame_idx:
            return ChargeTableResult(
                table=pd.DataFrame(columns=["frame_index", "iter", "atom_id", "atom_type", "charge"])
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
        return ChargeTableResult(table=table)


__all__ = [
    "ChargeTableRequest",
    "ChargeTableResult",
    "ChargeTableTask",
]
