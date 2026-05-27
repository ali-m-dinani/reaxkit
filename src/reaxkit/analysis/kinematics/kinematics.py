"""Atomic-kinematics analysis tasks."""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from typing import Any, Literal, Optional, Sequence

import pandas as pd

from reaxkit.analysis.base import AnalysisTask
from reaxkit.core.analysis_task_registry import register_task
from reaxkit.domain.base_request import BaseRequest
from reaxkit.domain.base_result import BaseResult
from reaxkit.domain.data_models import AtomicKinematicsData
from reaxkit.presentation.specs import PresentationSpec

KinematicsKey = Literal[
    "metadata",
    "coordinates",
    "velocities",
    "accelerations",
    "prev_accelerations",
]


def _get_vels_data(
    data: AtomicKinematicsData,
    key: KinematicsKey,
    *,
    atoms: Sequence[int] | None = None,
) -> pd.DataFrame:
    if key == "metadata":
        meta = dict(data.metadata or {})
        if data.lattice_parameters is not None:
            meta["lattice_parameters"] = data.lattice_parameters
        if data.md_temperature_K is not None:
            meta["md_temperature_K"] = data.md_temperature_K
        rows = [{"key": str(k), "value": v} for k, v in meta.items()]
        return pd.DataFrame(rows, columns=["key", "value"])

    if key == "coordinates":
        df = data.coordinates.copy()
    elif key == "velocities":
        df = data.velocities.copy()
    elif key == "accelerations":
        df = data.accelerations.copy()
    elif key == "prev_accelerations":
        df = data.previous_accelerations.copy()
    else:
        raise ValueError(f"Unknown vels key: {key}")

    if atoms:
        df = df[df["atom_index"].isin(list(atoms))].copy()
    return df.reset_index(drop=True)


@dataclass
class AtomicKinematicsRequest(BaseRequest):
    """Request for selected atomic kinematics data."""

    key: KinematicsKey = dc_field(
        metadata={
            "label": "Key",
            "help": "Kinematics dataset to extract. Example: key='velocities' returns vx, vy, vz columns.",
            "choices": ["metadata", "coordinates", "velocities", "accelerations", "prev_accelerations"],
        },
    )
    atoms: Optional[Sequence[int]] = dc_field(
        default=None,
        metadata={
            "label": "Atoms",
            "help": "Optional 1-based atom ids to filter rows. Example: atoms=[1, 3, 7]. Empty means all atoms.",
        },
    )


@dataclass
class AtomicKinematicsResult(BaseResult):
    """Atomic-kinematics extraction result.

    Output structure:
    - request: AtomicKinematicsRequest used to generate this result
    - table: pandas.DataFrame
      - key='metadata': columns ['key', 'value'] with one metadata entry per row
      - key='coordinates': table with atom positions (for example atom_index, x, y, z)
      - key='velocities': table with per-atom velocity components (vx, vy, vz)
      - key='accelerations': table with per-atom acceleration components (ax, ay, az)
      - key='prev_accelerations': table with previous-step acceleration components

    Example:
    request.key='velocities', atoms=[1,2] returns only rows for atom_index 1 and 2
    and includes velocity component columns for those atoms.
    """

    table: pd.DataFrame
    request: AtomicKinematicsRequest


@register_task("get_kinematics", label="Atomic Kinematics")
class AtomicKinematicsTask(AnalysisTask):
    """Return metadata or a selected atomic-kinematics table from vels-style files."""

    required_data = AtomicKinematicsData

    @staticmethod
    def recommended_presentations(
        _result: AtomicKinematicsResult, payload: dict[str, Any]
    ) -> list[PresentationSpec]:
        rows = payload.get("table")
        if not isinstance(rows, list) or not rows:
            return [PresentationSpec(renderer="table", label="Table", view_type="table")]
        sample = rows[0] if isinstance(rows[0], dict) else {}
        if "atom_index" not in sample:
            return [PresentationSpec(renderer="table", label="Table", view_type="table")]
        y_col = ""
        for candidate in ("vx", "vy", "vz", "ax", "ay", "az", "x", "y", "z"):
            if candidate in sample:
                y_col = candidate
                break
        if not y_col:
            return [PresentationSpec(renderer="table", label="Table", view_type="table")]
        return [
            PresentationSpec(renderer="table", label="Table", view_type="table"),
            PresentationSpec(
                renderer="single_plot",
                label=f"{y_col} vs atom_index",
                mapping={"x_col": "atom_index", "y_col": y_col, "group_by_col": ""},
                options={
                    "title": f"{y_col} vs atom_index",
                    "xlabel": "atom_index",
                    "ylabel": y_col,
                    "legend": False,
                },
                view_type="plot2d",
            ),
        ]

    def run(
        self,
        data: AtomicKinematicsData,
        request: AtomicKinematicsRequest,
        reporter=None,
    ) -> AtomicKinematicsResult:
        table = _get_vels_data(
            data,
            key=request.key,
            atoms=request.atoms,
        )
        return AtomicKinematicsResult(table=table, request=request)


__all__ = [
    "AtomicKinematicsRequest",
    "AtomicKinematicsResult",
    "AtomicKinematicsTask",
]
