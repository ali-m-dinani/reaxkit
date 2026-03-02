"""Atomic-kinematics analysis tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Sequence

import pandas as pd

from reaxkit.analysis.base import AnalysisTask
from reaxkit.core.task_registry import register_task
from reaxkit.domain.base_request import BaseRequest
from reaxkit.domain.base_result import BaseResult
from reaxkit.domain.data_models import AtomicKinematicsData

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
) -> tuple[pd.DataFrame | None, dict | None]:
    if key == "metadata":
        meta = dict(data.metadata or {})
        if data.lattice_parameters is not None:
            meta["lattice_parameters"] = data.lattice_parameters
        if data.md_temperature_K is not None:
            meta["md_temperature_K"] = data.md_temperature_K
        return None, meta

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
    return df.reset_index(drop=True), None


@dataclass
class AtomicKinematicsRequest(BaseRequest):
    """Request for selected atomic kinematics data."""

    key: KinematicsKey
    atoms: Optional[Sequence[int]] = None


@dataclass
class AtomicKinematicsResult(BaseResult):
    """Result for selected atomic kinematics data."""

    table: Optional[pd.DataFrame] = None
    metadata: Optional[dict] = None


@register_task("atomic_kinematics")
class AtomicKinematicsTask(AnalysisTask):
    """Return metadata or a selected atomic-kinematics table from vels-style files."""

    required_data = AtomicKinematicsData

    def run(
        self,
        data: AtomicKinematicsData,
        request: AtomicKinematicsRequest,
        reporter=None,
    ) -> AtomicKinematicsResult:
        table, metadata = _get_vels_data(
            data,
            key=request.key,
            atoms=request.atoms,
        )
        return AtomicKinematicsResult(table=table, metadata=metadata)


__all__ = [
    "AtomicKinematicsRequest",
    "AtomicKinematicsResult",
    "AtomicKinematicsTask",
]
