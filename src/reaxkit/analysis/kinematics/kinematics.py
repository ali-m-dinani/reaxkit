"""Provide analyzer tasks for atomic kinematics series extraction.

This module slices and normalizes coordinate, velocity, and acceleration
streams from atomic kinematics data into tabular analyzer outputs. It is scoped
to kinematics-domain series extraction and does not infer bonding or chemistry.

**Usage context**

- Motion analysis: Inspect per-atom position/velocity/acceleration signals.
- Frame selection: Extract kinematics series on selected frame subsets.
- Diagnostics: Export normalized time-series tables for downstream plotting.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from typing import Any, Literal, Optional, Sequence

import pandas as pd

from reaxkit.analysis.base import AnalysisTask
from reaxkit.core.registry.analysis_task_registry import register_task
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
    """Extract one selected kinematics table (or metadata key/value rows)."""
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
    """Request payload for atomic kinematics table extraction.

    This request selects one kinematics dataset and optionally restricts output
    rows to a subset of atom indices.

    Fields
    -----
    key : KinematicsKey
        Kinematics dataset selector: ``metadata``, ``coordinates``,
        ``velocities``, ``accelerations``, or ``prev_accelerations``.
    atoms : Optional[Sequence[int]]
        Optional list/sequence of atom indices to keep. If omitted, all atoms
        from the selected dataset are returned.

    Examples
    -----
    ```python
    request = AtomicKinematicsRequest(key="velocities", atoms=[1, 3, 7])
    ```
    The request returns velocity rows only for atoms 1, 3, and 7.
    """

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
    """Result payload for atomic kinematics extraction.

    The analyzer returns a normalized table corresponding to the selected
    kinematics key and optional atom filtering constraints.

    Fields
    -----
    table : pandas.DataFrame
        Extracted table for the selected dataset key. ``metadata`` returns
        ``key``/``value`` rows; kinematics arrays return per-atom numeric rows.
    request : AtomicKinematicsRequest
        Request object used to generate this result.

    Examples
    -----
    ```python
    row = {"atom_index": 1, "vx": -0.24, "vy": 0.03, "vz": 0.11}
    ```
    The sample row represents one velocity record for one atom.
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
        """Recommend table and simple axis plot views for kinematics outputs.

        Always returns a table view and adds a single-series plot when
        ``atom_index`` and at least one numeric component column are present.

        Works on
        Analyzer task output for ``get_kinematics``.

        Parameters
        -----
        _result : AtomicKinematicsResult
            Typed analyzer result instance (unused by current logic).
        payload : dict[str, Any]
            Serialized payload expected to include a ``table`` key.

        Returns
        -----
        list[PresentationSpec]
            Recommended renderer specifications for UI display.

        Examples
        -----
        ```python
        specs = AtomicKinematicsTask.recommended_presentations(
            _result,
            {"table": [{"atom_index": 1, "vx": -0.24}]},
        )
        ```
        The returned list includes a table and one component-vs-atom plot.
        """
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
        """Execute atomic kinematics extraction for the selected data key.

        Selects one kinematics table from parsed atomic kinematics data and
        applies optional atom-index filtering before returning a typed result.

        Works on
        ``AtomicKinematicsData`` parsed from kinematics/vels-style sources.

        Parameters
        -----
        data : AtomicKinematicsData
            Parsed kinematics data bundle.
        request : AtomicKinematicsRequest
            Request with dataset key and optional atom filter.
        reporter : Any, optional
            Progress callback accepted by analyzer tasks; unused here.

        Returns
        -----
        AtomicKinematicsResult
            Result containing the extracted kinematics table.

        Examples
        -----
        ```python
        result = AtomicKinematicsTask().run(
            data,
            AtomicKinematicsRequest(key="coordinates"),
        )
        ```
        ``result.table`` contains the selected coordinate rows.
        """
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
