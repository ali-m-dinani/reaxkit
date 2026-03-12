"""Generic time-series analysis tasks."""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from typing import Any, Literal, Optional, Sequence, Union

import numpy as np
import pandas as pd

from reaxkit.analysis.base import AnalysisTask
from reaxkit.core.alias import normalize_choice, resolve_alias_from_columns
from reaxkit.core.analysis_task_registry import register_task
from reaxkit.domain.base_request import BaseRequest
from reaxkit.domain.base_result import BaseResult
from reaxkit.domain.data_models import (
    ChargeData,
    ElectricFieldData,
    EregimeData,
    MolecularAnalysisData,
    PartialEnergyData,
    RestraintData,
    SimulationData,
    TrajectoryData,
)
from reaxkit.presentation.specs import PresentationSpec


def _frame_indices(n_frames: int, frames: Optional[Sequence[int]], every: int) -> list[int]:
    idx = list(range(n_frames)) if frames is None else [int(i) for i in frames]
    return [i for i in idx if 0 <= i < n_frames][:: max(1, int(every))]


def _rows_from_payload(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = payload.get("table")
    if not isinstance(rows, list):
        return []
    return [dict(row) for row in rows if isinstance(row, dict)]


def _table_only_presentation() -> list[PresentationSpec]:
    return [PresentationSpec(renderer="table", label="Table", view_type="table")]


def _single_plot_presentation(
    *,
    x_col: str,
    y_col: str,
    label: str,
    group_col: str = "",
) -> PresentationSpec:
    return PresentationSpec(
        renderer="single_plot",
        label=label,
        mapping={"x_col": x_col, "y_col": y_col, "group_by_col": group_col},
        options={
            "title": label,
            "xlabel": x_col,
            "ylabel": y_col,
            "legend": bool(group_col),
        },
        view_type="plot2d",
    )


def _value_series_presentations(
    payload: dict[str, Any],
    *,
    y_col: str,
    group_col: str = "",
    label: str | None = None,
    x_candidates: Sequence[str] = ("iter", "frame_index"),
) -> list[PresentationSpec]:
    rows = _rows_from_payload(payload)
    if not rows:
        return _table_only_presentation()
    cols = {str(key) for key in rows[0].keys()}
    x_col = next((candidate for candidate in x_candidates if candidate in cols), "")
    if not x_col or y_col not in cols:
        return _table_only_presentation()
    return [
        PresentationSpec(renderer="table", label="Table", view_type="table"),
        _single_plot_presentation(
            x_col=x_col,
            y_col=y_col,
            group_col=group_col if group_col in cols else "",
            label=label or f"{y_col} vs {x_col}",
        ),
    ]


def _wide_series_presentations(
    payload: dict[str, Any],
    *,
    y_cols: Sequence[str],
    label_prefix: str = "",
    x_candidates: Sequence[str] = ("iter", "frame_index"),
) -> list[PresentationSpec]:
    rows = _rows_from_payload(payload)
    if not rows:
        return _table_only_presentation()
    cols = {str(key) for key in rows[0].keys()}
    x_col = next((candidate for candidate in x_candidates if candidate in cols), "")
    if not x_col:
        return _table_only_presentation()
    views: list[PresentationSpec] = [PresentationSpec(renderer="table", label="Table", view_type="table")]
    for y_col in y_cols:
        if y_col not in cols:
            continue
        label = f"{label_prefix}{y_col} vs {x_col}" if label_prefix else f"{y_col} vs {x_col}"
        views.append(_single_plot_presentation(x_col=x_col, y_col=y_col, label=label))
    return views or _table_only_presentation()


@dataclass
class Series:
    x: np.ndarray
    y: np.ndarray
    label: str


@dataclass
class TimeSeriesResult(BaseResult):
    series: list[Series]
    x_label: str
    y_label: str
    request: BaseRequest
    table: pd.DataFrame
    metadata: dict | None = None


@dataclass
class SimulationScalarSeriesResult(BaseResult):
    """Scalar simulation-series result.

    Output structure:
    - request: SimulationScalarSeriesRequest
      - the exact request used to generate this result
      - includes selected scalar field, frames, and stride
    - table: pandas.DataFrame with columns
      ['frame_index', 'iter', 'field', 'value']
      - frame_index: source frame index in the sampled series
      - iter: iteration corresponding to each sampled frame
      - field: resolved simulation scalar name
      - value: scalar value for that field at the sampled frame
    """

    request: SimulationScalarSeriesRequest
    table: pd.DataFrame


@dataclass
class TrajectoryCoordinateSeriesResult(BaseResult):
    """Trajectory coordinate-series result.

    Output structure:
    - request: TrajectoryCoordinateSeriesRequest
      - the exact request used to generate this result
      - includes selected atoms/types, dimensions, frames, and stride
    - table: pandas.DataFrame with columns
      ['frame_index', 'iter', 'atom_id', 'atom_type', 'dim', 'coord']
      - frame_index: source frame index in the trajectory
      - iter: iteration corresponding to the frame
      - atom_id: atom identifier
      - atom_type: element/type label for the atom
      - dim: coordinate dimension ('x', 'y', or 'z')
      - coord: coordinate value along the selected direction
    """

    request: TrajectoryCoordinateSeriesRequest
    table: pd.DataFrame


@dataclass
class CellDimensionsResult(BaseResult):
    """Cell-dimensions result.

    Output structure:
    - request: CellDimensionsRequest
      - the exact request used to generate this result
      - includes selected fields (a/b/c/alpha/beta/gamma), frames, and stride
    - table: pandas.DataFrame with columns
      ['frame_index', 'iter', 'field', 'value']
      - frame_index: source frame index in the sampled trajectory
      - iter: iteration corresponding to each sampled frame
      - field: selected cell field name
      - value: scalar value for that cell field at the sampled frame
    """

    request: CellDimensionsRequest
    table: pd.DataFrame


@dataclass
class ChargeSeriesResult(BaseResult):
    """Charge-series result.

    Output structure:
    - request: ChargeSeriesRequest
      - the exact request used to generate this result
      - includes selected atom IDs, frames, and stride
    - table: pandas.DataFrame with columns
      ['frame_index', 'iter', 'atom_id', 'atom_type', 'charge']
      - frame_index: source frame index in the sampled trajectory
      - iter: iteration corresponding to each sampled frame
      - atom_id: selected atom identifier
      - atom_type: element/type label for that atom when available
      - charge: charge value for the atom at the sampled frame
    """

    request: ChargeSeriesRequest
    table: pd.DataFrame


@dataclass
class ElectricFieldSeriesResult(BaseResult):
    """Electric-field series result.

    Output structure:
    - request: ElectricFieldSeriesRequest
      - the exact request used to generate this result
      - includes components, field selection mode, frames, and stride
    - table: pandas.DataFrame with columns
      ['frame_index', 'iter', 'component', 'value']
      - frame_index: source sample index
      - iter: iteration associated with the sampled field value
      - component: selected component name (for example Ex/Ey/Ez)
      - value: electric-field scalar for that component at the sampled index
    """

    request: ElectricFieldSeriesRequest
    table: pd.DataFrame


@dataclass
class EregimeSeriesResult(BaseResult):
    """Eregime-series result.

    Output structure:
    - request: EregimeSeriesRequest
      - the exact request used to generate this result
      - includes selected field, frames, and stride
    - table: pandas.DataFrame with columns
      ['frame_index', 'iter', 'field', 'value']
      - frame_index: source index in the eregime sequence
      - iter: iteration corresponding to each sampled row
      - field: resolved eregime field name used for extraction
      - value: sampled value for the selected field at each row
    """

    request: EregimeSeriesRequest
    table: pd.DataFrame


@dataclass
class PartialEnergySeriesResult(BaseResult):
    """Partial-energy series result.

    Output structure:
    - request: PartialEnergySeriesRequest
      - the exact request used to generate this result
      - includes selected energy components, frames, and stride
    - table: pandas.DataFrame with columns
      ['frame_index', 'iter', 'component', 'value']
      - frame_index: source frame index
      - iter: iteration corresponding to each sampled frame
      - component: selected partial-energy component name
      - value: sampled component value at the frame
    """

    request: PartialEnergySeriesRequest
    table: pd.DataFrame


@dataclass
class RestraintSeriesResult(BaseResult):
    """Restraint-series result.

    Output structure:
    - request: RestraintSeriesRequest
      - the exact request used to generate this result
      - includes selected fields or restraint index, frame selection, and options
    - table: pandas.DataFrame with columns
      ['frame_index', 'iter', 'restraint_index', 'field', 'value']
      - frame_index: source frame index in restraint data
      - iter: iteration corresponding to each sampled frame
      - restraint_index: restraint number extracted from field name (for example r1_target -> 1), or NA
      - field: selected restraint field name
      - value: sampled numeric value for that field/frame pair
    """

    request: RestraintSeriesRequest
    table: pd.DataFrame


@dataclass
class MolecularFrequencySeriesResult(BaseResult):
    """Molecular-frequency series result.

    Output structure:
    - request: MolecularFrequencySeriesRequest
      - the exact request used to generate this result
      - includes selected molecules, frames, and stride
    - table: pandas.DataFrame with columns
      ['frame_index', 'iter', 'molecules', 'freq', 'molecular_mass']
      - frame_index: source frame index in molecular-analysis data
      - iter: iteration corresponding to each sampled frame
      - molecules: requested molecular formula label
      - freq: sampled frequency value for that molecule at the frame
      - molecular_mass: sampled molecular mass value for that molecule at the frame
    """

    request: MolecularFrequencySeriesRequest
    table: pd.DataFrame


@dataclass
class MolecularTotalsSeriesResult(BaseResult):
    """Molecular-totals series result.

    Output structure:
    - request: MolecularTotalsSeriesRequest
      - the exact request used to generate this result
      - includes selected quantities, frames, and stride
    - table: pandas.DataFrame with columns
      ['frame_index', 'iter', 'quantity', 'value']
      - frame_index: source frame index in molecular totals data
      - iter: iteration corresponding to each sampled frame
      - quantity: selected quantity name (for example total_molecules)
      - value: sampled scalar value for that quantity at the frame
    """

    request: MolecularTotalsSeriesRequest
    table: pd.DataFrame


@dataclass
class SimulationScalarSeriesRequest(BaseRequest):
    field: str = dc_field(
        metadata={
            "label": "Field",
            "help": "Simulation scalar field to extract.",
            "choices": [
                "potential_energy",
                "volume",
                "temperature",
                "pressure",
                "density",
                "elapsed_time",
                "num_of_atoms",
                "a",
                "b",
                "c",
                "alpha",
                "beta",
                "gamma",
            ],
        },
    )
    frames: Optional[Sequence[int]] = dc_field(
        default=None,
        metadata={'label': 'Frames', 'help': 'Frames parameter for SimulationScalarSeriesRequest.', 'units': 'frame_index'},
    )
    every: int = dc_field(
        default=1,
        metadata={'label': 'Every', 'help': 'Every parameter for SimulationScalarSeriesRequest.', 'min': 1, 'units': 'frames'},
    )


@dataclass
class TrajectoryCoordinateSeriesRequest(BaseRequest):
    atom_ids: Optional[Sequence[int]] = dc_field(
        default=None,
        metadata={'label': 'Atom Ids', 'help': 'Atom Ids parameter for TrajectoryCoordinateSeriesRequest.', 'units': 'index'},
    )
    atom_types: Optional[Sequence[str]] = dc_field(
        default=None,
        metadata={'label': 'Atom Types', 'help': 'Atom Types parameter for TrajectoryCoordinateSeriesRequest.'},
    )
    dims: Sequence[str] = dc_field(
        default=("x",),
        metadata={
            "label": "Dims",
            "help": "Coordinate dimensions to include.",
            "choices": ["x", "y", "z"],
        },
    )
    frames: Optional[Sequence[int]] = dc_field(
        default=None,
        metadata={'label': 'Frames', 'help': 'Frames parameter for TrajectoryCoordinateSeriesRequest.', 'units': 'frame_index'},
    )
    every: int = dc_field(
        default=1,
        metadata={'label': 'Every', 'help': 'Every parameter for TrajectoryCoordinateSeriesRequest.', 'min': 1, 'units': 'frames'},
    )


@dataclass
class CellDimensionsRequest(BaseRequest):
    fields: Sequence[str] = dc_field(
        default=("a", "b", "c"),
        metadata={
            "label": "Fields",
            "help": "Cell fields to include.",
            "choices": ["a", "b", "c", "alpha", "beta", "gamma"],
        },
    )
    frames: Optional[Sequence[int]] = dc_field(
        default=None,
        metadata={'label': 'Frames', 'help': 'Frames parameter for CellDimensionsRequest.', 'units': 'frame_index'},
    )
    every: int = dc_field(
        default=1,
        metadata={'label': 'Every', 'help': 'Every parameter for CellDimensionsRequest.', 'min': 1, 'units': 'frames'},
    )


@dataclass
class ChargeSeriesRequest(BaseRequest):
    atom_ids: Sequence[int] = dc_field(
        metadata={'label': 'Atom Ids', 'help': 'Atom Ids parameter for ChargeSeriesRequest.', 'units': 'index'},
    )
    frames: Optional[Sequence[int]] = dc_field(
        default=None,
        metadata={'label': 'Frames', 'help': 'Frames parameter for ChargeSeriesRequest.', 'units': 'frame_index'},
    )
    every: int = dc_field(
        default=1,
        metadata={'label': 'Every', 'help': 'Every parameter for ChargeSeriesRequest.', 'min': 1, 'units': 'frames'},
    )


@dataclass
class ElectricFieldSeriesRequest(BaseRequest):
    components: Sequence[str] = dc_field(
        metadata={
            "label": "Components",
            "help": "Electric-field components to include.",
            "choices": ["field_x", "field_y", "field_z", "E_field_x", "E_field_y", "E_field_z", "E_field"],
        },
    )
    field_kind: Literal["applied", "energy", "auto"] = dc_field(
        default="auto",
        metadata={
            "label": "Field Kind",
            "help": "Which electric-field group to use.",
            "choices": ["applied", "energy", "auto"],
        },
    )
    frames: Optional[Sequence[int]] = dc_field(
        default=None,
        metadata={'label': 'Frames', 'help': 'Frames parameter for ElectricFieldSeriesRequest.', 'units': 'frame_index'},
    )
    every: int = dc_field(
        default=1,
        metadata={'label': 'Every', 'help': 'Every parameter for ElectricFieldSeriesRequest.', 'min': 1, 'units': 'frames'},
    )


@dataclass
class EregimeSeriesRequest(BaseRequest):
    field: str = dc_field(
        metadata={
            "label": "Field",
            "help": "Eregime column to extract.",
            "choices": ["field", "field_zones", "field_dir"],
        },
    )
    frames: Optional[Sequence[int]] = dc_field(
        default=None,
        metadata={'label': 'Frames', 'help': 'Frames parameter for EregimeSeriesRequest.', 'units': 'frame_index'},
    )
    every: int = dc_field(
        default=1,
        metadata={'label': 'Every', 'help': 'Every parameter for EregimeSeriesRequest.', 'min': 1, 'units': 'frames'},
    )


@dataclass
class PartialEnergySeriesRequest(BaseRequest):
    components: Optional[Sequence[str]] = dc_field(
        default=None,
        metadata={
            "label": "Components",
            "help": "Partial-energy components to include.",
            "choices": [
                "ebond",
                "eover",
                "eunder",
                "eangle",
                "epen",
                "etors",
                "econj",
                "evdw",
                "ecoul",
                "ehb",
                "eself",
                "E_pot",
            ],
        },
    )
    frames: Optional[Sequence[int]] = dc_field(
        default=None,
        metadata={'label': 'Frames', 'help': 'Frames parameter for PartialEnergySeriesRequest.', 'units': 'frame_index'},
    )
    every: int = dc_field(
        default=1,
        metadata={'label': 'Every', 'help': 'Every parameter for PartialEnergySeriesRequest.', 'min': 1, 'units': 'frames'},
    )


@dataclass
class RestraintSeriesRequest(BaseRequest):
    fields: Optional[Sequence[str]] = dc_field(
        default=None,
        metadata={
            'label': 'Fields',
            'help': (
                "Restraint fields to extract. Use 'E_res'/'E_pot' for energy values, "
                "or use 'r_target'/'r_actual' together with restraint_index "
                "to select a specific restraint."
            ),
            'choices': ['E_res', 'E_pot', 'r_target', 'r_actual'],
        },
    )
    restraint_index: Optional[Union[int, Sequence[int]]] = dc_field(
        default=None,
        metadata={
            'label': 'Restraint Index',
            'help': (
                "1-based restraint index (or indices) used with "
                "'r_target'/'r_actual'. Example: 1 or [1, 2]."
            ),
        },
    )
    dropna_rows: bool = dc_field(
        default=False,
        metadata={'label': 'Dropna Rows', 'help': 'Dropna Rows parameter for RestraintSeriesRequest.', 'choices': [True, False]},
    )
    frames: Optional[Sequence[int]] = dc_field(
        default=None,
        metadata={'label': 'Frames', 'help': 'Frames parameter for RestraintSeriesRequest.', 'units': 'frame_index'},
    )
    every: int = dc_field(
        default=1,
        metadata={'label': 'Every', 'help': 'Every parameter for RestraintSeriesRequest.', 'min': 1, 'units': 'frames'},
    )


@dataclass
class MolecularFrequencySeriesRequest(BaseRequest):
    molecules: Sequence[str] = dc_field(
        metadata={'label': 'Molecules', 'help': 'Molecules parameter for MolecularFrequencySeriesRequest.'},
    )
    frames: Optional[Sequence[int]] = dc_field(
        default=None,
        metadata={'label': 'Frames', 'help': 'Frames parameter for MolecularFrequencySeriesRequest.', 'units': 'frame_index'},
    )
    every: int = dc_field(
        default=1,
        metadata={'label': 'Every', 'help': 'Every parameter for MolecularFrequencySeriesRequest.', 'min': 1, 'units': 'frames'},
    )


@dataclass
class MolecularTotalsSeriesRequest(BaseRequest):
    quantities: Sequence[str] = dc_field(
        default=("total_molecules", "total_atoms", "total_molecular_mass"),
        metadata={
            "label": "Quantities",
            "help": "Quantities to include in the molecular totals table.",
            "choices": ["total_molecules", "total_atoms", "total_molecular_mass"],
        },
    )
    frames: Optional[Sequence[int]] = dc_field(
        default=None,
        metadata={'label': 'Frames', 'help': 'Frames parameter for MolecularTotalsSeriesRequest.', 'units': 'frame_index'},
    )
    every: int = dc_field(
        default=1,
        metadata={'label': 'Every', 'help': 'Every parameter for MolecularTotalsSeriesRequest.', 'min': 1, 'units': 'frames'},
    )


def _simulation_field_array(data: SimulationData, field: str) -> tuple[np.ndarray, str]:
    key = str(field).strip()
    if key == "potential_energy":
        arr = data.potential_energy
        label = "potential_energy"
    elif key in {"volume", "V"}:
        arr = data.volume
        label = "volume"
    elif key in {"temperature", "T"}:
        arr = data.temperature
        label = "temperature"
    elif key in {"pressure", "P"}:
        arr = data.pressure
        label = "pressure"
    elif key in {"density", "D"}:
        arr = data.density
        label = "density"
    elif key in {"elapsed_time", "elap_time"}:
        arr = data.elapsed_time
        label = "elapsed_time"
    elif key == "num_of_atoms":
        arr = data.num_of_atoms
        label = "num_of_atoms"
    elif key in {"a", "b", "c"}:
        if data.cell_lengths is None:
            arr = None
        else:
            j = {"a": 0, "b": 1, "c": 2}[key]
            arr = np.asarray(data.cell_lengths, dtype=float)[:, j]
        label = key
    elif key in {"alpha", "beta", "gamma"}:
        if data.cell_angles is None:
            arr = None
        else:
            j = {"alpha": 0, "beta": 1, "gamma": 2}[key]
            arr = np.asarray(data.cell_angles, dtype=float)[:, j]
        label = key
    else:
        raise KeyError(
            f"Unsupported simulation field {field!r}. "
            "Choose from: potential_energy, volume, temperature, pressure, density, elapsed_time, num_of_atoms, a, b, c, alpha, beta, gamma."
        )

    if arr is None:
        raise ValueError(f"Simulation field {field!r} is not available in loaded data.")
    return np.asarray(arr), label


@register_task("simulation_series", label="Simulation Series")
class SimulationScalarSeriesTask(AnalysisTask):
    """Build a scalar time series from ``SimulationData``."""

    required_data = SimulationData

    @staticmethod
    def recommended_presentations(_result: SimulationScalarSeriesResult, payload: dict[str, Any]) -> list[PresentationSpec]:
        table_rows = payload.get("table")
        if not isinstance(table_rows, list) or not table_rows:
            return [PresentationSpec(renderer="table", label="Table", view_type="table")]
        sample = table_rows[0] if isinstance(table_rows[0], dict) else {}
        x_axis = "iter" if "iter" in sample else ("frame_index" if "frame_index" in sample else "")
        if not x_axis or "value" not in sample:
            return [PresentationSpec(renderer="table", label="Table", view_type="table")]
        group_by = "field" if "field" in sample else ""
        return [
            PresentationSpec(renderer="table", label="Table", view_type="table"),
            PresentationSpec(
                renderer="single_plot",
                label="value vs iter",
                mapping={"x_col": x_axis, "y_col": "value", "group_by_col": group_by},
                options={"title": "value vs iter", "xlabel": x_axis, "ylabel": "value", "legend": bool(group_by)},
                view_type="plot2d",
            ),
        ]

    def run(self, data: SimulationData, request: SimulationScalarSeriesRequest, reporter=None) -> SimulationScalarSeriesResult:
        values, label = _simulation_field_array(data, request.field)
        n_frames = int(values.shape[0])
        frame_idx = _frame_indices(n_frames, request.frames, request.every)
        iterations = (
            np.asarray(data.iterations, dtype=int)
            if data.iterations is not None
            else np.arange(n_frames, dtype=int)
        )
        if iterations.shape[0] != n_frames:
            raise ValueError("SimulationData.iterations length must match simulation scalar length.")

        table = pd.DataFrame(
            {
                "frame_index": np.asarray(frame_idx, dtype=int),
                "iter": iterations[frame_idx],
                "field": label,
                "value": np.asarray(values[frame_idx], dtype=float),
            }
        )
        return SimulationScalarSeriesResult(
            request=request,
            table=table,
        )


@register_task("trajectory_coordinate_series", label="Trajectory Coordinate Series")
class TrajectoryCoordinateSeriesTask(AnalysisTask):
    """Build coordinate time series for one or more atoms."""

    required_data = TrajectoryData
    VERSION = "2"

    @staticmethod
    def recommended_presentations(_result: TrajectoryCoordinateSeriesResult, payload: dict[str, Any]) -> list[PresentationSpec]:
        rows = _rows_from_payload(payload)
        if not rows:
            return [PresentationSpec(renderer="table", label="Table", view_type="table")]
        cols = {str(key) for key in rows[0].keys()}
        x_col = "iter" if "iter" in cols else ("frame_index" if "frame_index" in cols else "")
        if not x_col or "coord" not in cols:
            return [PresentationSpec(renderer="table", label="Table", view_type="table")]
        group_col = "series_label" if "series_label" in cols else ("dim" if "dim" in cols else "atom_id")
        return [
            PresentationSpec(renderer="table", label="Table", view_type="table"),
            PresentationSpec(
                renderer="single_plot",
                label=f"coord vs {x_col}",
                mapping={
                    "x_col": x_col,
                    "y_col": "coord",
                    "group_by_col": group_col if group_col in cols else "",
                },
                options={
                    "title": f"coord vs {x_col}",
                    "xlabel": x_col,
                    "ylabel": "coord",
                    "legend": bool(group_col and group_col in cols),
                },
                view_type="plot2d",
            ),
        ]

    def run(self, data: TrajectoryData, request: TrajectoryCoordinateSeriesRequest, reporter=None) -> TrajectoryCoordinateSeriesResult:
        dims = tuple(str(d).lower() for d in request.dims if str(d).lower() in {"x", "y", "z"})
        if not dims:
            raise ValueError("TrajectoryCoordinateSeriesRequest.dims must include at least one of: x, y, z.")
        positions = np.asarray(data.positions, dtype=float)
        n_frames, n_atoms = positions.shape[:2]
        frame_idx = _frame_indices(n_frames, request.frames, request.every)
        iterations = (
            np.asarray(data.iterations, dtype=int)
            if data.iterations is not None
            else np.arange(n_frames, dtype=int)
        )
        if iterations.shape[0] != n_frames:
            raise ValueError("TrajectoryData.iterations length must match number of frames.")

        if request.atom_ids is not None:
            atom_ids = [int(a) for a in request.atom_ids]
            atom_indices = []
            for atom_id in atom_ids:
                atom_idx = int(atom_id) - 1
                if not (0 <= atom_idx < n_atoms):
                    raise ValueError(f"atom_id {atom_id} out of range 1..{n_atoms}.")
                atom_indices.append(atom_idx)
        elif request.atom_types:
            chosen = {str(t) for t in request.atom_types}
            atom_indices = [i for i, elem in enumerate(data.elements) if str(elem) in chosen]
            atom_ids = [int(data.atom_ids[i]) for i in atom_indices]
        else:
            atom_indices = list(range(n_atoms))
            atom_ids = [int(a) for a in data.atom_ids]

        rows: list[dict[str, object]] = []
        dim_to_col = {"x": 0, "y": 1, "z": 2}
        for atom_id, atom_idx in zip(atom_ids, atom_indices):
            atom_type = str(data.elements[atom_idx])
            for dim in dims:
                col = dim_to_col[dim]
                values = positions[frame_idx, atom_idx, col]
                for rel_i, fi in enumerate(frame_idx):
                    rows.append(
                        {
                            "frame_index": int(fi),
                            "iter": int(iterations[fi]),
                            "atom_id": int(atom_id),
                            "atom_type": atom_type,
                            "dim": str(dim),
                            "coord": float(values[rel_i]),
                        }
                    )

        table = pd.DataFrame(rows)
        print(table.head())
        if not table.empty:
            table = table.sort_values(["frame_index", "atom_id", "dim"], kind="stable").reset_index(drop=True)

        return TrajectoryCoordinateSeriesResult(table=table, request=request)


@register_task("cell_dimensions", label="Cell Dimensions")
class CellDimensionsTask(AnalysisTask):
    """Build cell-dimension time series from ``SimulationData``."""

    required_data = SimulationData

    @staticmethod
    def recommended_presentations(_result: CellDimensionsResult, payload: dict[str, Any]) -> list[PresentationSpec]:
        table_rows = payload.get("table")
        if not isinstance(table_rows, list) or not table_rows:
            return [PresentationSpec(renderer="table", label="Table", view_type="table")]
        sample = table_rows[0] if isinstance(table_rows[0], dict) else {}
        x_axis = "iter" if "iter" in sample else ("frame_index" if "frame_index" in sample else "")
        if not x_axis or "value" not in sample:
            return [PresentationSpec(renderer="table", label="Table", view_type="table")]
        group_by = "field" if "field" in sample else ""
        return [
            PresentationSpec(renderer="table", label="Table", view_type="table"),
            PresentationSpec(
                renderer="single_plot",
                label=f"value vs {x_axis}",
                mapping={"x_col": x_axis, "y_col": "value", "group_by_col": group_by},
                options={"title": f"value vs {x_axis}", "xlabel": x_axis, "ylabel": "value", "legend": bool(group_by)},
                view_type="plot2d",
            ),
        ]

    def run(self, data: SimulationData, request: CellDimensionsRequest, reporter=None) -> CellDimensionsResult:
        if data.cell_lengths is not None:
            n_frames = len(data.cell_lengths)
        elif data.cell_angles is not None:
            n_frames = len(data.cell_angles)
        elif data.iterations is not None:
            n_frames = len(data.iterations)
        else:
            n_frames = 0
        frame_idx = _frame_indices(
            n_frames,
            request.frames,
            request.every,
        )
        iterations = (
            np.asarray(data.iterations, dtype=int)
            if data.iterations is not None
            else np.arange(n_frames, dtype=int)
        )
        rows: list[dict[str, object]] = []
        for field in request.fields:
            values, label = _simulation_field_array(data, str(field))
            for rel_i, fi in enumerate(frame_idx):
                rows.append(
                    {
                        "frame_index": int(fi),
                        "iter": int(iterations[fi]),
                        "field": str(label),
                        "value": float(values[frame_idx][rel_i]),
                    }
                )
        table = pd.DataFrame(rows)
        if not table.empty:
            table = table.sort_values(["frame_index", "field"], kind="stable").reset_index(drop=True)
        return CellDimensionsResult(
            request=request,
            table=table,
        )


@register_task("charge_series", label="Charge Series")
class ChargeSeriesTask(AnalysisTask):
    """Build charge time series for one or more atoms."""

    required_data = ChargeData

    @staticmethod
    def recommended_presentations(_result: ChargeSeriesResult, payload: dict[str, Any]) -> list[PresentationSpec]:
        table_rows = payload.get("table")
        if not isinstance(table_rows, list) or not table_rows:
            return [PresentationSpec(renderer="table", label="Table", view_type="table")]
        sample = table_rows[0] if isinstance(table_rows[0], dict) else {}
        x_axis = "iter" if "iter" in sample else ("frame_index" if "frame_index" in sample else "")
        if not x_axis or "charge" not in sample:
            return [PresentationSpec(renderer="table", label="Table", view_type="table")]
        group_by = "atom_id" if "atom_id" in sample else ""
        return [
            PresentationSpec(renderer="table", label="Table", view_type="table"),
            PresentationSpec(
                renderer="single_plot",
                label=f"charge vs {x_axis}",
                mapping={"x_col": x_axis, "y_col": "charge", "group_by_col": group_by},
                options={"title": f"charge vs {x_axis}", "xlabel": x_axis, "ylabel": "charge", "legend": bool(group_by)},
                view_type="plot2d",
            ),
        ]

    def run(self, data: ChargeData, request: ChargeSeriesRequest, reporter=None) -> ChargeSeriesResult:
        charges = np.asarray(data.charges, dtype=float)
        if charges.ndim != 2:
            raise ValueError("ChargeData.charges must have shape (n_frames, n_atoms).")

        n_frames, n_atoms = charges.shape
        frame_idx = _frame_indices(n_frames, request.frames, request.every)
        iterations = (
            np.asarray(data.iterations, dtype=int).reshape(-1)
            if data.iterations is not None
            else np.arange(n_frames, dtype=int)
        )
        if iterations.shape[0] != n_frames:
            raise ValueError("ChargeData.iterations length must match number of frames.")

        if data.simulation is not None and data.simulation.atom_ids is not None:
            available_atom_ids = [int(a) for a in data.simulation.atom_ids]
        else:
            available_atom_ids = list(range(1, n_atoms + 1))
        if len(available_atom_ids) != n_atoms:
            raise ValueError("ChargeData atom_ids length must match number of atoms.")

        if data.simulation is not None and data.simulation.elements is not None:
            elements = [str(e) for e in data.simulation.elements]
        else:
            elements = [""] * n_atoms

        atom_id_to_idx = {atom_id: i for i, atom_id in enumerate(available_atom_ids)}
        rows: list[dict[str, object]] = []
        for atom_id in request.atom_ids:
            atom_id_int = int(atom_id)
            if atom_id_int not in atom_id_to_idx:
                raise ValueError(f"atom_id {atom_id_int} not found in ChargeData.")
            atom_idx = atom_id_to_idx[atom_id_int]
            y = np.asarray(charges[frame_idx, atom_idx], dtype=float)
            atom_type = elements[atom_idx] if atom_idx < len(elements) else ""
            for rel_i, fi in enumerate(frame_idx):
                rows.append(
                    {
                        "frame_index": int(fi),
                        "iter": int(iterations[fi]),
                        "atom_id": atom_id_int,
                        "atom_type": atom_type,
                        "charge": float(y[rel_i]),
                    }
                )

        table = pd.DataFrame(rows)
        if not table.empty:
            table = table.sort_values(["frame_index", "atom_id"], kind="stable").reset_index(drop=True)

        return ChargeSeriesResult(
            request=request,
            table=table,
        )


def _electric_field_group(
    data: ElectricFieldData,
    field_kind: str,
) -> tuple[np.ndarray, list[str], str]:
    kind = str(field_kind).lower()
    if kind == "applied":
        return (
            np.asarray(data.applied_field_values, dtype=float),
            [str(c) for c in data.applied_field_components],
            "applied_field",
        )
    if kind == "energy":
        return (
            np.asarray(data.field_energy_values, dtype=float),
            [str(c) for c in data.field_energy_components],
            "field_energy",
        )
    return np.empty((0, 0), dtype=float), [], ""


def _resolve_eregime_dc_field(df: pd.DataFrame, name: str) -> str:
    if not name:
        raise ValueError("Column name is empty.")

    canonical = normalize_choice(name)
    hit = resolve_alias_from_columns(df.columns, canonical)
    if hit:
        return hit

    if canonical in {"field", "field_dir"}:
        for i in range(1, 16):
            cand = f"{canonical}{i}" if canonical != "field_dir" else f"field_dir{i}"
            hit = resolve_alias_from_columns(df.columns, cand)
            if hit:
                return hit

    raise ValueError(
        f"Column '{name}' not found (after alias resolution). "
        f"Available columns: {list(df.columns)}"
    )


def _partial_energy_frame(data: PartialEnergyData) -> pd.DataFrame:
    df = pd.DataFrame({"iter": pd.Series(data.iterations, dtype=int)})
    if data.components:
        df = pd.concat([df, pd.DataFrame(data.values, columns=list(data.components))], axis=1)
    return df


def _resolve_partial_energy_components(
    df: pd.DataFrame,
    components: Optional[Sequence[str]],
) -> list[str]:
    available = [str(c) for c in df.columns if str(c) != "iter"]
    if components is None or len(components) == 0:
        return available

    resolved: list[str] = []
    for component in components:
        canonical = normalize_choice(str(component))
        actual = resolve_alias_from_columns(available, canonical)
        if actual is None:
            raise KeyError(
                f"Requested component '{component}' not found in partial-energy data. "
                f"Available components: {available}"
            )
        if actual not in resolved:
            resolved.append(actual)
    return resolved


def _restraint_frame(data: RestraintData) -> pd.DataFrame:
    df = pd.DataFrame({"iter": pd.Series(data.iterations, dtype=int)})
    n_restraints = int(data.metadata.get("n_restraints", 0)) if data.metadata else 0
    if data.restraint_energy is not None:
        df["E_res"] = pd.Series(data.restraint_energy, dtype=float)
    else:
        df["E_res"] = pd.Series([pd.NA] * len(df))
    if data.potential_energy is not None:
        df["E_pot"] = pd.Series(data.potential_energy, dtype=float)
    else:
        df["E_pot"] = pd.Series([pd.NA] * len(df))
    for i in range(n_restraints):
        target_col = f"r{i + 1}_target"
        actual_col = f"r{i + 1}_actual"
        if i < data.target_values.shape[1]:
            df[target_col] = pd.Series(data.target_values[:, i], dtype=float)
        else:
            df[target_col] = pd.Series([pd.NA] * len(df))
        if i < data.actual_values.shape[1]:
            df[actual_col] = pd.Series(data.actual_values[:, i], dtype=float)
        else:
            df[actual_col] = pd.Series([pd.NA] * len(df))
    return df


def _resolve_restraint_dc_field(df: pd.DataFrame, requested: str) -> str:
    cols = [str(c) for c in df.columns]
    key = str(requested).strip().lower()
    if not key:
        raise KeyError("Requested restraint column is empty.")

    if "restraint" in key or key.startswith("r"):
        parts = key.replace("_", " ").split()
        idx = None
        for part in parts:
            if part.isdigit():
                idx = int(part)
                break
        if idx is not None:
            if "target" in parts:
                col = f"r{idx}_target"
            elif "actual" in parts:
                col = f"r{idx}_actual"
            else:
                raise KeyError(f"Restraint column must specify target or actual: '{requested}'")
            if col in cols:
                return col
            raise KeyError(
                f"Resolved '{requested}' to '{col}', but that column is not available. "
                f"Available columns: {cols}"
            )

    canonical = normalize_choice(str(requested))
    hit = resolve_alias_from_columns(cols, canonical)
    if hit is not None:
        return hit
    if str(requested) in cols:
        return str(requested)
    raise KeyError(
        f"Could not resolve requested restraint column '{requested}'. "
        f"Available columns: {cols}"
    )


def _resolve_restraint_fields(df: pd.DataFrame, request: RestraintSeriesRequest) -> list[str]:
    idx_values: list[int] = []
    if request.restraint_index is not None:
        raw_idx = request.restraint_index
        if isinstance(raw_idx, Sequence) and not isinstance(raw_idx, (str, bytes)):
            idx_values = [int(i) for i in raw_idx]
        else:
            idx_values = [int(raw_idx)]
        idx_values = list(dict.fromkeys(idx_values))
        if any(i < 1 for i in idx_values):
            raise ValueError("restraint_index values must be >= 1.")
        if request.fields is None or len(request.fields) == 0:
            out: list[str] = []
            for idx in idx_values:
                out.extend([f"r{idx}_target", f"r{idx}_actual"])
            return out
    if request.fields is None or len(request.fields) == 0:
        raise ValueError("RestraintSeriesRequest requires either fields or restraint_index.")
    resolved: list[str] = []
    for field in request.fields:
        key = normalize_choice(str(field))
        if key in {"r_target", "r_actual"}:
            if not idx_values:
                raise ValueError("restraint_index is required when fields include 'r_target' or 'r_actual'.")
            suffix = "target" if key == "r_target" else "actual"
            for idx in idx_values:
                actual = f"r{idx}_{suffix}"
                if actual not in [str(c) for c in df.columns]:
                    raise KeyError(
                        f"Requested restraint field '{field}' with restraint_index={idx} resolved to '{actual}', "
                        f"but that column is not available."
                    )
                if actual not in resolved:
                    resolved.append(actual)
            continue
        else:
            actual = _resolve_restraint_dc_field(df, str(field))
        if actual not in resolved:
            resolved.append(actual)
    return resolved


@register_task("electric_field_series", label="Electric Field Series")
class ElectricFieldSeriesTask(AnalysisTask):
    """Build time series for applied-field or field-energy components."""

    required_data = ElectricFieldData

    @staticmethod
    def recommended_presentations(_result: ElectricFieldSeriesResult, payload: dict[str, Any]) -> list[PresentationSpec]:
        table_rows = payload.get("table")
        if not isinstance(table_rows, list) or not table_rows:
            return [PresentationSpec(renderer="table", label="Table", view_type="table")]
        sample = table_rows[0] if isinstance(table_rows[0], dict) else {}
        x_axis = "iter" if "iter" in sample else ("frame_index" if "frame_index" in sample else "")
        if not x_axis or "value" not in sample:
            return [PresentationSpec(renderer="table", label="Table", view_type="table")]
        group_by = "component" if "component" in sample else ""
        return [
            PresentationSpec(renderer="table", label="Table", view_type="table"),
            PresentationSpec(
                renderer="single_plot",
                label=f"value vs {x_axis}",
                mapping={"x_col": x_axis, "y_col": "value", "group_by_col": group_by},
                options={"title": f"value vs {x_axis}", "xlabel": x_axis, "ylabel": "value", "legend": bool(group_by)},
                view_type="plot2d",
            ),
        ]

    def run(self, data: ElectricFieldData, request: ElectricFieldSeriesRequest, reporter=None) -> ElectricFieldSeriesResult:
        frame_values: np.ndarray
        component_names: list[str]
        y_label: str

        if str(request.field_kind).lower() == "auto":
            requested = [str(c) for c in request.components]
            applied_values, applied_names, _ = _electric_field_group(data, "applied")
            energy_values, energy_names, _ = _electric_field_group(data, "energy")
            if all(comp in applied_names for comp in requested):
                frame_values, component_names, y_label = applied_values, applied_names, "applied_field"
            elif all(comp in energy_names for comp in requested):
                frame_values, component_names, y_label = energy_values, energy_names, "field_energy"
            else:
                available = sorted(set(applied_names + energy_names))
                raise KeyError(
                    f"Requested components {requested} are not available in a single field group. "
                    f"Available components: {available}"
                )
        else:
            frame_values, component_names, y_label = _electric_field_group(data, request.field_kind)

        if frame_values.ndim == 1:
            frame_values = frame_values.reshape(-1, 1)
        if frame_values.ndim != 2:
            raise ValueError("ElectricFieldData values must be 1D or 2D.")

        n_frames = frame_values.shape[0]
        iterations = (
            np.asarray(data.sampled_field_iterations, dtype=int).reshape(-1)
            if data.sampled_field_iterations is not None
            else np.arange(n_frames, dtype=int)
        )
        if iterations.shape[0] != n_frames:
            raise ValueError("ElectricFieldData.sampled_field_iterations length must match number of samples.")
        frame_idx = _frame_indices(n_frames, request.frames, request.every)

        rows: list[dict[str, object]] = []
        for comp in request.components:
            comp_name = str(comp)
            if comp_name not in component_names:
                raise KeyError(f"Electric field component '{comp_name}' not found in {component_names}.")
            j = component_names.index(comp_name)
            y = np.asarray(frame_values[frame_idx, j], dtype=float)
            for rel_i, fi in enumerate(frame_idx):
                rows.append(
                    {
                        "frame_index": int(fi),
                        "iter": int(iterations[fi]),
                        "component": comp_name,
                        "value": float(y[rel_i]),
                    }
                )

        table = pd.DataFrame(rows)
        if not table.empty:
            table = table.sort_values(["frame_index", "component"], kind="stable").reset_index(drop=True)

        return ElectricFieldSeriesResult(
            request=request,
            table=table,
        )


@register_task("eregime_series", label="Eregime Series")
class EregimeSeriesTask(AnalysisTask):
    """Build iteration-based series for one eregime field column."""

    required_data = EregimeData

    @staticmethod
    def recommended_presentations(_result: EregimeSeriesResult, payload: dict[str, Any]) -> list[PresentationSpec]:
        table_rows = payload.get("table")
        if not isinstance(table_rows, list) or not table_rows:
            return [PresentationSpec(renderer="table", label="Table", view_type="table")]
        sample = table_rows[0] if isinstance(table_rows[0], dict) else {}
        x_axis = "iter" if "iter" in sample else ("frame_index" if "frame_index" in sample else "")
        if not x_axis or "value" not in sample:
            return [PresentationSpec(renderer="table", label="Table", view_type="table")]
        group_by = "field" if "field" in sample else ""
        return [
            PresentationSpec(renderer="table", label="Table", view_type="table"),
            PresentationSpec(
                renderer="single_plot",
                label=f"value vs {x_axis}",
                mapping={"x_col": x_axis, "y_col": "value", "group_by_col": group_by},
                options={"title": f"value vs {x_axis}", "xlabel": x_axis, "ylabel": "value", "legend": bool(group_by)},
                view_type="plot2d",
            ),
        ]

    def run(self, data: EregimeData, request: EregimeSeriesRequest, reporter=None) -> EregimeSeriesResult:
        iterations = np.asarray(data.iterations, dtype=int).reshape(-1)
        field_zones = np.asarray(data.field_zones, dtype=int).reshape(-1)
        field_dir = np.asarray(data.field_dir, dtype=object).reshape(-1)
        field = np.asarray(data.field, dtype=float).reshape(-1)
        df = pd.DataFrame(
            {
                "iter": iterations,
                "field_zones": field_zones,
                "field_dir": field_dir,
                "field": field,
            }
        )
        field_col = _resolve_eregime_dc_field(df, request.field)
        n_frames = iterations.shape[0]
        frame_idx = _frame_indices(n_frames, request.frames, request.every)

        table = pd.DataFrame(
            {
                "frame_index": np.asarray(frame_idx, dtype=int),
                "iter": iterations[frame_idx],
                "field": str(field_col),
                "value": df.iloc[frame_idx][field_col].to_numpy(),
            }
        )
        return EregimeSeriesResult(
            request=request,
            table=table,
        )


@register_task("partial_energy_series", label="Partial Energy Series")
class PartialEnergySeriesTask(AnalysisTask):
    """Build iteration-based series for one or more fort.73 energy components."""

    required_data = PartialEnergyData

    @staticmethod
    def recommended_presentations(_result: PartialEnergySeriesResult, payload: dict[str, Any]) -> list[PresentationSpec]:
        table_rows = payload.get("table")
        if not isinstance(table_rows, list) or not table_rows:
            return [PresentationSpec(renderer="table", label="Table", view_type="table")]
        sample = table_rows[0] if isinstance(table_rows[0], dict) else {}
        x_axis = "iter" if "iter" in sample else ("frame_index" if "frame_index" in sample else "")
        if not x_axis or "value" not in sample:
            return [PresentationSpec(renderer="table", label="Table", view_type="table")]
        group_by = "component" if "component" in sample else ""
        return [
            PresentationSpec(renderer="table", label="Table", view_type="table"),
            PresentationSpec(
                renderer="single_plot",
                label=f"value vs {x_axis}",
                mapping={"x_col": x_axis, "y_col": "value", "group_by_col": group_by},
                options={"title": f"value vs {x_axis}", "xlabel": x_axis, "ylabel": "value", "legend": bool(group_by)},
                view_type="plot2d",
            ),
        ]

    def run(self, data: PartialEnergyData, request: PartialEnergySeriesRequest, reporter=None) -> PartialEnergySeriesResult:
        df = _partial_energy_frame(data)
        iterations = np.asarray(data.iterations, dtype=int).reshape(-1)
        n_frames = iterations.shape[0]
        if len(df) != n_frames:
            raise ValueError("PartialEnergyData.values length must match PartialEnergyData.iterations length.")

        components = _resolve_partial_energy_components(df, request.components)
        frame_idx = _frame_indices(n_frames, request.frames, request.every)

        rows: list[dict[str, object]] = []
        for component in components:
            y = pd.to_numeric(df.iloc[frame_idx][component], errors="coerce").to_numpy(dtype=float)
            for rel_i, fi in enumerate(frame_idx):
                rows.append(
                    {
                        "frame_index": int(fi),
                        "iter": int(iterations[fi]),
                        "component": str(component),
                        "value": float(y[rel_i]),
                    }
                )

        table = pd.DataFrame(rows)
        if not table.empty:
            table = table.sort_values(["frame_index", "component"], kind="stable").reset_index(drop=True)

        return PartialEnergySeriesResult(
            request=request,
            table=table,
        )


@register_task("restraint_series", label="Restraint Series")
class RestraintSeriesTask(AnalysisTask):
    """Build iteration-based series for fort.76 restraint data."""

    required_data = RestraintData

    @staticmethod
    def recommended_presentations(_result: RestraintSeriesResult, payload: dict[str, Any]) -> list[PresentationSpec]:
        table_rows = payload.get("table")
        if not isinstance(table_rows, list) or not table_rows:
            return [PresentationSpec(renderer="table", label="Table", view_type="table")]
        sample = table_rows[0] if isinstance(table_rows[0], dict) else {}
        x_axis = "iter" if "iter" in sample else ("frame_index" if "frame_index" in sample else "")
        if not x_axis or "value" not in sample:
            return [PresentationSpec(renderer="table", label="Table", view_type="table")]
        group_by = "field" if "field" in sample else ("restraint_index" if "restraint_index" in sample else "")
        return [
            PresentationSpec(renderer="table", label="Table", view_type="table"),
            PresentationSpec(
                renderer="single_plot",
                label=f"value vs {x_axis}",
                mapping={"x_col": x_axis, "y_col": "value", "group_by_col": group_by},
                options={"title": f"value vs {x_axis}", "xlabel": x_axis, "ylabel": "value", "legend": bool(group_by)},
                view_type="plot2d",
            ),
        ]

    def run(self, data: RestraintData, request: RestraintSeriesRequest, reporter=None) -> RestraintSeriesResult:
        df = _restraint_frame(data)
        iterations = np.asarray(data.iterations, dtype=int).reshape(-1)
        n_frames = iterations.shape[0]
        if len(df) != n_frames:
            raise ValueError("RestraintData arrays must all align with RestraintData.iterations.")

        fields = _resolve_restraint_fields(df, request)
        frame_idx = _frame_indices(n_frames, request.frames, request.every)
        selected = df.iloc[frame_idx].copy()
        selected["frame_index"] = np.asarray(frame_idx, dtype=int)
        selected["iter"] = iterations[frame_idx]
        selected = selected.loc[:, ["frame_index", "iter", *[c for c in selected.columns if c not in {"frame_index", "iter"}]]]
        if request.dropna_rows:
            non_iter = [c for c in fields if c not in {"iter", "frame_index"}]
            if non_iter:
                selected = selected.dropna(axis=0, how="all", subset=non_iter).reset_index(drop=True)
        rows: list[dict[str, object]] = []
        for row in selected.itertuples(index=False):
            for field in fields:
                value = pd.to_numeric(pd.Series([getattr(row, field)]), errors="coerce").iloc[0]
                fname = str(field)
                field_label = fname
                restraint_index: int | None = None
                if fname.startswith("r"):
                    parts = fname.split("_", 1)
                    num = parts[0][1:]
                    suffix = parts[1] if len(parts) > 1 else ""
                    if num.isdigit():
                        restraint_index = int(num)
                    if num.isdigit() and suffix in {"target", "actual"}:
                        field_label = f"r_{suffix}"
                rows.append(
                    {
                        "frame_index": int(getattr(row, "frame_index")),
                        "iter": int(getattr(row, "iter")),
                        "restraint_index": restraint_index,
                        "field": field_label,
                        "value": float(value) if pd.notna(value) else np.nan,
                    }
                )
        table = pd.DataFrame(rows)
        if not table.empty:
            table = table.sort_values(["frame_index", "field"], kind="stable").reset_index(drop=True)

        return RestraintSeriesResult(
            request=request,
            table=table,
        )


@register_task("molecular_frequency_series", label="Molecular Frequency Series")
class MolecularFrequencySeriesTask(AnalysisTask):
    """Build molecular-frequency time series for one or more molecular formulas."""

    required_data = MolecularAnalysisData

    @staticmethod
    def recommended_presentations(_result: MolecularFrequencySeriesResult, payload: dict[str, Any]) -> list[PresentationSpec]:
        table_rows = payload.get("table")
        if not isinstance(table_rows, list) or not table_rows:
            return [PresentationSpec(renderer="table", label="Table", view_type="table")]
        sample = table_rows[0] if isinstance(table_rows[0], dict) else {}
        x_axis = "iter" if "iter" in sample else ("frame_index" if "frame_index" in sample else "")
        if not x_axis or "freq" not in sample:
            return [PresentationSpec(renderer="table", label="Table", view_type="table")]
        group_by = "molecules" if "molecules" in sample else ""
        return [
            PresentationSpec(renderer="table", label="Table", view_type="table"),
            PresentationSpec(
                renderer="single_plot",
                label=f"freq vs {x_axis}",
                mapping={"x_col": x_axis, "y_col": "freq", "group_by_col": group_by},
                options={"title": f"freq vs {x_axis}", "xlabel": x_axis, "ylabel": "freq", "legend": bool(group_by)},
                view_type="plot2d",
            ),
        ]

    def run(
        self,
        data: MolecularAnalysisData,
        request: MolecularFrequencySeriesRequest,
        reporter=None,
    ) -> MolecularFrequencySeriesResult:
        df = data.molecular_species.copy()
        iterations = np.asarray(data.iterations, dtype=int).reshape(-1)
        n_frames = iterations.shape[0]
        frame_idx = _frame_indices(n_frames, request.frames, request.every)
        selected_iters = iterations[frame_idx]

        rows: list[dict[str, object]] = []
        for molecule in request.molecules:
            sub = df[df["molecular_formula"] == str(molecule)][["iter", "freq", "molecular_mass"]].copy()
            sub["iter"] = pd.to_numeric(sub["iter"], errors="coerce").astype(int)
            sub["freq"] = pd.to_numeric(sub["freq"], errors="coerce").astype(float)
            sub["molecular_mass"] = pd.to_numeric(sub["molecular_mass"], errors="coerce").astype(float)
            freq_by_iter = pd.Series(sub["freq"].to_numpy(dtype=float), index=sub["iter"].to_numpy(dtype=int))
            mass_by_iter = pd.Series(sub["molecular_mass"].to_numpy(dtype=float), index=sub["iter"].to_numpy(dtype=int))
            y = freq_by_iter.reindex(selected_iters, fill_value=0.0).to_numpy(dtype=float)
            mass = mass_by_iter.reindex(selected_iters, fill_value=np.nan).to_numpy(dtype=float)
            for rel_i, fi in enumerate(frame_idx):
                rows.append(
                    {
                        "frame_index": int(fi),
                        "iter": int(selected_iters[rel_i]),
                        "molecules": str(molecule),
                        "freq": float(y[rel_i]),
                        "molecular_mass": float(mass[rel_i]) if not np.isnan(mass[rel_i]) else np.nan,
                    }
                )

        table = pd.DataFrame(rows)
        if not table.empty:
            table = table.sort_values(["frame_index", "molecules"], kind="stable").reset_index(drop=True)

        return MolecularFrequencySeriesResult(
            request=request,
            table=table,
        )


@register_task("molecular_totals_series", label="Molecular Totals Series")
class MolecularTotalsSeriesTask(AnalysisTask):
    """Build total molecule/atom/mass time series from MolecularAnalysisData.totals."""

    required_data = MolecularAnalysisData

    @staticmethod
    def recommended_presentations(_result: MolecularTotalsSeriesResult, payload: dict[str, Any]) -> list[PresentationSpec]:
        table_rows = payload.get("table")
        if not isinstance(table_rows, list) or not table_rows:
            return [PresentationSpec(renderer="table", label="Table", view_type="table")]
        sample = table_rows[0] if isinstance(table_rows[0], dict) else {}
        x_col = "iter" if "iter" in sample else ("frame_index" if "frame_index" in sample else "")
        if not x_col or "value" not in sample:
            return [PresentationSpec(renderer="table", label="Table", view_type="table")]
        group_col = "quantity" if "quantity" in sample else ""
        return [
            PresentationSpec(renderer="table", label="Table", view_type="table"),
            PresentationSpec(
                renderer="single_plot",
                label=f"value vs {x_col}",
                mapping={"x_col": x_col, "y_col": "value", "group_by_col": group_col},
                options={"title": f"value vs {x_col}", "xlabel": x_col, "ylabel": "value", "legend": bool(group_col)},
                view_type="plot2d",
            ),
        ]

    def run(
        self,
        data: MolecularAnalysisData,
        request: MolecularTotalsSeriesRequest,
        reporter=None,
    ) -> MolecularTotalsSeriesResult:
        df = data.totals.copy()
        if df.empty:
            return MolecularTotalsSeriesResult(
                request=request,
                table=pd.DataFrame(),
            )

        iterations = pd.to_numeric(df["iter"], errors="coerce").to_numpy(dtype=int)
        n_frames = len(iterations)
        frame_idx = _frame_indices(n_frames, request.frames, request.every)
        if not frame_idx:
            return MolecularTotalsSeriesResult(
                request=request,
                table=pd.DataFrame(),
            )

        quantities = [q for q in request.quantities if q in df.columns]
        rows: list[dict[str, object]] = []
        for quantity in quantities:
            y = pd.to_numeric(df.iloc[frame_idx][quantity], errors="coerce").to_numpy(dtype=float)
            for rel_i, fi in enumerate(frame_idx):
                rows.append(
                    {
                        "frame_index": int(fi),
                        "iter": int(iterations[fi]),
                        "quantity": str(quantity),
                        "value": float(y[rel_i]),
                    }
                )

        table = pd.DataFrame(rows)
        if not table.empty:
            table = table.sort_values(["frame_index", "quantity"], kind="stable").reset_index(drop=True)

        return MolecularTotalsSeriesResult(
            request=request,
            table=table,
        )


__all__ = [
    "Series",
    "TimeSeriesResult",
    "SimulationScalarSeriesResult",
    "TrajectoryCoordinateSeriesResult",
    "CellDimensionsResult",
    "ChargeSeriesResult",
    "ElectricFieldSeriesResult",
    "EregimeSeriesResult",
    "PartialEnergySeriesResult",
    "RestraintSeriesResult",
    "MolecularFrequencySeriesResult",
    "MolecularTotalsSeriesResult",
    "SimulationScalarSeriesRequest",
    "SimulationScalarSeriesTask",
    "TrajectoryCoordinateSeriesRequest",
    "TrajectoryCoordinateSeriesTask",
    "CellDimensionsRequest",
    "CellDimensionsTask",
    "ChargeSeriesRequest",
    "ChargeSeriesTask",
    "ElectricFieldSeriesRequest",
    "ElectricFieldSeriesTask",
    "EregimeSeriesRequest",
    "EregimeSeriesTask",
    "PartialEnergySeriesRequest",
    "PartialEnergySeriesTask",
    "RestraintSeriesRequest",
    "RestraintSeriesTask",
    "MolecularFrequencySeriesRequest",
    "MolecularFrequencySeriesTask",
    "MolecularTotalsSeriesRequest",
    "MolecularTotalsSeriesTask",
]
