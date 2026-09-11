"""Shared CLI support for dedicated time-series workflows."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from reaxkit.analysis import timeseries as _timeseries_tasks  # noqa: F401
from reaxkit.analysis.timeseries.geometry_optimization import GeometryOptimizationRequest
from reaxkit.analysis.timeseries.timeseries import (
    CellDimensionsRequest,
    ChargeSeriesRequest,
    ElectricFieldSeriesRequest,
    EregimeSeriesRequest,
    MolecularFrequencySeriesRequest,
    MolecularTotalsSeriesRequest,
    PartialEnergySeriesRequest,
    RestraintSeriesRequest,
    SimulationScalarSeriesRequest,
    TrajectoryCoordinateSeriesRequest,
    TrajectoryDisplacementSeriesRequest,
)
from reaxkit.core.registry.analysis_task_registry import TASK_REGISTRY
from reaxkit.core.runtime.analysis_executor import AnalysisExecutor
from reaxkit.core.storage.storage_layout import add_storage_cli_arguments
from reaxkit.core.utils.frame_utils import parse_frame_indices
from reaxkit.presentation.convert import convert_xaxis
from reaxkit.presentation.dispatcher import present_result

RequestBuilder = Callable[[argparse.Namespace], object]


def _frames(args: argparse.Namespace) -> list[int] | None:
    return parse_frame_indices(getattr(args, "frames", None))


def _add_runtime_arguments(parser: argparse.ArgumentParser, *inputs: str) -> None:
    parser.add_argument("--engine", choices=["reaxff", "ams", "lammps"], default=None)
    parser.add_argument("--input", default=".", help="Input file or directory used for engine detection.")
    parser.add_argument("--run-dir", default=".", help="Run directory used as a fallback for input discovery.")
    defaults = {
        "xmolout": "xmolout",
        "summary": None,
        "fort7": "fort.7",
        "fort73": "fort.73",
        "fort76": "fort.76",
        "fort78": "fort.78",
        "fort57": "fort.57",
        "eregime": "eregime.in",
        "molfra": "molfra.out",
    }
    descriptions = {
        "xmolout": "Trajectory input path.",
        "summary": "Optional summary.txt input path.",
        "fort7": "Charge data input path.",
        "fort73": "Partial-energy input path.",
        "fort76": "Restraint data input path.",
        "fort78": "Electric-field input path.",
        "fort57": "Geometry-optimization input path.",
        "eregime": "Electric-field regime input path.",
        "molfra": "Molecular-analysis input path.",
    }
    for name in inputs:
        parser.add_argument(f"--{name}", default=defaults[name], help=descriptions[name])
    parser.add_argument("--log", choices=["verbose", "quiet"], default=None)
    add_storage_cli_arguments(parser)


def _add_sampling_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--frames",
        nargs="*",
        default=None,
        help='Frame selector, for example --frames 0:20:2 or --frames 0,5,10.',
    )
    parser.add_argument("--every", type=int, default=1, help="Keep every Nth selected frame.")


def _add_presentation_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--plot", choices=["single", "subplot"], default=None)
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--save", default=None)
    parser.add_argument("--export", default=None)
    parser.add_argument("--grid", default=None, help="Subplot grid such as 2x2.")
    parser.add_argument("--xaxis", choices=["iter", "frame", "time"], default="iter")
    parser.add_argument(
        "--control",
        default="control",
        help=(
            "Control file used for frame (iout2) and time conversion. "
            "The default also searches beside the selected input files."
        ),
    )
    parser.add_argument(
        "--frame-source",
        default=None,
        help=(
            "Trajectory file whose headers define frame iterations when no control "
            "file exists. By default, the configured or sibling xmolout is used."
        ),
    )
    parser.add_argument(
        "--frame-count",
        type=int,
        default=None,
        help=(
            "Total trajectory frame count used to infer frame spacing only when neither "
            "a control file nor a trajectory frame source exists."
        ),
    )


def configure_parser(
    parser: argparse.ArgumentParser,
    *,
    command: str,
    description: str,
    inputs: Sequence[str],
) -> argparse.ArgumentParser:
    """Attach arguments shared by one dedicated time-series command."""
    parser.set_defaults(command=command, progress=True)
    parser.formatter_class = argparse.RawTextHelpFormatter
    parser.description = description
    _add_runtime_arguments(parser, *inputs)
    _add_sampling_arguments(parser)
    _add_presentation_arguments(parser)
    return parser


def build_simulation_request(args: argparse.Namespace, field: str) -> SimulationScalarSeriesRequest:
    return SimulationScalarSeriesRequest(field=field, frames=_frames(args), every=int(args.every))


def build_trajectory_request(args: argparse.Namespace) -> TrajectoryCoordinateSeriesRequest:
    return TrajectoryCoordinateSeriesRequest(
        atom_ids=tuple(args.atom_ids) if args.atom_ids else None,
        atom_types=tuple(args.atom_types) if args.atom_types else None,
        dims=tuple(args.dims),
        frames=_frames(args),
        every=int(args.every),
    )


def build_displacement_request(args: argparse.Namespace) -> TrajectoryDisplacementSeriesRequest:
    return TrajectoryDisplacementSeriesRequest(
        atom_ids=tuple(args.atom_ids) if args.atom_ids else None,
        atom_types=tuple(args.atom_types) if args.atom_types else None,
        dims=tuple(args.dims),
        reference_frame=int(args.reference_frame),
        frames=_frames(args),
        every=int(args.every),
    )


def build_charge_request(args: argparse.Namespace) -> ChargeSeriesRequest:
    atom_ids = tuple(args.atom_ids) if args.atom_ids else None
    return ChargeSeriesRequest(atom_ids=atom_ids, frames=_frames(args), every=int(args.every))


def build_cell_dimensions_request(args: argparse.Namespace) -> CellDimensionsRequest:
    return CellDimensionsRequest(fields=tuple(args.fields), frames=_frames(args), every=int(args.every))


def build_electric_field_request(args: argparse.Namespace) -> ElectricFieldSeriesRequest:
    return ElectricFieldSeriesRequest(
        components=tuple(args.components),
        field_kind=args.field_kind,
        frames=_frames(args),
        every=int(args.every),
    )


def build_eregime_request(args: argparse.Namespace) -> EregimeSeriesRequest:
    return EregimeSeriesRequest(field=args.field, frames=_frames(args), every=int(args.every))


def build_partial_energy_request(args: argparse.Namespace) -> PartialEnergySeriesRequest:
    components = tuple(args.components) if args.components else None
    return PartialEnergySeriesRequest(components=components, frames=_frames(args), every=int(args.every))


def build_restraint_request(args: argparse.Namespace) -> RestraintSeriesRequest:
    fields = tuple(args.fields) if args.fields else None
    indices = tuple(args.restraint_index) if args.restraint_index else None
    return RestraintSeriesRequest(
        fields=fields,
        restraint_index=indices,
        dropna_rows=bool(args.dropna_rows),
        frames=_frames(args),
        every=int(args.every),
    )


def build_molecular_frequency_request(args: argparse.Namespace) -> MolecularFrequencySeriesRequest:
    return MolecularFrequencySeriesRequest(
        molecules=tuple(args.molecules), frames=_frames(args), every=int(args.every)
    )


def build_molecular_totals_request(
    args: argparse.Namespace,
    quantities: Sequence[str] | None = None,
) -> MolecularTotalsSeriesRequest:
    selected = tuple(quantities) if quantities is not None else tuple(args.quantities)
    return MolecularTotalsSeriesRequest(quantities=selected, frames=_frames(args), every=int(args.every))


def build_geometry_optimization_request(args: argparse.Namespace) -> GeometryOptimizationRequest:
    components = tuple(args.components) if args.components else None
    return GeometryOptimizationRequest(
        component=components,
        include_geo_descriptor=bool(args.include_geo_descriptor),
    )


def _plot_axis(table: pd.DataFrame, args: argparse.Namespace) -> tuple[np.ndarray, str, str]:
    mode = str(getattr(args, "xaxis", "iter"))
    if mode == "frame" and "iter" in table:
        control_file = _axis_control_file(args)
        iterations = pd.to_numeric(table["iter"], errors="coerce").to_numpy(dtype=int)
        values, label = convert_xaxis(
            iterations,
            "frame",
            control_file=control_file,
            trajectory_file=_axis_frame_source(args),
            frame_count=getattr(args, "frame_count", None),
        )
        return np.asarray(values), label, "iter"
    if mode == "frame" and "frame_index" in table:
        return pd.to_numeric(table["frame_index"], errors="coerce").to_numpy(dtype=float), "frame", "frame_index"
    if "iter" not in table:
        if "frame_index" not in table:
            raise ValueError("The result has neither an iteration nor a frame axis.")
        return pd.to_numeric(table["frame_index"], errors="coerce").to_numpy(dtype=float), "frame", "frame_index"
    iterations = pd.to_numeric(table["iter"], errors="coerce").to_numpy(dtype=int)
    if mode == "time":
        values, label = convert_xaxis(iterations, "time", control_file=_axis_control_file(args))
        return np.asarray(values), label, "iter"
    return iterations, "iter", "iter"


def _axis_input_directories(args: argparse.Namespace) -> list[Path]:
    """Return ordered source directories that may contain axis metadata."""

    directories: list[Path] = []
    seen: set[str] = set()

    for input_name in ("summary", "xmolout", "fort78", "fort76", "fort73", "fort7"):
        raw_path = getattr(args, input_name, None)
        if raw_path is None:
            continue
        input_path = Path(str(raw_path))
        directory = input_path if input_path.is_dir() else input_path.parent
        key = str(directory.resolve())
        if key not in seen:
            seen.add(key)
            directories.append(directory)

    for input_name in ("_snapshot_source_dir", "run_dir", "input"):
        raw_path = getattr(args, input_name, None)
        if raw_path is None:
            continue
        directory = Path(str(raw_path))
        if directory.is_file():
            directory = directory.parent
        key = str(directory.resolve())
        if key not in seen:
            seen.add(key)
            directories.append(directory)
    return directories


def _axis_control_file(args: argparse.Namespace) -> str:
    """Resolve a default control file beside any time-series input source."""

    configured = Path(str(getattr(args, "control", "control")))
    if configured != Path("control"):
        return str(configured)

    for directory in _axis_input_directories(args):
        candidate = directory / "control"
        if candidate.is_file():
            return str(candidate)
    return str(configured)


def _axis_frame_source(args: argparse.Namespace) -> str | None:
    """Resolve an explicit or sibling trajectory used for frame cadence."""

    configured = getattr(args, "frame_source", None)
    if configured:
        return str(configured)

    configured_xmolout = getattr(args, "xmolout", None)
    if configured_xmolout:
        xmolout_path = Path(str(configured_xmolout))
        if xmolout_path.is_dir():
            xmolout_path = xmolout_path / "xmolout"
        if xmolout_path.is_file():
            return str(xmolout_path)

    for directory in _axis_input_directories(args):
        candidate = directory / "xmolout"
        if candidate.is_file():
            return str(candidate)
    return None


def build_plot_payload(command: str, result, args: argparse.Namespace) -> dict[str, object] | None:
    """Build a table-driven plot payload for any dedicated time-series result."""
    table = getattr(result, "table", None)
    if not isinstance(table, pd.DataFrame) or table.empty:
        return None

    y_col = next((name for name in ("coord", "freq", "value") if name in table), None)
    if y_col is None:
        return None
    x_values, xlabel, source_x_col = _plot_axis(table, args)
    working = table.copy()
    working["__plot_x"] = x_values

    group_candidates = (
        ("atom_id", "atom_type", "dim")
        if y_col == "coord"
        else ("molecules",)
        if y_col == "freq"
        else ("component", "field", "quantity", "restraint_index")
    )
    group_cols = [name for name in group_candidates if name in working]
    series: list[dict[str, object]] = []
    if group_cols:
        grouped = working.groupby(group_cols, dropna=False, sort=False)
        for keys, group in grouped:
            key_values = keys if isinstance(keys, tuple) else (keys,)
            label = ", ".join(f"{name}={value}" for name, value in zip(group_cols, key_values))
            series.append(
                {
                    "x": group["__plot_x"].tolist(),
                    "y": pd.to_numeric(group[y_col], errors="coerce").tolist(),
                    "label": label,
                }
            )
    else:
        series.append(
            {
                "x": working["__plot_x"].tolist(),
                "y": pd.to_numeric(working[y_col], errors="coerce").tolist(),
                "label": y_col,
            }
        )
    if not series:
        return None

    title = command.removeprefix("get_").replace("_", " ").title()
    if getattr(args, "plot", None) == "subplot":
        return {
            "plot_type": "multi_subplots",
            "subplots": [[item] for item in series],
            "xlabel": xlabel,
            "ylabel": y_col,
            "title": title,
            "legend": False,
            "grid": getattr(args, "grid", None),
        }
    return {
        "plot_type": "single_plot",
        "series": series,
        "xlabel": xlabel,
        "ylabel": y_col,
        "title": title,
        "legend": len(series) > 1,
    }


def _apply_frame_axis_to_result(
    command: str,
    result: object,
    args: argparse.Namespace,
) -> None:
    """Put the resolved frame coordinate in the table before persistence/export."""

    if str(getattr(args, "xaxis", "iter")) != "frame":
        return
    table = getattr(result, "table", None)
    if not isinstance(table, pd.DataFrame):
        return

    frame_values, _label, _source = _plot_axis(table, args)
    if len(frame_values) != len(table):
        raise ValueError("Resolved frame axis length does not match the result table.")
    converted = table.copy()
    converted["frame_index"] = frame_values
    setattr(result, "table", converted)
    if command == "get_electric_field":
        numeric_frames = pd.to_numeric(converted["frame_index"], errors="coerce").to_numpy(
            dtype=float
        )
        rounded_frames = np.rint(numeric_frames)
        integer_mask = np.isfinite(numeric_frames) & np.isclose(
            numeric_frames,
            rounded_frames,
            rtol=0.0,
            atol=1.0e-9,
        )
        integer_frames = converted.loc[integer_mask].copy()
        integer_frames["frame_index"] = rounded_frames[integer_mask].astype(int)
        setattr(
            result,
            "csv_tables",
            {
                "all_frames": converted,
                "integer_frames": integer_frames,
            },
        )


def run_task(command: str, task_name: str, request: object, args: argparse.Namespace) -> int:
    """Execute and present one dedicated time-series workflow."""
    task_cls = TASK_REGISTRY[task_name]
    result = AnalysisExecutor().run(task_cls(), request, vars(args))
    _apply_frame_axis_to_result(command, result, args)
    present_result(command, result, args, plot_payload_builder=build_plot_payload)
    return 0

