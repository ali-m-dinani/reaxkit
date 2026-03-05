"""Dispatcher workflow for time-series analyses."""

from __future__ import annotations

import argparse
import re

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
    TimeSeriesResult,
    TrajectoryCoordinateSeriesRequest,
)
from reaxkit.core.analysis_executor import AnalysisExecutor
from reaxkit.core.frame_utils import parse_frames
from reaxkit.core.analysis_task_registry import TASK_REGISTRY
from reaxkit.core.storage_layout import add_storage_cli_arguments
from reaxkit.presentation.convert import convert_xaxis
from reaxkit.presentation.dispatcher import present_result

_ATOM_FIELD_RE = re.compile(r"^atom\[(?P<ids>[0-9,\s]+)\]\.(?P<axis>[xyzXYZ])$")
_CHARGE_FIELD_RE = re.compile(r"^(?:charge|q)\[(?P<ids>[0-9,\s]+)\]$")
_CELL_FIELD_RE = re.compile(r"^(?:cell(?:_dimensions)?|lattice)\[(?P<fields>[A-Za-z0-9_,\s]+)\]$")
_EFIELD_FIELD_RE = re.compile(r"^(?:electric_field|efield)\[(?P<components>[A-Za-z0-9_,\s]+)\]$")
_EFIELD_DOT_FIELD_RE = re.compile(r"^(?:electric_field|efield)\.(?P<component>[A-Za-z0-9_]+)$")
_EREGIME_FIELD_RE = re.compile(r"^eregime\.(?P<field>[A-Za-z0-9_]+)$")
_PARTIAL_ENERGY_RE = re.compile(r"^(?:partial_energy|energy)\[(?P<components>[A-Za-z0-9_,\s]+)\]$")
_PARTIAL_ENERGY_FIELD_RE = re.compile(r"^(?:partial_energy|energy)\.(?P<component>[A-Za-z0-9_]+)$")
_RESTRAINT_INDEX_RE = re.compile(r"^restraint\[(?P<index>[0-9]+)\]$")
_RESTRAINT_FIELD_RE = re.compile(r"^restraint\.(?P<field>[A-Za-z0-9_]+)$")
_MOLECULE_FREQ_RE = re.compile(r"^(?:molecule|freq)\[(?P<molecules>[A-Za-z0-9_,\s]+)\]$")
_MOLECULAR_TOTALS_RE = re.compile(r"^(?:totals|molecular_totals)\[(?P<quantities>[A-Za-z0-9_,\s]+)\]$")
_GEO_OPT_LIST_RE = re.compile(r"^(?:geo_opt|geometry_optimization)\[(?P<cols>[A-Za-z0-9_,\s]+)\]$")
_GEO_OPT_FIELD_RE = re.compile(r"^(?:geo_opt|geometry_optimization)\.(?P<col>[A-Za-z0-9_]+)$")

_SIMULATION_FIELD_MAP = {
    "potential_energy": "potential_energy",
    "energy": "potential_energy",
    "e_pot": "potential_energy",
    "epot": "potential_energy",
    "num_of_atoms": "num_of_atoms",
    "num_atoms": "num_of_atoms",
    "atoms": "num_of_atoms",
    "volume": "volume",
    "v": "volume",
    "temperature": "temperature",
    "temp": "temperature",
    "t": "temperature",
    "pressure": "pressure",
    "p": "pressure",
    "density": "density",
    "d": "density",
    "elapsed_time": "elapsed_time",
    "elap_time": "elapsed_time",
    "time_elapsed": "elapsed_time",
    "a": "a",
    "b": "b",
    "c": "c",
    "alpha": "alpha",
    "beta": "beta",
    "gamma": "gamma",
}

_MOLECULAR_TOTALS = {"total_molecules", "total_atoms", "total_molecular_mass"}
_DIRECT_EFIELD_COMPONENT_RE = re.compile(r"^(?:field_[xyz]|E_field(?:_[xyz])?)$")
_GEO_OPT_ALL_COLUMNS = ("iter", "E_pot", "T", "T_set", "RMSG", "nfc")


def _parse_frame_selector(spec: str | None) -> list[int] | None:
    if not spec:
        return None
    sel = parse_frames(spec)
    if sel is None:
        return None
    if isinstance(sel, slice):
        start = 0 if sel.start is None else int(sel.start)
        stop = int(sel.stop) if sel.stop is not None else start
        step = 1 if sel.step is None else int(sel.step)
        return list(range(start, stop, step))
    return [int(v) for v in sel]


def _split_csv_tokens(spec: str) -> list[str]:
    return [token.strip() for token in re.split(r"[\s,]+", spec.strip()) if token.strip()]


def _parse_int_tokens(spec: str) -> list[int]:
    return [int(token) for token in _split_csv_tokens(spec)]


def _add_runtime_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--field", default=None, help="Dispatcher field expression, for example temperature, atom[1].x, charge[1], geo_opt.E_pot")
    parser.add_argument("--engine", choices=["reaxff", "ams", "lammps"], default=None)
    parser.add_argument("--input", default=".", help="Input file or directory for engine resolution")
    parser.add_argument("--run-dir", default=".", help="Run directory fallback for engine detection")
    parser.add_argument("--xmolout", default="xmolout", help="Path to xmolout")
    parser.add_argument("--summary", default=None, help="Optional summary.txt path")
    parser.add_argument("--fort7", default="fort.7", help="Path to fort.7")
    parser.add_argument("--fort73", default="fort.73", help="Path to fort.73-style file")
    parser.add_argument("--fort76", default="fort.76", help="Path to fort.76")
    parser.add_argument("--fort78", default="fort.78", help="Path to fort.78")
    parser.add_argument("--fort57", default="fort.57", help="Path to fort.57")
    parser.add_argument("--eregime", default="eregime.in", help="Path to eregime.in")
    parser.add_argument("--molfra", default="molfra.out", help="Path to molfra.out")
    parser.add_argument("--control", default="control", help="Path to control file for time-axis conversion")
    parser.add_argument("--log", choices=["verbose", "quiet"], default=None, help="Logging level")
    add_storage_cli_arguments(parser)


def _add_presentation_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--plot", choices=["single", "subplot"], default=None, help="Render a plot")
    parser.add_argument("--show", action="store_true", help="Show the generated plot window")
    parser.add_argument("--save", default=None, help="Save the generated plot to a file path")
    parser.add_argument("--export", default=None, help="Write the result table to CSV")
    parser.add_argument("--grid", default=None, help="Subplot grid like 2x2 or 2*2")
    parser.add_argument("--xaxis", choices=["iter", "frame", "time"], default="iter", help="X-axis domain")


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--frames", default=None, help='Frame selector: "0,10,20" or "0:100:5"')
    parser.add_argument("--every", type=int, default=1, help="Use every Nth selected frame")
    parser.add_argument("--format", choices=["long", "wide"], default="long", help="Trajectory output table format")
    parser.add_argument("--atoms", default=None, help='Legacy trajectory atom selector, for example "1,5,12"')
    parser.add_argument("--atom-types", nargs="*", default=None, help="Legacy trajectory atom-type selector")
    parser.add_argument("--dims", nargs="*", default=None, choices=["x", "y", "z"], help="Legacy trajectory coordinate dimensions")
    parser.add_argument("--boxdims", action="store_true", help="Legacy shortcut for cell-dimension extraction from xmolout")
    parser.add_argument("--cell-fields", nargs="*", default=None, help="Legacy cell-dimension fields, for example a b c alpha beta gamma")
    parser.add_argument("--field-kind", choices=["applied", "energy", "auto"], default="auto", help="Electric-field group")
    parser.add_argument("--dropna-rows", action="store_true", help="Drop rows that are all-NaN across selected restraint fields")
    parser.add_argument("--include-geo-descriptor", action="store_true", help="Include geo descriptor for geometry optimization data")


def _coerce_atom_ids(value) -> tuple[int, ...] | None:
    if value is None:
        return None
    if isinstance(value, str):
        return tuple(_parse_int_tokens(value))
    return tuple(int(v) for v in value)


def _resolve_legacy_xmolout_request(args: argparse.Namespace):
    frames = _parse_frame_selector(args.frames)
    if bool(args.boxdims) or bool(args.cell_fields):
        fields = tuple(args.cell_fields or ("a", "b", "c", "alpha", "beta", "gamma"))
        return (
            "cell_dimensions",
            CellDimensionsRequest(
                fields=fields,
                frames=frames,
                every=int(args.every),
            ),
        )

    atom_ids = _coerce_atom_ids(args.atoms)
    atom_types = tuple(args.atom_types) if args.atom_types else None
    dims = tuple(args.dims or ("x", "y", "z"))
    if atom_ids is not None or atom_types is not None or args.dims is not None:
        return (
            "trajectory_coordinate_series",
            TrajectoryCoordinateSeriesRequest(
                atom_ids=atom_ids,
                atom_types=atom_types,
                dims=dims,
                format=args.format,
                frames=frames,
                every=int(args.every),
            ),
        )
    return None


def _resolve_task_and_request(args: argparse.Namespace):
    if args.field is None:
        legacy = _resolve_legacy_xmolout_request(args)
        if legacy is not None:
            return legacy
        raise ValueError(
            "Provide --field, or use legacy xmolout-style flags such as "
            "--atoms/--atom-types/--dims or --boxdims/--cell-fields."
        )

    field = str(args.field).strip()
    frames = _parse_frame_selector(args.frames)
    field_key = field.lower()

    match = _ATOM_FIELD_RE.match(field)
    if match:
        return (
            "trajectory_coordinate_series",
            TrajectoryCoordinateSeriesRequest(
                atom_ids=_parse_int_tokens(match.group("ids")),
                dims=(match.group("axis").lower(),),
                format=args.format,
                frames=frames,
                every=int(args.every),
            ),
        )

    match = _CHARGE_FIELD_RE.match(field)
    if match:
        return (
            "charge_series",
            ChargeSeriesRequest(
                atom_ids=tuple(_parse_int_tokens(match.group("ids"))),
                frames=frames,
                every=int(args.every),
            ),
        )

    match = _CELL_FIELD_RE.match(field)
    if match:
        return (
            "cell_dimensions",
            CellDimensionsRequest(
                fields=tuple(_split_csv_tokens(match.group("fields"))),
                frames=frames,
                every=int(args.every),
            ),
        )

    match = _EFIELD_FIELD_RE.match(field)
    if match:
        return (
            "electric_field_series",
            ElectricFieldSeriesRequest(
                components=tuple(_split_csv_tokens(match.group("components"))),
                field_kind=args.field_kind,
                frames=frames,
                every=int(args.every),
            ),
        )

    match = _EFIELD_DOT_FIELD_RE.match(field)
    if match:
        return (
            "electric_field_series",
            ElectricFieldSeriesRequest(
                components=(match.group("component"),),
                field_kind=args.field_kind,
                frames=frames,
                every=int(args.every),
            ),
        )

    if _DIRECT_EFIELD_COMPONENT_RE.match(field):
        return (
            "electric_field_series",
            ElectricFieldSeriesRequest(
                components=(field,),
                field_kind=args.field_kind,
                frames=frames,
                every=int(args.every),
            ),
        )

    match = _EREGIME_FIELD_RE.match(field)
    if match:
        return (
            "eregime_series",
            EregimeSeriesRequest(
                field=match.group("field"),
                frames=frames,
                every=int(args.every),
            ),
        )

    match = _PARTIAL_ENERGY_RE.match(field)
    if match:
        return (
            "partial_energy_series",
            PartialEnergySeriesRequest(
                components=tuple(_split_csv_tokens(match.group("components"))),
                frames=frames,
                every=int(args.every),
            ),
        )

    match = _PARTIAL_ENERGY_FIELD_RE.match(field)
    if match:
        component = str(match.group("component")).strip()
        components = None if component.lower() == "all" else (component,)
        return (
            "partial_energy_series",
            PartialEnergySeriesRequest(
                components=components,
                frames=frames,
                every=int(args.every),
            ),
        )

    match = _RESTRAINT_INDEX_RE.match(field)
    if match:
        return (
            "restraint_series",
            RestraintSeriesRequest(
                restraint_index=int(match.group("index")),
                dropna_rows=bool(args.dropna_rows),
                frames=frames,
                every=int(args.every),
            ),
        )

    match = _RESTRAINT_FIELD_RE.match(field)
    if match:
        return (
            "restraint_series",
            RestraintSeriesRequest(
                fields=(match.group("field"),),
                dropna_rows=bool(args.dropna_rows),
                frames=frames,
                every=int(args.every),
            ),
        )

    match = _MOLECULE_FREQ_RE.match(field)
    if match:
        return (
            "molecular_frequency_series",
            MolecularFrequencySeriesRequest(
                molecules=tuple(_split_csv_tokens(match.group("molecules"))),
                frames=frames,
                every=int(args.every),
            ),
        )

    match = _MOLECULAR_TOTALS_RE.match(field)
    if match:
        return (
            "molecular_totals_series",
            MolecularTotalsSeriesRequest(
                quantities=tuple(_split_csv_tokens(match.group("quantities"))),
                xaxis=args.xaxis,
                control_file=args.control,
                frames=frames,
                every=int(args.every),
            ),
        )

    if field_key in _MOLECULAR_TOTALS:
        return (
            "molecular_totals_series",
            MolecularTotalsSeriesRequest(
                quantities=(field_key,),
                xaxis=args.xaxis,
                control_file=args.control,
                frames=frames,
                every=int(args.every),
            ),
        )

    match = _GEO_OPT_LIST_RE.match(field)
    if match:
        return (
            "geometry_optimization_data",
            GeometryOptimizationRequest(
                cols=tuple(_split_csv_tokens(match.group("cols"))),
                include_geo_descriptor=bool(args.include_geo_descriptor),
            ),
        )

    match = _GEO_OPT_FIELD_RE.match(field)
    if match:
        col = match.group("col")
        if str(col).strip().lower() == "all":
            return (
                "geometry_optimization_data",
                GeometryOptimizationRequest(
                    cols=_GEO_OPT_ALL_COLUMNS,
                    include_geo_descriptor=bool(args.include_geo_descriptor),
                ),
            )
        return (
            "geometry_optimization_data",
            GeometryOptimizationRequest(
                cols=(col,),
                include_geo_descriptor=bool(args.include_geo_descriptor),
            ),
        )

    sim_field = _SIMULATION_FIELD_MAP.get(field_key)
    if sim_field is not None:
        return (
            "simulation_series",
            SimulationScalarSeriesRequest(
                field=sim_field,
                frames=frames,
                every=int(args.every),
            ),
        )

    raise ValueError(
        f"Unsupported field {field!r}. "
        "Examples: temperature, potential_energy, atom[1].x, charge[1], "
        "field_z, eregime.field, energy[E_bond], restraint[1], "
        "molecule[H2O], total_molecules, geo_opt.E_pot"
    )


def build_parser(p: argparse.ArgumentParser) -> None:
    p.formatter_class = argparse.RawTextHelpFormatter
    p.description = (
        "Dispatcher for time-series and related sequential analyses.\n\n"
        "Examples:\n"
        "  reaxkit timeseries --field temperature --summary summary.txt --plot single\n"
        "  reaxkit timeseries --field atom[1,2].z --xaxis time --save atom_z.png\n"
        "  reaxkit timeseries --field charge[1] --fort7 fort.7 --export charges.csv\n"
        "  reaxkit timeseries --field molecule[H2O,OH] --molfra molfra.out --plot single\n"
        "  reaxkit timeseries --field totals[total_molecules,total_atoms] --molfra molfra.out --plot subplot\n"
        "  reaxkit timeseries --field restraint.E_res --fort76 fort.76 --xaxis time --plot single\n"
        "  reaxkit timeseries --field restraint[2] --fort76 fort.76 --xaxis frame --export restraint2.csv\n"
        "  reaxkit timeseries --field electric_field.E_field_x --fort78 fort.78 --xaxis time --plot single\n"
        "  reaxkit timeseries --field energy.Ebond --fort73 fort.73 --plot single\n"
        "  reaxkit timeseries --field energy.all --fort73 energylog --plot subplot\n"
        "  reaxkit timeseries --field geo_opt.E_pot --fort57 fort.57 --plot single\n"
        "  reaxkit timeseries --field geo_opt.all --fort57 fort.57 --plot subplot\n\n"
        "for legacy CLI commands use\n"
        "  reaxkit timeseries --atoms 1,5,12 --dims z --xaxis time --save atom_z.png\n"
        "  reaxkit timeseries --boxdims --cell-fields a b c --xaxis iter --plot subplot\n"
    )
    _add_runtime_arguments(p)
    _add_presentation_arguments(p)
    _add_common_arguments(p)


def _series_with_xaxis(command: str, result: TimeSeriesResult, args: argparse.Namespace):
    if command == "molecular_totals_series":
        xlabel = str(result.x_label)
        out = []
        for series in result.series:
            out.append(
                {
                    "x": np.asarray(series.x).tolist(),
                    "y": np.asarray(series.y, dtype=float).tolist(),
                    "label": series.label,
                }
            )
        return out, xlabel

    meta = result.metadata or {}
    iterations = np.asarray(meta.get("iterations", np.empty((0,), dtype=int)), dtype=int)
    if iterations.size == 0 and result.series:
        iterations = np.asarray(result.series[0].x, dtype=int)
    xvals, xlabel = convert_xaxis(iterations, args.xaxis, control_file=args.control)
    out = []
    for series in result.series:
        out.append({"x": np.asarray(xvals).tolist(), "y": np.asarray(series.y, dtype=float).tolist(), "label": series.label})
    return out, xlabel


def _result_to_frame(result: TimeSeriesResult, xlabel: str, plot_series: list[dict[str, object]]) -> pd.DataFrame:
    if not result.series:
        return pd.DataFrame(columns=[xlabel])

    if len(result.series) == 1:
        series = result.series[0]
        xvals = np.asarray(plot_series[0]["x"], dtype=float) if plot_series else np.asarray(series.x)
        return pd.DataFrame({xlabel: xvals, series.label: np.asarray(series.y, dtype=float)})

    same_x = all(np.array_equal(np.asarray(s.x), np.asarray(result.series[0].x)) for s in result.series)
    if same_x and plot_series:
        out = pd.DataFrame({xlabel: np.asarray(plot_series[0]["x"], dtype=float)})
        for series in result.series:
            out[series.label] = np.asarray(series.y, dtype=float)
        return out

    rows = []
    for payload in plot_series:
        for xv, yv in zip(np.asarray(payload["x"]), np.asarray(payload["y"], dtype=float)):
            rows.append({xlabel: xv, "value": yv, "label": payload["label"]})
    return pd.DataFrame(rows)


def _prepare_result_table(command: str, result, args: argparse.Namespace):
    if isinstance(result, TimeSeriesResult) and result.table is None:
        plot_series, xlabel = _series_with_xaxis(command, result, args)
        result.table = _result_to_frame(result, xlabel, plot_series)


def _plot_payload(command: str, result, args: argparse.Namespace) -> dict[str, object] | None:
    if isinstance(result, TimeSeriesResult):
        if not result.series:
            return None
        plot_series, xlabel = _series_with_xaxis(command, result, args)
        title = f"{result.y_label} vs {xlabel}"
        if getattr(args, "plot", None) == "subplot":
            return {
                "plot_type": "multi_subplots",
                "subplots": [[series] for series in plot_series],
                "xlabel": xlabel,
                "ylabel": result.y_label,
                "title": title,
                "legend": False,
                "grid": getattr(args, "grid", None),
            }
        return {
            "plot_type": "single_plot",
            "series": plot_series,
            "xlabel": xlabel,
            "ylabel": result.y_label,
            "title": title,
            "legend": len(plot_series) > 1,
        }

    if command == "geometry_optimization_data":
        table = result.table
        if table.empty or "iter" not in table.columns:
            return None
        xvals, xlabel = convert_xaxis(table["iter"].to_numpy(dtype=int), args.xaxis, control_file=args.control)
        y_cols = [col for col in table.columns if col not in {"iter", "geo_descriptor"}]
        if not y_cols:
            return None
        series = []
        for col in y_cols:
            y = pd.to_numeric(table[col], errors="coerce").to_numpy(dtype=float)
            series.append({"x": np.asarray(xvals).tolist(), "y": y.tolist(), "label": str(col)})
        if getattr(args, "plot", None) == "subplot":
            return {
                "plot_type": "multi_subplots",
                "subplots": [[item] for item in series],
                "xlabel": xlabel,
                "ylabel": "geometry_optimization",
                "title": "Geometry Optimization Data",
                "legend": False,
                "grid": getattr(args, "grid", None),
            }
        return {
            "plot_type": "single_plot",
            "series": series,
            "xlabel": xlabel,
            "ylabel": "geometry_optimization",
            "title": "Geometry Optimization Data",
            "legend": len(series) > 1,
        }

    return None


def run_main(args: argparse.Namespace) -> int:
    command, request = _resolve_task_and_request(args)
    task_cls = TASK_REGISTRY[command]

    executor = AnalysisExecutor()
    result = executor.run(task_cls(), request, vars(args))
    _prepare_result_table(command, result, args)
    present_result(command, result, args, plot_payload_builder=_plot_payload)
    return 0


def register_tasks(subparsers: argparse._SubParsersAction) -> None:
    _ = subparsers
    return
