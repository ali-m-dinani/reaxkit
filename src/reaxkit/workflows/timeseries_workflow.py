"""Dispatcher workflow for time-series analyses.

This module implements CLI workflow orchestration for its command family, including argument parsing, request construction, execution dispatch, and result presentation handoff.

**Usage context**

- Command routing: Resolve CLI aliases and normalized command names.
- Task execution: Build request objects and invoke registered tasks.
- Output handling: Forward results to table, plot, export, or report flows.
"""

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
    TrajectoryDisplacementSeriesRequest,
)
from reaxkit.core.runtime.analysis_executor import AnalysisExecutor
from reaxkit.core.utils.frame_utils import parse_frame_indices
from reaxkit.core.registry.analysis_task_registry import TASK_REGISTRY
from reaxkit.core.storage.storage_layout import add_storage_cli_arguments
from reaxkit.presentation.convert import convert_xaxis
from reaxkit.presentation.dispatcher import present_result

ALL_COMMANDS = ("timeseries",)
ALL_LEGACY_COMMANDS = ()

_TRAJECTORY_FIELD_RE = re.compile(
    r"^(?P<kind>trajectory|atom|displacement)(?:\[(?P<ids>[^\]]*)\])?(?:\.(?P<component>[xyzXYZ]{1,3}))?$"
)
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


def _parse_frame_selector(spec) -> list[int] | None:
    """Parse frame selector."""
    return parse_frame_indices(spec)


def _split_csv_tokens(spec: str) -> list[str]:
    """Split csv tokens."""
    return [token.strip() for token in re.split(r"[\s,]+", spec.strip()) if token.strip()]


def _parse_int_tokens(spec: str) -> list[int]:
    """Parse int tokens."""
    return [int(token) for token in _split_csv_tokens(spec)]


def _parse_atom_selector(spec: str | None) -> tuple[int, ...] | None:
    """Parse atom selector."""
    if spec is None:
        return None
    text = str(spec).strip()
    if not text:
        return None
    values = parse_frame_indices(text)
    if values is None:
        return None
    out = [int(v) for v in values]
    for v in out:
        if v <= 0:
            raise ValueError("Atom selectors are 1-based and must be positive integers.")
    return tuple(out)


def _normalize_trajectory_components(component: str | None) -> tuple[str, ...]:
    """Normalize trajectory components."""
    if component is None or not str(component).strip():
        return ("xyz",)
    token = str(component).strip().lower()
    if any(ch not in {"x", "y", "z"} for ch in token):
        raise ValueError(f"Unsupported trajectory/displacement component {component!r}.")
    if len(set(token)) != len(token):
        raise ValueError(f"Duplicate axes are not allowed in component {component!r}.")
    canonical = "".join(ch for ch in "xyz" if ch in token)
    return (canonical,)


def _add_runtime_arguments(parser: argparse.ArgumentParser) -> None:
    """Add runtime arguments."""
    parser.add_argument("--field", default=None, help="Dispatcher field expression. Example: --field temperature, which selects simulation temperature time series.")
    parser.add_argument("--engine", choices=["reaxff", "ams", "lammps"], default=None, help="Engine override. Example: --engine reaxff, which applies ReaxFF-specific loaders.")
    parser.add_argument("--input", default=".", help="Input file or directory for engine resolution. Example: --input runs/job1, which sets base context for file detection.")
    parser.add_argument("--run-dir", default=".", help="Run directory fallback for engine detection. Example: --run-dir runs/job1, which is used as backup lookup path.")
    parser.add_argument("--xmolout", default="xmolout", help="Path to xmolout. Example: --xmolout runs/job1/xmolout, which supplies trajectory coordinate data.")
    parser.add_argument("--summary", default=None, help="Optional summary.txt path. Example: --summary runs/job1/summary.txt, which provides scalar simulation series data.")
    parser.add_argument("--fort7", default="fort.7", help="Path to fort.7. Example: --fort7 runs/job1/fort.7, which provides charge/bond-order source data.")
    parser.add_argument("--fort73", default="fort.73", help="Path to fort.73-style file. Example: --fort73 fort.73, which provides partial-energy time series data.")
    parser.add_argument("--fort76", default="fort.76", help="Path to fort.76. Example: --fort76 fort.76, which provides restraint series data.")
    parser.add_argument("--fort78", default="fort.78", help="Path to fort.78. Example: --fort78 fort.78, which provides electric-field series data.")
    parser.add_argument("--fort57", default="fort.57", help="Path to fort.57. Example: --fort57 fort.57, which provides geometry-optimization series data.")
    parser.add_argument("--eregime", default="eregime.in", help="Path to eregime.in. Example: --eregime eregime.in, which provides imposed field program values.")
    parser.add_argument("--molfra", default="molfra.out", help="Path to molfra.out. Example: --molfra molfra.out, which provides molecular frequency/total series.")
    parser.add_argument("--control", default="control", help="Path to control file for time-axis conversion. Example: --control control, which provides timestep metadata.")
    parser.add_argument("--log", choices=["verbose", "quiet"], default=None, help="Logging level. Example: --log verbose, which prints more runtime details.")
    add_storage_cli_arguments(parser)


def _add_presentation_arguments(parser: argparse.ArgumentParser) -> None:
    """Add presentation arguments."""
    parser.add_argument("--plot", choices=["single", "subplot"], default=None, help="Render a plot. Example: --plot single, which creates one combined chart.")
    parser.add_argument("--show", action="store_true", help="Show the generated plot window. Example: --show, which opens the figure interactively.")
    parser.add_argument("--save", default=None, help="Save the generated plot to a file path. Example: --save temperature.png, which writes the plot image.")
    parser.add_argument("--export", default=None, help="Write the result table to CSV. Example: --export temperature.csv, which saves tabular output.")
    parser.add_argument("--grid", default=None, help="Subplot grid like 2x2 or 2*2. Example: --grid 2x2, which arranges subplot panels in two rows and two columns.")
    parser.add_argument("--xaxis", choices=["iter", "frame", "time"], default="iter", help="X-axis domain. Example: --xaxis time, which converts iterations to physical time when possible.")


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    """Add common arguments."""
    parser.add_argument(
        "--frames",
        nargs="*",
        default=None,
        help='Frame selector syntax. Example: --frames 0:20:2, which selects frames 0,2,4,...,20.',
    )
    parser.add_argument("--every", type=int, default=1, help="Use every Nth selected frame. Example: --every 5, which subsamples selected frames by five.")
    parser.add_argument("--format", choices=["long", "wide"], default="long", help="Trajectory output table format. Example: --format wide, which pivots compatible outputs into wide columns.")
    parser.add_argument("--atoms", default=None, help='Legacy trajectory atom selector. Example: --atoms "1,5,12", which limits trajectory-series extraction to those atom ids.')
    parser.add_argument("--atom-types", nargs="*", default=None, help="Legacy trajectory atom-type selector. Example: --atom-types O H, which limits trajectory-series extraction to oxygen/hydrogen.")
    parser.add_argument("--dims", nargs="*", default=None, choices=["x", "y", "z"], help="Legacy trajectory coordinate dimensions. Example: --dims z, which extracts only z-coordinate series.")
    parser.add_argument("--reference-frame", type=int, default=0, help="Reference frame index used by displacement fields. Example: --reference-frame 10, which subtracts frame 10 coordinates from each selected frame.")
    parser.add_argument("--boxdims", action="store_true", help="Legacy shortcut for cell-dimension extraction from xmolout. Example: --boxdims, which switches to lattice-parameter series mode.")
    parser.add_argument("--cell-fields", nargs="*", default=None, help="Legacy cell-dimension fields. Example: --cell-fields a b c alpha beta gamma, which selects listed lattice fields.")
    parser.add_argument("--field-kind", choices=["applied", "energy", "auto"], default="auto", help="Electric-field group. Example: --field-kind applied, which selects externally applied field channels.")
    parser.add_argument("--dropna-rows", action="store_true", help="Drop rows that are all-NaN across selected restraint fields. Example: --dropna-rows, which removes empty restraint records.")
    parser.add_argument("--include-geo-descriptor", action="store_true", help="Include geo descriptor for geometry optimization data. Example: --include-geo-descriptor, which keeps descriptor annotations in output.")


def _coerce_atom_ids(value) -> tuple[int, ...] | None:
    """Coerce atom ids."""
    if value is None:
        return None
    if isinstance(value, str):
        parsed = _parse_atom_selector(value)
        return parsed
    return tuple(int(v) for v in value)


def _resolve_legacy_xmolout_request(args: argparse.Namespace):
    """Resolve legacy xmolout request."""
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
                frames=frames,
                every=int(args.every),
            ),
        )
    return None


def _resolve_task_and_request(args: argparse.Namespace):
    """Resolve task and request."""
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

    match = _TRAJECTORY_FIELD_RE.match(field)
    if match:
        atom_ids = _parse_atom_selector(match.group("ids"))
        components = _normalize_trajectory_components(match.group("component"))
        kind = str(match.group("kind")).lower()
        if kind == "displacement":
            return (
                "trajectory_displacement_series",
                TrajectoryDisplacementSeriesRequest(
                    atom_ids=atom_ids,
                    dims=components,
                    reference_frame=int(args.reference_frame),
                    frames=frames,
                    every=int(args.every),
                ),
            )
        return (
            "trajectory_coordinate_series",
            TrajectoryCoordinateSeriesRequest(
                atom_ids=atom_ids,
                dims=components,
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
        "Examples: temperature, potential_energy, trajectory[1].x, displacement[1:20].xy, charge[1], "
        "field_z, eregime.field, energy[E_bond], restraint[1], "
        "molecule[H2O], total_molecules, geo_opt.E_pot"
    )


def build_parser(p: argparse.ArgumentParser) -> None:
    """Build parser.

    Execute the workflow function for this command path and return the
    computed result for downstream CLI handling.

    Parameters
    -----
    p : Any
        Function argument.

    Returns
    -----
    None
        Function return value.

    Examples
    -----
    >>> # See workflow CLI usage for concrete examples.
    """
    p.set_defaults(progress=True)
    p.formatter_class = argparse.RawTextHelpFormatter
    p.description = (
        "Dispatcher for time-series and related sequential analyses.\n"
        "This command routes `--field` expressions to the appropriate analysis backend\n"
        "for simulation scalars, trajectory coordinates, charges, fields, energies, restraints,\n"
        "molecular frequencies/totals, and geometry-optimization data.\n\n"
        
        "Examples:\n"
        "  1. Plot simulation scalar series such as temperature:\n"
        "   reaxkit timeseries --field temperature --summary summary.txt --plot single\n\n"
        
        "  2. Plot trajectory/displacement series on time axis:\n"
        "   - getting the trajectory of atoms 1 and 2 in z dimension:\n"
        "       reaxkit timeseries --field trajectory[1,2].z --xaxis time --save atom_z.png\n"
        "   - getting the displacement of atoms 1 to 20 in x and y dimensions with reference frame 0:\n"
        "     [Note] when more than 1 dimension is selected, it finds the magnitude of the combined components (i.e., sqrt(dx^2 + dy^2) in the example below).\n"
        "       reaxkit timeseries --field displacement[1:20].xy --reference-frame 0 --xaxis time --plot single\n\n"
        
        "  3. Export charge series for atom 1:\n"
        "   reaxkit timeseries --field charge[1] --fort7 fort.7 --export charges.csv\n\n"
        
        "  4. Plot molecular frequency/totals series:\n"
        "   reaxkit timeseries --field molecule[H2O,OH] --molfra molfra.out --plot single\n"
        "   reaxkit timeseries --field totals[total_molecules,total_atoms] --molfra molfra.out --plot subplot\n\n"
        
        "  5. Plot restraint/electric-field/energy series:\n"
        "   reaxkit timeseries --field restraint.E_res --fort76 fort.76 --xaxis time --plot single\n"
        "   reaxkit timeseries --field electric_field.E_field_x --fort78 fort.78 --xaxis time --plot single\n"
        "   reaxkit timeseries --field energy.Ebond --fort73 fort.73 --plot single\n\n"
        
        "  6. Plot geometry-optimization results (i.e., energy vs iter):\n"
        "   reaxkit timeseries --field geo_opt.E_pot --fort57 fort.57 --plot single\n"
        "   reaxkit timeseries --field geo_opt.all --fort57 fort.57 --plot subplot\n\n"
    )
    _add_runtime_arguments(p)
    _add_presentation_arguments(p)
    _add_common_arguments(p)


def _series_with_xaxis(command: str, result: TimeSeriesResult, args: argparse.Namespace):
    """Series with xaxis."""
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
    """Result to frame."""
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
    """Prepare result table."""
    if isinstance(result, TimeSeriesResult) and result.table is None:
        plot_series, xlabel = _series_with_xaxis(command, result, args)
        result.table = _result_to_frame(result, xlabel, plot_series)


def _plot_payload(command: str, result, args: argparse.Namespace) -> dict[str, object] | None:
    """Plot payload."""
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

    if command in {"trajectory_coordinate_series", "trajectory_displacement_series"}:
        table = getattr(result, "table", None)
        if table is None or not isinstance(table, pd.DataFrame) or table.empty:
            return None
        if "coord" not in table.columns:
            return None
        if "iter" in table.columns:
            x_col = "iter"
            xlabel = "iter"
        elif "frame_index" in table.columns:
            x_col = "frame_index"
            xlabel = "frame_index"
        else:
            return None

        series_key_cols = [col for col in ("atom_id", "atom_type", "dim") if col in table.columns]
        if not series_key_cols:
            series_key_cols = [x_col]

        series_payload: list[dict[str, object]] = []
        grouped = table.sort_values(series_key_cols + [x_col], kind="stable").groupby(series_key_cols, dropna=False)
        for keys, group in grouped:
            if not isinstance(keys, tuple):
                keys = (keys,)
            label_parts = [f"{col}={val}" for col, val in zip(series_key_cols, keys)]
            label = ", ".join(label_parts) if label_parts else "coord"
            x = pd.to_numeric(group[x_col], errors="coerce").to_numpy(dtype=float)
            y = pd.to_numeric(group["coord"], errors="coerce").to_numpy(dtype=float)
            series_payload.append({"x": x.tolist(), "y": y.tolist(), "label": label})

        if not series_payload:
            return None
        is_displacement = command == "trajectory_displacement_series"
        ylabel = "displacement" if is_displacement else "coord"
        title = "Trajectory Displacement Series" if is_displacement else "Trajectory Coordinate Series"
        if getattr(args, "plot", None) == "subplot":
            return {
                "plot_type": "multi_subplots",
                "subplots": [[s] for s in series_payload],
                "xlabel": xlabel,
                "ylabel": ylabel,
                "title": title,
                "legend": False,
                "grid": getattr(args, "grid", None),
            }
        return {
            "plot_type": "single_plot",
            "series": series_payload,
            "xlabel": xlabel,
            "ylabel": ylabel,
            "title": title,
            "legend": len(series_payload) > 1,
        }

    return None


def run_main(args: argparse.Namespace) -> int:
    """Run main.

    Execute the workflow function for this command path and return the
    computed result for downstream CLI handling.

    Parameters
    -----
    args : Any
        Function argument.

    Returns
    -----
    int
        Function return value.

    Examples
    -----
    >>> # See workflow CLI usage for concrete examples.
    """
    command, request = _resolve_task_and_request(args)
    task_cls = TASK_REGISTRY[command]

    executor = AnalysisExecutor()
    result = executor.run(task_cls(), request, vars(args))
    _prepare_result_table(command, result, args)
    present_result(command, result, args, plot_payload_builder=_plot_payload)
    return 0


def register_tasks(subparsers: argparse._SubParsersAction) -> None:
    """Register tasks.

    Execute the workflow function for this command path and return the
    computed result for downstream CLI handling.

    Parameters
    -----
    subparsers : Any
        Function argument.

    Returns
    -----
    None
        Function return value.

    Examples
    -----
    >>> # See workflow CLI usage for concrete examples.
    """
    _ = subparsers
    return
