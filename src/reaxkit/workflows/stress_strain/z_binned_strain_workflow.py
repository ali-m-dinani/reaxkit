"""CLI workflow for z-binned top/bottom and deformation-gradient strain.

This module maps command-line selections into typed stress/strain requests,
executes registered analyzers through the standard runtime, and hands tabular
or profile-plot output to the presentation layer. Scientific computation stays
in `reaxkit.analysis.stress_strain`.

**Usage context**

- Direct commands: Analyze trajectory files from the ReaxKit CLI.
- Output handling: Persist results, export CSV, and render z profiles.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

import pandas as pd

from reaxkit.analysis import stress_strain as _stress_strain_tasks  # noqa: F401
from reaxkit.analysis.stress_strain.z_binned_deformation_gradient_strain import (
    ENGINEERING_SHEAR_COLUMNS,
    NORMAL_STRAIN_COLUMNS,
    PRINCIPAL_STRAIN_COLUMNS,
    TENSOR_SHEAR_COLUMNS,
    ZBinnedDeformationGradientStrainRequest,
)
from reaxkit.analysis.stress_strain.z_binned_strain_using_top_bottom_atoms import (
    ZBinnedTopBottomStrainRequest,
)
from reaxkit.core.registry.analysis_task_registry import TASK_REGISTRY
from reaxkit.core.resolve.command_alias_resolver import resolve_command_name
from reaxkit.core.runtime.analysis_executor import AnalysisExecutor
from reaxkit.core.storage.storage_layout import add_storage_cli_arguments
from reaxkit.core.utils.frame_utils import parse_frame_indices
from reaxkit.presentation.dispatcher import present_result
from reaxkit.presentation.stress_strain import (
    generate_deformation_gradient_plots,
    generate_top_bottom_plots,
)

ALL_COMMANDS = (
    "get_z_binned_top_bottom_strain",
    "get_z_binned_deformation_gradient_strain",
)
ALL_LEGACY_COMMANDS = (
    "z_binned_strain_using_top_bottom_atoms",
    "z-binned-strain-using-top-bottom-atoms",
    "z_binned_deformation_gradient_strain",
    "z-binned-deformation-gradient-strain",
)
COMMAND_ALIASES = {
    "get_z_binned_top_bottom_strain": ALL_LEGACY_COMMANDS[:2],
    "get_z_binned_deformation_gradient_strain": ALL_LEGACY_COMMANDS[2:],
}


def _build_top_bottom_request(args: argparse.Namespace) -> ZBinnedTopBottomStrainRequest:
    """Build a robust top/bottom strain request from CLI arguments."""
    return ZBinnedTopBottomStrainRequest(
        z_bins=args.z_bins,
        atom_types=args.atom_types,
        bin_range=args.bin_range,
        n_extreme_atoms=args.top_bottom_count,
        selected_frames=parse_frame_indices(args.frames),
        every=args.every,
        unwrap=args.unwrap,
        periodic=args.periodic,
        zero_tolerance=args.zero_tolerance,
    )


def _build_deformation_gradient_request(args: argparse.Namespace) -> ZBinnedDeformationGradientStrainRequest:
    """Build a deformation-gradient strain request from CLI arguments."""
    return ZBinnedDeformationGradientStrainRequest(
        z_bins=args.z_bins,
        atom_types=args.atom_types,
        bin_range=args.bin_range,
        minimum_atoms=args.minimum_atoms,
        max_condition_number=args.max_condition_number,
        selected_frames=parse_frame_indices(args.frames),
        every=args.every,
        unwrap=args.unwrap,
        periodic=args.periodic,
    )


REQUEST_BUILDERS: dict[str, Callable[[argparse.Namespace], object]] = {
    "get_z_binned_top_bottom_strain": _build_top_bottom_request,
    "get_z_binned_deformation_gradient_strain": _build_deformation_gradient_request,
}


def _add_runtime_arguments(parser: argparse.ArgumentParser) -> None:
    """Add engine, input, logging, and storage arguments."""
    parser.add_argument("--engine", choices=["reaxff", "ams", "lammps"], default=None, help="Engine override. Example: --engine reaxff, which uses ReaxFF trajectory loading.")
    parser.add_argument("--run-dir", "--dir", dest="run_dir", default=".", help="Fallback directory for input detection. Example: --run-dir runs/slab, which searches that simulation directory.")
    parser.add_argument("--xmolout", "--file", dest="xmolout", default=None, help="Trajectory input path. Example: --xmolout runs/slab/xmolout, which reads that coordinate trajectory.")
    parser.add_argument("--log", choices=["verbose", "quiet"], default="quiet", help="Logging detail. Example: --log verbose, which prints additional runtime progress.")
    add_storage_cli_arguments(parser)


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    """Add selections shared by both strain methods."""
    parser.add_argument("--atom-types", nargs="*", default=["Al", "N"], help="Elements included in bin fits. Example: --atom-types Al N, which excludes all other elements.")
    parser.add_argument("--z-bins", type=int, required=True, help="Number of equal-width z bins. Example: --z-bins 20, which creates twenty through-thickness regions.")
    parser.add_argument("--bin-range", choices=["reference", "current"], default="reference", help="How z-bin edges are defined. Example: --bin-range reference, which fixes edges from frame zero.")
    parser.add_argument("--frames", nargs="*", default=None, help="Frame selector syntax. Example: --frames 0:101:10, which reports frames 0 through 100 every ten frames.")
    parser.add_argument("--every", type=int, default=1, help="Stride over the selected frame list. Example: --every 5, which keeps every fifth selected frame.")
    parser.add_argument("--unwrap", action=argparse.BooleanOptionalAction, default=True, help="Cumulatively unwrap selected periodic axes. Example: --no-unwrap, which analyzes raw wrapped coordinates.")
    parser.add_argument("--wrapped", action="store_false", dest="unwrap", help="Standalone-compatible alias for --no-unwrap.")
    parser.add_argument("--periodic", choices=["none", "x", "y", "z", "xy", "xz", "yz", "xyz"], default="xyz", help="Axes treated as periodic during unwrapping. Example: --periodic xy, which leaves slab-normal z unwrapped as-is.")
    parser.add_argument("--plot", choices=["single", "subplot"], default=None, help="Render z profiles. Example: --plot subplot, which creates one panel per selected frame.")
    parser.add_argument("--show", action="store_true", help="Show the generated plot interactively. Example: --show, which opens a plot window after analysis.")
    parser.add_argument("--save", default=None, help="Save the generated plot. Example: --save strain.png, which writes a combined profile figure.")
    parser.add_argument("--export", default=None, help="Export the result table to CSV. Example: --export strain.csv, which writes all frame/bin rows.")
    parser.add_argument("--grid", default=None, help="Subplot grid dimensions. Example: --grid 2x3, which arranges six frame panels.")
    parser.add_argument("--gen-plots", action="store_true", help="Generate all four standalone-compatible frame/bin plot directories.")
    parser.add_argument("--y-scale", choices=["global", "frame"], default="global", help="Batch-plot scaling. Global keeps axes consistent across each complete plot family.")
    parser.add_argument("--plot-every", type=int, default=1, help="Generate every Nth frame plot; bin-history plots always include every bin.")
    parser.add_argument("--dpi", type=int, default=180, help="Resolution of batch PNG plots.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Batch plot root. Defaults to a method-specific directory beside xmolout.")


def build_parser(parser: argparse.ArgumentParser, *, command: str) -> argparse.ArgumentParser:
    """Configure a parser for one z-binned strain command.

    Parameters
    -----
    parser : argparse.ArgumentParser
        Parser instance to configure.
    command : str
        Canonical command or supported alias.

    Returns
    -----
    argparse.ArgumentParser
        Configured command parser.

    Examples
    -----
    `build_parser(argparse.ArgumentParser(), command="get_z_binned_top_bottom_strain")`
    returns a parser with bin, frame, PBC, and output controls.
    """
    canonical = resolve_command_name(command, task_names=ALL_COMMANDS, aliases=COMMAND_ALIASES)
    parser.set_defaults(command=canonical, progress=True)
    parser.formatter_class = argparse.RawTextHelpFormatter
    _add_runtime_arguments(parser)
    _add_common_arguments(parser)

    if canonical == "get_z_binned_top_bottom_strain":
        parser.description = (
            "Compute robust normal strain from top/bottom coordinate spans in z bins.\n"
            "Each span averages multiple coordinate extrema and is compared with frame zero.\n\n"
            "Examples:\n"
            "  1. Analyze an xy-periodic slab:\n"
            "   reaxkit get_z_binned_top_bottom_strain --xmolout xmolout --atom-types Al N --z-bins 20 --periodic xy --gen-plots --y-scale global\n\n"
            "  2. Plot selected frames with larger extreme sets:\n"
            "   reaxkit get_z_binned_top_bottom_strain --z-bins 10 --frames 0:101:10 --top-bottom-count 8 --component strain_xx --plot subplot"
        )
        parser.add_argument("--top-bottom-count", "--n-extreme-atoms", dest="top_bottom_count", type=int, default=5, help="Atoms averaged at each coordinate extreme. Example: --top-bottom-count 8, which averages eight top and eight bottom atoms.")
        parser.add_argument("--zero-tolerance", type=float, default=1.0e-12, help="Smallest usable frame-zero span. Example: --zero-tolerance 1e-10, which leaves strain blank for smaller baselines.")
        parser.add_argument("--component", choices=["strain_xx", "strain_yy", "strain_zz", "span_change_x", "span_change_y", "span_change_z"], default="strain_zz", help="Quantity plotted against z. Example: --component strain_xx, which plots x-normal strain profiles.")
    elif canonical == "get_z_binned_deformation_gradient_strain":
        parser.description = (
            "Fit affine deformation gradients and Green-Lagrange strain in z bins.\n"
            "Frame-zero atom groups define material regions and fit diagnostics are retained.\n\n"
            "Examples:\n"
            "  1. Analyze an xy-periodic slab:\n"
            "   reaxkit get_z_binned_deformation_gradient_strain --xmolout xmolout --atom-types Al N --z-bins 20 --periodic xy --gen-plots --y-scale global\n\n"
            "  2. Plot engineering shear profiles:\n"
            "   reaxkit get_z_binned_deformation_gradient_strain --z-bins 10 --frames 0 20 40 --component gamma_xy --plot single"
        )
        parser.add_argument("--minimum-atoms", type=int, default=4, help="Minimum atoms required for a 3D affine fit. Example: --minimum-atoms 8, which leaves smaller bins undefined.")
        parser.add_argument("--max-condition-number", type=float, default=1.0e12, help="Largest accepted reference-fit condition number. Example: --max-condition-number 1e8, which rejects less stable bin geometries.")
        parser.add_argument("--component", choices=[*NORMAL_STRAIN_COLUMNS, *TENSOR_SHEAR_COLUMNS, *ENGINEERING_SHEAR_COLUMNS, *PRINCIPAL_STRAIN_COLUMNS, "J", "volumetric_change", "fit_rmse"], default="strain_zz", help="Quantity plotted against z. Example: --component gamma_xy, which plots engineering xy shear.")
    else:  # pragma: no cover - guarded by alias resolution
        raise KeyError(f"Unsupported stress/strain command '{canonical}'.")
    return parser


def _plot_payload(command: str, result, args: argparse.Namespace) -> dict[str, object] | None:
    """Build a validated z-profile plot payload for presentation."""
    table = result.table
    component = str(args.component)
    required = {"frame", "bin_mean_z", component}
    if table.empty or not required.issubset(table.columns):
        return None
    series: list[dict[str, object]] = []
    subplots: list[list[dict[str, object]]] = []
    for frame, subset in table.groupby("frame", sort=True):
        subset = subset.sort_values("bin_mean_z")
        finite = pd.to_numeric(subset[component], errors="coerce").notna()
        if not finite.any():
            continue
        item = {
            "x": subset.loc[finite, "bin_mean_z"].tolist(),
            "y": subset.loc[finite, component].tolist(),
            "label": f"frame {int(frame)}",
        }
        series.append(item)
        subplots.append([item])
    if not series:
        return None
    title = command.replace("get_", "").replace("_", " ").title()
    if args.plot == "subplot":
        return {"plot_type": "multi_subplots", "subplots": subplots, "title": title, "xlabel": "Bin mean z", "ylabel": component, "legend": False, "grid": args.grid}
    return {"plot_type": "single_plot", "series": series, "title": title, "xlabel": "Bin mean z", "ylabel": component, "legend": len(series) > 1}


def run_main(command: str, args: argparse.Namespace) -> int:
    """Execute a z-binned strain workflow end-to-end.

    Parameters
    -----
    command : str
        Canonical command or supported alias.
    args : argparse.Namespace
        Parsed workflow arguments.

    Returns
    -----
    int
        Zero after successful analysis and presentation.

    Examples
    -----
    `run_main("get_z_binned_top_bottom_strain", args)` returns `0` after
    executing the registered analyzer and persisting its results.
    """
    canonical = resolve_command_name(command, task_names=ALL_COMMANDS, aliases=COMMAND_ALIASES)
    if canonical not in REQUEST_BUILDERS:
        raise KeyError(f"Unsupported stress/strain command '{canonical}'.")
    request = REQUEST_BUILDERS[canonical](args)
    runtime_args = vars(args).copy()
    # Cumulative minimum-image unwrapping needs every intermediate frame from
    # frame zero, even when the result reports only a sparse frame selection.
    # Keep the public selection in the typed request and prevent the engine
    # adapter from treating the CLI flag as a partial-load instruction.
    runtime_args["frames"] = None
    result = AnalysisExecutor().run(TASK_REGISTRY[canonical](), request, runtime_args)
    present_result(canonical, result, args, plot_payload_builder=_plot_payload)
    if args.gen_plots:
        output_dir = _batch_output_directory(canonical, args)
        generator = (
            generate_top_bottom_plots
            if canonical == "get_z_binned_top_bottom_strain"
            else generate_deformation_gradient_plots
        )
        plots = generator(
            result.table,
            output_dir,
            dpi=args.dpi,
            plot_every=args.plot_every,
            y_scale=args.y_scale,
        )
        print(f"Wrote {len(plots):,} batch plots into four plot-family folders in {output_dir}")
    return 0


def _batch_output_directory(command: str, args: argparse.Namespace) -> Path:
    """Resolve the standalone-compatible batch plot output root."""
    if args.output_dir is not None:
        return Path(args.output_dir).resolve()
    if args.xmolout:
        source = Path(args.xmolout).resolve()
        parent = source if source.is_dir() else source.parent
    else:
        parent = Path(args.run_dir).resolve()
    folder = (
        "z_binned_top_bottom_strain_outputs"
        if command == "get_z_binned_top_bottom_strain"
        else "z_binned_deformation_gradient_outputs"
    )
    return parent / folder
