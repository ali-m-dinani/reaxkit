"""Documented CLI workflow for ReaxFF Coulomb potential and local electric fields."""

from __future__ import annotations

import argparse
from pathlib import Path

from reaxkit.analysis import electrostatics as _tasks  # noqa: F401
from reaxkit.analysis.electrostatics.potential_and_electric_field import PotentialElectricFieldRequest
from reaxkit.core.registry.analysis_task_registry import TASK_REGISTRY
from reaxkit.core.runtime.analysis_executor import AnalysisExecutor
from reaxkit.presentation.dispatcher import present_result
from .artifacts import write_binning, write_tables
from .common import add_input_arguments, artifact_directory, attach_energylog_reference, request_kwargs, runtime_arguments

COMMAND = "get-potential-and-electric-field"
ALL_COMMANDS = (COMMAND,)
ALL_LEGACY_COMMANDS = ("get_potential_and_electric_field", "reaxff-local-field")


def build_parser(parser: argparse.ArgumentParser, *, command: str) -> argparse.ArgumentParser:
    """Configure the local electrostatics command-line parser."""
    if command not in (*ALL_COMMANDS, *ALL_LEGACY_COMMANDS): raise KeyError(command)
    parser.set_defaults(command=COMMAND, progress=True)
    parser.formatter_class = argparse.RawTextHelpFormatter
    parser.description = """Calculate ReaxFF Coulomb energies, local potentials, and electric fields at atom coordinates.

Use this command to reconstruct the shielded Coulomb contribution from xmolout,
fort.7, and ffield. It writes per-atom and total energies, one potential/field CSV
per +1e probe species, and an equal-probe average. Optional spatial binning produces
1D profiles, 2D heatmaps, or 3D occupied-bin plots with a shared color scale.

Examples:
  1. Calculate every twentieth saved frame with automatic probe species:
     reaxkit get-potential-and-electric-field --run-dir ./run --frames ::20 --periodic xyz

  2. Bin Ez along z and write one globally scaled plot per frame and probe:
     reaxkit get-potential-and-electric-field --run-dir ./run --frames 0:101:20 --bin-axes z --bins 50 --plot-bins --plot-field-component z

  3. Validate the analytic field with a numerical derivative and no taper:
     reaxkit get-potential-and-electric-field --run-dir ./run --field-method numerical --field-step 0.0005 --disable-taper
"""
    add_input_arguments(parser)
    parser.add_argument(
        "--bin-axes", choices=["x", "y", "z", "xy", "xz", "yz", "xyz"], default=None,
        help="Choose coordinates used for spatial binning. Example: --bin-axes xz, produces x-z heatmap data by averaging atoms within each x-z bin.",
    )
    parser.add_argument(
        "--bins", nargs="+", type=int, default=[50],
        help="Set one bin count for all axes or one count per axis. Example: --bin-axes xz --bins 40 80, creates 40 x bins and 80 z bins.",
    )
    parser.add_argument(
        "--plot-bins", action="store_true",
        help="Generate binned field plots in addition to CSV files. Example: --bin-axes z --plot-bins, writes one z profile per selected frame and probe.",
    )
    parser.add_argument(
        "--plot-field-component", choices=["x", "y", "z", "magnitude", "mean-magnitude"], default="magnitude",
        help="Choose the scalar shown in binned plots. Example: --plot-field-component z, plots the signed mean Ez component in each bin.",
    )
    parser.add_argument(
        "--plot-field-units", choices=["v/angstrom", "mv/cm"], default="mv/cm",
        help="Choose electric-field units for plot values and color scales. Example: --plot-field-units v/angstrom, labels plots in volts per angstrom.",
    )
    parser.add_argument(
        "--figure-dpi", type=int, default=180,
        help="Set PNG plot resolution in dots per inch. Example: --figure-dpi 300, writes higher-resolution figures.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Choose the directory for CSV and plot artifacts. Example: --output-dir ./results/local_field, writes all analysis artifacts there.",
    )
    return parser


def build_request(args) -> PotentialElectricFieldRequest:
    return PotentialElectricFieldRequest(**request_kwargs(args))


def run_main(command: str, args: argparse.Namespace) -> int:
    if args.plot_bins and not args.bin_axes: raise ValueError("--plot-bins requires --bin-axes.")
    result = AnalysisExecutor().run(TASK_REGISTRY[COMMAND](), build_request(args), runtime_arguments(args))
    attach_energylog_reference(result, args)
    output = artifact_directory(args, COMMAND); paths = list(write_tables(result, output).values())
    if args.bin_axes:
        paths.extend(write_binning(result, output, axes=args.bin_axes, bins=args.bins, plot=args.plot_bins,
                                   component=args.plot_field_component, units=args.plot_field_units, dpi=args.figure_dpi))
    args.suppress_table = True; present_result(COMMAND, result, args)
    print(f"Processed {len(result.frame_indices):,} frame(s); wrote {len(paths):,} artifacts under {output}.")
    return 0


__all__ = ["ALL_COMMANDS", "ALL_LEGACY_COMMANDS", "COMMAND", "build_parser", "build_request", "run_main"]
