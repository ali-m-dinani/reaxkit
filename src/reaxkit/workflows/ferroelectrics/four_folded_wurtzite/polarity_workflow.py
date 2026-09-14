"""CLI workflow for site-resolved four-fold wurtzite polarity."""

from __future__ import annotations

import argparse
from pathlib import Path

from reaxkit.analysis import ferroelectrics as _ferroelectric_tasks  # noqa: F401
from reaxkit.analysis.ferroelectrics.four_folded_wurtzite.plotting import (
    plot_site_resolved_polarity,
)
from reaxkit.analysis.ferroelectrics.four_folded_wurtzite.polarity import (
    WurtzitePolarityRequest,
)
from reaxkit.core.registry.analysis_task_registry import TASK_REGISTRY
from reaxkit.core.runtime.analysis_executor import AnalysisExecutor
from reaxkit.presentation.dispatcher import present_result
from reaxkit.workflows.ferroelectrics.four_folded_wurtzite.artifacts import (
    write_polarity_tables,
)
from reaxkit.workflows.ferroelectrics.four_folded_wurtzite.common import (
    add_input_arguments,
    add_structure_arguments,
    artifact_directory,
    runtime_arguments,
    structural_request_kwargs,
)

COMMAND = "get-wurtzite-polarity"
ALL_COMMANDS = (COMMAND,)
ALL_LEGACY_COMMANDS = ("get_wurtzite_polarity", "wurtzite-polarity")


def build_parser(parser: argparse.ArgumentParser, *, command: str) -> argparse.ArgumentParser:
    if command not in (*ALL_COMMANDS, *ALL_LEGACY_COMMANDS):
        raise KeyError(f"Unsupported wurtzite-polarity command: {command}")
    parser.set_defaults(command=COMMAND, progress=True)
    parser.formatter_class = argparse.RawTextHelpFormatter
    parser.description = """Calculate site-resolved polar order for four-fold wurtzite environments.

The command assigns four distance-defined neighbors through the neighbor module, then
calculates geometric UP/DOWN polarity, apical and basal bond projections, delta,
charge-weighted local dipoles, eta_c, proton-proximity groups, and frame-zero basal
changes. Separate CSV files preserve center, neighbor, apical, basal, site, and summary
data. Optional plots show the selected property in 3D or as an XY/XZ/YZ projection.

Examples:
  1. Analyze AlN using automatically detected ReaxFF charges:
     reaxkit get-wurtzite-polarity --center Al --neighbor N --charge-source formal --formal-charge Al=3 N=-3 H=1

  2. Analyze AlN using formal charges:
     reaxkit get-wurtzite-polarity --center Al --neighbor N --charge-source formal --formal-charge Al=3 N=-3 H=1

  3. Generate an XZ plot of frame-zero basal changes for a Y slice:
     reaxkit get-wurtzite-polarity --gen-plots --plane xz --slice-range 0 3.2 --plot-value basal-difference

  4. Generate 3D eta_c plots with one color scale per frame:
     reaxkit get-wurtzite-polarity --gen-plots --plot-value eta --color-scale frame
"""
    add_input_arguments(parser)
    add_structure_arguments(parser, include_polarity=True)
    parser.add_argument(
        "--complete-only",
        action="store_true",
        help="Exclude incomplete sites from polarity.csv. Example: --complete-only, writes only centers with four neighbors.",
    )
    parser.add_argument(
        "--gen-plots",
        action="store_true",
        help="Generate one site plot per frame. Example: --gen-plots, writes plots/ alongside the CSV files.",
    )
    parser.add_argument(
        "--plane",
        choices=["xy", "xz", "yz"],
        default=None,
        help="Project plots onto a plane; omission gives 3D. Example: --plane xz, plots x horizontally and z vertically.",
    )
    parser.add_argument(
        "--slice-range",
        nargs=2,
        type=float,
        default=None,
        help="Limit the coordinate omitted by --plane. Example: --plane xz --slice-range 0 3.2, includes only 0 <= y <= 3.2 angstrom.",
    )
    parser.add_argument(
        "--plot-value",
        choices=["polarity", "eta", "delta", "basal", "basal-difference"],
        default="polarity",
        help="Choose the plotted site property. Example: --plot-value eta, colors sites by eta_c in e*angstrom.",
    )
    parser.add_argument(
        "--plot-real-values",
        action="store_true",
        help="Plot continuous geometric delta values. Example: --plot-real-values, is equivalent to --plot-value delta.",
    )
    parser.add_argument(
        "--color-scale",
        choices=["global", "frame"],
        default="global",
        help="Choose continuous color normalization. Example: --color-scale frame, rescales colors independently in every frame.",
    )
    parser.add_argument(
        "--view-elevation",
        type=float,
        default=18.0,
        help="Set the 3D camera elevation in degrees. Example: --view-elevation 25, raises the camera above the xy plane.",
    )
    parser.add_argument(
        "--view-azimuth",
        type=float,
        default=-60.0,
        help="Set the 3D camera azimuth in degrees. Example: --view-azimuth 45, rotates the view around z.",
    )
    parser.add_argument(
        "--marker-size",
        type=float,
        default=12.0,
        help="Set plotted site-marker area. Example: --marker-size 20, draws larger site markers.",
    )
    parser.add_argument(
        "--figure-dpi",
        type=int,
        default=180,
        help="Set output plot resolution. Example: --figure-dpi 300, writes publication-resolution PNG files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Choose the CSV and plot output directory. Example: --output-dir ./polarity, writes all artifacts there.",
    )
    return parser


def build_request(args: argparse.Namespace) -> WurtzitePolarityRequest:
    return WurtzitePolarityRequest(
        **structural_request_kwargs(args),
        polarity_tolerance=float(args.polarity_tolerance),
    )


def run_main(command: str, args: argparse.Namespace) -> int:
    result = AnalysisExecutor().run(
        TASK_REGISTRY[COMMAND](),
        build_request(args),
        runtime_arguments(args),
    )
    output = artifact_directory(args, COMMAND)
    paths = write_polarity_tables(result, output, complete_only=bool(args.complete_only))
    args.suppress_table = True
    present_result(COMMAND, result, args)
    print(f"Wrote site-resolved polarity to {paths['polarity']}")
    print(f"Wrote polarity variable guide to {paths['variables']}")
    print(f"Wrote underlying neighbor-analysis CSVs under {output / 'other_helpful_data'}")
    if args.gen_plots:
        figures = plot_site_resolved_polarity(
            result.table,
            output / "plots",
            plane=args.plane,
            value="delta" if args.plot_real_values else args.plot_value,
            color_scale=args.color_scale,
            slice_range=args.slice_range,
            marker_size=args.marker_size,
            dpi=args.figure_dpi,
            view_elevation=args.view_elevation,
            view_azimuth=args.view_azimuth,
        )
        print(f"Wrote {len(figures):,} 2D/3D frame plots to {output / 'plots'}")
    return 0


__all__ = [
    "ALL_COMMANDS",
    "ALL_LEGACY_COMMANDS",
    "COMMAND",
    "build_parser",
    "build_request",
    "run_main",
]
