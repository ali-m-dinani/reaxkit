"""CLI workflow for basal-only three-folded wurtzite polarity."""

from __future__ import annotations

import argparse
from pathlib import Path

from reaxkit.analysis.ferroelectrics.four_folded_wurtzite.plotting import (
    plot_site_resolved_polarity,
)
from reaxkit.analysis.ferroelectrics.three_folded_wurtzite.polarity import (
    WurtzitePolarityRequest,
)
from reaxkit.core.registry.analysis_task_registry import TASK_REGISTRY
from reaxkit.core.resolve.command_alias_resolver import resolve_command_name
from reaxkit.core.runtime.analysis_executor import AnalysisExecutor
from reaxkit.presentation.dispatcher import present_result
from reaxkit.workflows.ferroelectrics.three_folded_wurtzite.artifacts import (
    write_polarity_tables,
)
from reaxkit.workflows.ferroelectrics.three_folded_wurtzite.common import (
    add_input_arguments,
    add_structure_arguments,
    artifact_directory,
    runtime_arguments,
    structural_request_kwargs,
)

COMMAND = "get-three-folded-wurtzite-polarity"
ALL_COMMANDS = (COMMAND,)
ALL_LEGACY_COMMANDS = (
    "get_three_folded_wurtzite_polarity",
    "three-folded-wurtzite-polarity",
)
TASK_KEY_BY_COMMAND = {COMMAND: "get-three-folded-wurtzite-polarity"}


def _canonical_command(command: str) -> str:
    return resolve_command_name(
        command, ALL_COMMANDS, aliases={COMMAND: ALL_LEGACY_COMMANDS}
    )


def build_parser(parser: argparse.ArgumentParser, *, command: str) -> argparse.ArgumentParser:
    canonical = _canonical_command(command)
    parser.set_defaults(command=canonical, progress=True)
    parser.formatter_class = argparse.RawTextHelpFormatter
    parser.description = """Calculate site polarity from three basal bonds before assigning an apical neighbor.

The three smallest absolute c-axis projections define the basal set and the negative
of their signed mean defines delta. Three basal neighbors are sufficient for a complete polarity
site. An apical candidate is selected afterward only when it lies on the side
required by UP or DOWN polarity. CSV outputs preserve selected and ignored candidates.

Examples:
  1. Analyze an AlN slab using formal charges and open z boundaries:
     reaxkit get-three-folded-wurtzite-polarity --center Al --neighbor N --periodic xy --charge-source formal --formal-charge Al=3 N=-3

  2. Plot basal-only delta on an XZ slice:
     reaxkit get-three-folded-wurtzite-polarity --gen-plots --plane xz --slice-range 0 3.2 --plot-value delta
"""
    add_input_arguments(parser)
    add_structure_arguments(parser, include_polarity=True)
    parser.add_argument(
        "--complete-only", action="store_true",
        help="Exclude sites with fewer than three basal neighbors from polarity.csv. Example: --complete-only, retains all valid three-coordinate surface sites.",
    )
    parser.add_argument(
        "--gen-plots", action="store_true",
        help="Generate one site plot per frame. Example: --gen-plots, writes PNG files under the plots directory.",
    )
    parser.add_argument(
        "--plane", choices=["xy", "xz", "yz"], default=None,
        help="Project plots onto a plane; omission gives 3D. Example: --plane xz, plots x horizontally and z vertically.",
    )
    parser.add_argument(
        "--slice-range", nargs=2, type=float, default=None,
        help="Limit the coordinate omitted by --plane. Example: --plane xz --slice-range 0 3.2, includes sites with y from 0 to 3.2 angstrom.",
    )
    parser.add_argument(
        "--plot-value", choices=["polarity", "eta", "delta", "basal", "basal-difference"],
        default="polarity",
        help="Choose the plotted site property. Example: --plot-value delta, colors sites by the signed basal-only displacement.",
    )
    parser.add_argument(
        "--plot-real-values", action="store_true",
        help="Plot continuous delta values. Example: --plot-real-values, selects the same values as --plot-value delta.",
    )
    parser.add_argument(
        "--color-scale", choices=["global", "frame"], default="global",
        help="Choose continuous color normalization. Example: --color-scale frame, rescales colors independently for each frame.",
    )
    parser.add_argument(
        "--view-elevation", type=float, default=18.0,
        help="Set 3D camera elevation in degrees. Example: --view-elevation 25, raises the camera above the xy plane.",
    )
    parser.add_argument(
        "--view-azimuth", type=float, default=-60.0,
        help="Set 3D camera azimuth in degrees. Example: --view-azimuth 45, rotates the view around z.",
    )
    parser.add_argument(
        "--marker-size", type=float, default=12.0,
        help="Set site-marker area. Example: --marker-size 20, draws larger site markers.",
    )
    parser.add_argument(
        "--figure-dpi", type=int, default=180,
        help="Set plot resolution. Example: --figure-dpi 300, writes 300-DPI PNG files.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Choose the CSV and plot directory. Example: --output-dir ./three_folded_polarity, writes all analysis artifacts there.",
    )
    return parser


def build_request(args: argparse.Namespace) -> WurtzitePolarityRequest:
    return WurtzitePolarityRequest(
        **structural_request_kwargs(args),
        polarity_tolerance=float(args.polarity_tolerance),
    )


REQUEST_BUILDERS = {COMMAND: build_request}


def _plot_payload(
    command: str,
    result,
    args: argparse.Namespace,
) -> dict[str, object] | None:
    """Return validated plotting options, or None when no plot can be produced."""

    _canonical_command(command)
    if not args.gen_plots or result.table.empty:
        return None
    value = "delta" if args.plot_real_values else str(args.plot_value)
    value_columns = {
        "polarity": "polarity",
        "eta": "eta_c (e*angstrom)",
        "delta": "delta_eff (angstrom)",
        "basal": "mean_basal_bond_c (angstrom)",
        "basal-difference": "mean_basal_bond_c_change_from_frame_0 (angstrom)",
    }
    required = {
        "frame_index", "site_x (angstrom)", "site_y (angstrom)",
        "site_z (angstrom)", value_columns[value],
    }
    if not required.issubset(result.table.columns):
        return None
    return {
        "plane": args.plane,
        "value": value,
        "color_scale": args.color_scale,
        "slice_range": args.slice_range,
        "marker_size": args.marker_size,
        "dpi": args.figure_dpi,
        "view_elevation": args.view_elevation,
        "view_azimuth": args.view_azimuth,
    }


def run_main(command: str, args: argparse.Namespace) -> int:
    canonical = _canonical_command(command)
    result = AnalysisExecutor().run(
        TASK_REGISTRY[TASK_KEY_BY_COMMAND[canonical]](),
        REQUEST_BUILDERS[canonical](args),
        runtime_arguments(args),
    )
    output = artifact_directory(args, canonical)
    paths = write_polarity_tables(result, output, complete_only=bool(args.complete_only), args=args)
    args.suppress_table = True
    present_result(canonical, result, args)
    print(f"Wrote site-resolved polarity to {paths['polarity']}")
    print(f"Wrote polarity variable guide to {paths['variables']}")
    print(f"Wrote candidate-neighbor CSVs under {output / 'other_helpful_data'}")
    plot_payload = _plot_payload(canonical, result, args)
    if plot_payload is not None:
        figures = plot_site_resolved_polarity(
            result.table,
            output / "plots",
            **plot_payload,
        )
        print(f"Wrote {len(figures):,} frame plots to {output / 'plots'}")
    return 0


__all__ = [
    "ALL_COMMANDS", "ALL_LEGACY_COMMANDS", "COMMAND", "REQUEST_BUILDERS",
    "TASK_KEY_BY_COMMAND", "_plot_payload", "build_parser", "build_request", "run_main",
]
