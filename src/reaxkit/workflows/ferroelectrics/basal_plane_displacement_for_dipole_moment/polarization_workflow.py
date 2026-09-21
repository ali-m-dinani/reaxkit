"""CLI for polarization from Hayden et al. basal-plane displacement dipoles.

Reference: Hayden et al., "Ferroelectricity in boron-substituted aluminum
nitride thin films," Physical Review Materials 5, 044412 (2021),
https://doi.org/10.1103/PhysRevMaterials.5.044412.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from reaxkit.analysis.ferroelectrics.basal_plane_displacement_for_dipole_moment.polarization import (
    BasalPlanePolarizationRequest,
)
from reaxkit.core.registry.analysis_task_registry import TASK_REGISTRY
from reaxkit.core.resolve.command_alias_resolver import resolve_command_name
from reaxkit.core.runtime.analysis_executor import AnalysisExecutor
from reaxkit.presentation.dispatcher import present_result
from reaxkit.workflows.ferroelectrics.three_folded_wurtzite.common import (
    add_input_arguments,
    add_structure_arguments,
    artifact_directory,
    runtime_arguments,
    structural_request_kwargs,
)
from reaxkit.workflows.ferroelectrics.three_folded_wurtzite.polarization_workflow import (
    generate_polarization_heatmaps,
)

COMMAND = "get-basal-plane-displacement-polarization"
ALL_COMMANDS = (COMMAND,)
ALL_LEGACY_COMMANDS = (
    "get_basal_plane_displacement_polarization",
    "basal-plane-polarization",
)
TASK_KEY_BY_COMMAND = {COMMAND: COMMAND}


def _canonical_command(command: str) -> str:
    return resolve_command_name(command, ALL_COMMANDS, aliases={COMMAND: ALL_LEGACY_COMMANDS})


def build_parser(parser: argparse.ArgumentParser, *, command: str) -> argparse.ArgumentParser:
    canonical = _canonical_command(command)
    parser.set_defaults(command=canonical, progress=True)
    parser.formatter_class = argparse.RawTextHelpFormatter
    parser.description = """Bin basal-plane displacement dipoles and calculate polarization.

The dipole reference for each Al/B center is the mean position of its three
basal N atoms. The N framework is fixed, and each center contributes once with
the explicitly negative electron charge. Contributions are summed in each
spatial bin and divided by a hull, bounding-box, or cell volume. For a slab,
cell volume includes vacuum; hull is the default material-volume estimate.

Method reference: Hayden et al., "Ferroelectricity in boron-substituted aluminum
nitride thin films," Physical Review Materials 5, 044412 (2021).

Examples:
  1. Calculate material-volume polarization in an x-y grid:
     reaxkit get-basal-plane-displacement-polarization --bins-x 20 --bins-y 20 --volume-method hull --periodic xy --charge-source formal --formal-charge Al=3 N=-3

  2. Generate globally scaled Pz maps for selected frames:
     reaxkit get-basal-plane-displacement-polarization --frames 0:101:10 --bins-x 30 --bins-y 30 --heatmaps --global-scaling
"""
    add_input_arguments(parser)
    add_structure_arguments(parser, include_polarity=True)
    parser.add_argument(
        "--bins-x", type=int, default=1,
        help="Set x-direction bins. Example: --bins-x 20, divides the reference x extent into 20 intervals.",
    )
    parser.add_argument(
        "--bins-y", type=int, default=1,
        help="Set y-direction bins. Example: --bins-y 20, divides the reference y extent into 20 intervals.",
    )
    parser.add_argument(
        "--bins-z", type=int, default=1,
        help="Set z-direction bins. Example: --bins-z 10, divides the reference z extent into 10 intervals.",
    )
    parser.add_argument(
        "--volume-method", choices=["hull", "bbox", "cell"], default="hull",
        help="Choose the bin-volume estimator. Example: --volume-method hull, uses the occupied atomic convex hull in each bin; cell includes slab vacuum.",
    )
    parser.add_argument(
        "--heatmaps", action="store_true",
        help="Write a 2D map for every selected frame. Example: --heatmaps, creates PNG files under heatmaps.",
    )
    parser.add_argument(
        "--heatmap-plane", choices=["xy", "xz", "yz"], default="xy",
        help="Choose the projected plane. Example: --heatmap-plane xz, sums dipoles and volumes along y.",
    )
    parser.add_argument(
        "--heatmap-component", choices=["x", "y", "z"], default="z",
        help="Choose the polarization component. Example: --heatmap-component z, plots Pz.",
    )
    parser.add_argument(
        "--global-scaling", action=argparse.BooleanOptionalAction, default=False,
        help="Share color limits across frames. Example: --global-scaling, makes frame colors comparable; --no-global-scaling rescales each frame.",
    )
    parser.add_argument(
        "--figure-dpi", type=int, default=180,
        help="Set heatmap resolution. Example: --figure-dpi 300, writes 300-DPI PNG files.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Choose the output directory. Example: --output-dir ./basal_polarization, writes CSV and heatmap artifacts there.",
    )
    return parser


def build_request(args: argparse.Namespace) -> BasalPlanePolarizationRequest:
    return BasalPlanePolarizationRequest(
        **structural_request_kwargs(args),
        polarity_tolerance=float(args.polarity_tolerance),
        bins_x=int(args.bins_x), bins_y=int(args.bins_y), bins_z=int(args.bins_z),
        volume_method=str(args.volume_method),
    )


REQUEST_BUILDERS = {COMMAND: build_request}


def run_main(command: str, args: argparse.Namespace) -> int:
    canonical = _canonical_command(command)
    result = AnalysisExecutor().run(
        TASK_REGISTRY[TASK_KEY_BY_COMMAND[canonical]](),
        REQUEST_BUILDERS[canonical](args),
        runtime_arguments(args),
    )
    output = artifact_directory(args, canonical)
    output.mkdir(parents=True, exist_ok=True)
    dipoles = output / "basal_plane_dipoles.csv"
    ions = output / "basal_plane_ions.csv"
    bins = output / "basal_plane_polarization.csv"
    summary = output / "basal_plane_polarization_summary.csv"
    result.dipole_result.table.to_csv(dipoles, index=False)
    result.dipole_result.ions.to_csv(ions, index=False)
    result.table.to_csv(bins, index=False)
    result.summary.to_csv(summary, index=False)
    figures = []
    if args.heatmaps:
        figures = generate_polarization_heatmaps(
            result, output, plane=args.heatmap_plane, component=args.heatmap_component,
            global_scaling=bool(args.global_scaling), dpi=int(args.figure_dpi),
            title_prefix="Basal-plane displacement",
        )
    args.suppress_table = True
    present_result(canonical, result, args)
    print(f"Wrote basal-plane polarization to {bins}")
    print(f"Wrote source dipoles to {dipoles}")
    print(f"Wrote unique-ion contributions to {ions}")
    print(f"Wrote polarization summary to {summary}")
    if figures:
        scale = "global" if args.global_scaling else "per-frame"
        print(f"Wrote {len(figures):,} {scale}-scaled heatmap(s) under {output / 'heatmaps'}")
    return 0


__all__ = [
    "ALL_COMMANDS", "ALL_LEGACY_COMMANDS", "COMMAND", "REQUEST_BUILDERS",
    "TASK_KEY_BY_COMMAND", "build_parser", "build_request", "run_main",
]
