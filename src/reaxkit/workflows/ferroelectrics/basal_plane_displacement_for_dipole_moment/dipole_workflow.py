"""CLI for the Hayden et al. basal-plane displacement dipole method.

Reference: Hayden et al., "Ferroelectricity in boron-substituted aluminum
nitride thin films," Physical Review Materials 5, 044412 (2021),
https://doi.org/10.1103/PhysRevMaterials.5.044412.
"""

from __future__ import annotations

from reaxkit.presentation.workflow_artifacts import write_workflow_tables

import argparse
from pathlib import Path

from reaxkit.analysis.ferroelectrics.basal_plane_displacement_for_dipole_moment.dipole import (
    BasalPlaneDipoleRequest,
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

COMMAND = "get-basal-plane-displacement-dipole"
ALL_COMMANDS = (COMMAND,)
ALL_LEGACY_COMMANDS = ("get_basal_plane_displacement_dipole", "basal-plane-dipole")
TASK_KEY_BY_COMMAND = {COMMAND: COMMAND}


def _canonical_command(command: str) -> str:
    return resolve_command_name(command, ALL_COMMANDS, aliases={COMMAND: ALL_LEGACY_COMMANDS})


def build_parser(parser: argparse.ArgumentParser, *, command: str) -> argparse.ArgumentParser:
    canonical = _canonical_command(command)
    parser.set_defaults(command=canonical, progress=True)
    parser.formatter_class = argparse.RawTextHelpFormatter
    parser.description = """Calculate local dipoles from a three-N basal reference plane.

Three-folded neighbor selection determines the three basal atoms around every
Al/B center. The reference position is their coordinate mean. The N framework
is held fixed, while each center displacement from its own basal plane is
multiplied by its formal or ReaxFF charge and the explicitly negative electron
charge. Every supercell ion is written once; reference-framework ions have zero
displacement instead of being measured from unrelated tetrahedra.

Method reference: Hayden et al., "Ferroelectricity in boron-substituted aluminum
nitride thin films," Physical Review Materials 5, 044412 (2021).

Examples:
  1. Use formal AlN charges with an open z boundary:
     reaxkit get-basal-plane-displacement-dipole --periodic xy --charge-source formal --formal-charge Al=3 N=-3

  2. Use dynamic charges for selected frames:
     reaxkit get-basal-plane-displacement-dipole --charge-source reaxff --fort7 fort.7 --frames 0:101:10
"""
    add_input_arguments(parser)
    add_structure_arguments(parser, include_polarity=True)
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Choose the dipole CSV directory. Example: --output-dir ./basal_dipoles, writes analysis tables there.",
    )
    return parser


def build_request(args: argparse.Namespace) -> BasalPlaneDipoleRequest:
    return BasalPlaneDipoleRequest(
        **structural_request_kwargs(args),
        polarity_tolerance=float(args.polarity_tolerance),
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
    summary = output / "basal_plane_dipole_summary.csv"
    write_workflow_tables({dipoles: result.table, ions: result.ions, summary: result.summary}, args=args, summary=(summary.name,))
    args.suppress_table = True
    present_result(canonical, result, args)
    print(f"Wrote basal-plane displacement dipoles to {dipoles}")
    print(f"Wrote unique-ion contributions to {ions}")
    if getattr(args, "output_profile", "standard") != "minimal":
        print(f"Wrote dipole summary to {summary}")
    return 0


__all__ = [
    "ALL_COMMANDS", "ALL_LEGACY_COMMANDS", "COMMAND", "REQUEST_BUILDERS",
    "TASK_KEY_BY_COMMAND", "build_parser", "build_request", "run_main",
]
