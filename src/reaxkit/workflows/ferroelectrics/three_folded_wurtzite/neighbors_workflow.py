"""CLI workflow for three-folded wurtzite neighbor geometry."""

from __future__ import annotations

from reaxkit.presentation.workflow_artifacts import write_workflow_tables

import argparse
from pathlib import Path

from reaxkit.analysis.ferroelectrics.three_folded_wurtzite.neighbors import (
    WurtziteNeighborRequest,
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

COMMAND = "get-three-folded-wurtzite-neighbors"
ALL_COMMANDS = (COMMAND,)
ALL_LEGACY_COMMANDS = (
    "get_three_folded_wurtzite_neighbors",
    "three-folded-wurtzite-neighbors",
)
TASK_KEY_BY_COMMAND = {COMMAND: "get-three-folded-wurtzite-neighbors"}


def _canonical_command(command: str) -> str:
    return resolve_command_name(
        command, ALL_COMMANDS, aliases={COMMAND: ALL_LEGACY_COMMANDS}
    )


def build_parser(parser: argparse.ArgumentParser, *, command: str) -> argparse.ArgumentParser:
    canonical = _canonical_command(command)
    parser.set_defaults(command=canonical, progress=True)
    parser.formatter_class = argparse.RawTextHelpFormatter
    parser.description = """Find basal neighbors and apical candidates around three-folded wurtzite sites.

Use this command to inspect the geometry consumed by the basal-only polarity
calculation. Three candidates with the smallest absolute c-axis projections are
marked basal; every other candidate inside the cutoff is retained for later
polarity-aware apical selection. This command writes geometry tables only.

Examples:
  1. Inspect an AlN surface with an open z boundary:
     reaxkit get-three-folded-wurtzite-neighbors --center Al --neighbor N --periodic xy

  2. Use explicit formal charges for ZnO:
     reaxkit get-three-folded-wurtzite-neighbors --center Zn --neighbor O --charge-source formal --formal-charge Zn=2 O=-2
"""
    add_input_arguments(parser)
    add_structure_arguments(parser)
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Choose the CSV output directory. Example: --output-dir ./three_folded_neighbors, writes centers.csv and neighbors.csv there.",
    )
    return parser


def build_request(args: argparse.Namespace) -> WurtziteNeighborRequest:
    return WurtziteNeighborRequest(**structural_request_kwargs(args))


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
    centers_path = output / "centers.csv"
    neighbors_path = output / "neighbors.csv"
    write_workflow_tables({centers_path: result.csv_tables["centers"], neighbors_path: result.csv_tables["neighbors"]}, args=args)
    args.suppress_table = True
    present_result(canonical, result, args)
    print(f"Wrote {len(result.centers):,} center rows to {centers_path}")
    print(f"Wrote {len(result.neighbors):,} candidate-neighbor rows to {neighbors_path}")
    return 0


__all__ = [
    "ALL_COMMANDS", "ALL_LEGACY_COMMANDS", "COMMAND", "REQUEST_BUILDERS",
    "TASK_KEY_BY_COMMAND", "build_parser", "build_request", "run_main",
]
