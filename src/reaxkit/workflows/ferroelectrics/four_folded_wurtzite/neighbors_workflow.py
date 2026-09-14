"""CLI workflow for center atoms and their four nearest neighbors."""

from __future__ import annotations

import argparse
from pathlib import Path

from reaxkit.analysis import ferroelectrics as _ferroelectric_tasks  # noqa: F401
from reaxkit.analysis.ferroelectrics.four_folded_wurtzite.neighbors import WurtziteNeighborRequest
from reaxkit.core.registry.analysis_task_registry import TASK_REGISTRY
from reaxkit.core.runtime.analysis_executor import AnalysisExecutor
from reaxkit.presentation.dispatcher import present_result
from reaxkit.workflows.ferroelectrics.four_folded_wurtzite.common import (
    add_input_arguments,
    add_structure_arguments,
    artifact_directory,
    runtime_arguments,
    structural_request_kwargs,
)

COMMAND = "get-wurtzite-neighbors"
ALL_COMMANDS = (COMMAND,)
ALL_LEGACY_COMMANDS = ("get_wurtzite_neighbors", "wurtzite-neighbors")


def build_parser(parser: argparse.ArgumentParser, *, command: str) -> argparse.ArgumentParser:
    if command not in (*ALL_COMMANDS, *ALL_LEGACY_COMMANDS):
        raise KeyError(f"Unsupported wurtzite-neighbor command: {command}")
    parser.set_defaults(command=COMMAND, progress=True)
    parser.formatter_class = argparse.RawTextHelpFormatter
    parser.description = """Find the four nearest selected neighbors around every selected center atom.

Use this command to inspect the geometric input to the wurtzite-polarity calculation
without calculating polarity. Neighbor assignment uses distances rather than fort.7
bond connectivity. It writes centers.csv and neighbors.csv with atom IDs, species,
coordinates, minimum-image bond vectors, distances, proton proximity, and dynamic or
formal charges.

Examples:
  1. AlN with automatically detected ReaxFF charges:
     reaxkit get-wurtzite-neighbors --center Al --neighbor N

  2. Require the lightweight ReaxFF fort.7 charge stream:
     reaxkit get-wurtzite-neighbors --engine reaxff --charge-source reaxff

  3. ZnO with formal charges and an explicit output directory:
     reaxkit get-wurtzite-neighbors --center Zn --neighbor O --charge-source formal --formal-charge Zn=2 O=-2
"""
    add_input_arguments(parser)
    add_structure_arguments(parser)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Choose the CSV output directory. Example: --output-dir ./neighbors, writes centers.csv and neighbors.csv there.",
    )
    return parser


def build_request(args: argparse.Namespace) -> WurtziteNeighborRequest:
    return WurtziteNeighborRequest(**structural_request_kwargs(args))


def run_main(command: str, args: argparse.Namespace) -> int:
    result = AnalysisExecutor().run(
        TASK_REGISTRY[COMMAND](),
        build_request(args),
        runtime_arguments(args),
    )
    output = artifact_directory(args, COMMAND)
    output.mkdir(parents=True, exist_ok=True)
    centers_path = output / "centers.csv"
    neighbors_path = output / "neighbors.csv"
    result.csv_tables["centers"].to_csv(centers_path, index=False)
    result.csv_tables["neighbors"].to_csv(neighbors_path, index=False)
    for duplicate in (output / "centers_and_neighbors.csv", output / "table.csv"):
        duplicate.unlink(missing_ok=True)
    args.suppress_table = True
    present_result(COMMAND, result, args)
    print(f"Wrote {len(result.centers):,} center rows to {centers_path}")
    print(f"Wrote {len(result.neighbors):,} neighbor rows to {neighbors_path}")
    return 0


__all__ = [
    "ALL_COMMANDS",
    "ALL_LEGACY_COMMANDS",
    "COMMAND",
    "build_parser",
    "build_request",
    "run_main",
]
