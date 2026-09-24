"""CLI workflow for three-folded polarity Extended XYZ trajectories."""

from __future__ import annotations

import argparse
import shutil
from dataclasses import replace
from pathlib import Path

from reaxkit.analysis.ferroelectrics.three_folded_wurtzite.trajectory import (
    PolarityExtendedXYZRequest,
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
    runtime_arguments,
    structural_request_kwargs,
    workspace_artifact_directory,
)

COMMAND = "write-three-folded-trajectory-with-polarity"
ALL_COMMANDS = (COMMAND,)
ALL_LEGACY_COMMANDS = (
    "write_three_folded_trajectory_with_polarity",
    "three-folded-polarity-extxyz",
)
TASK_KEY_BY_COMMAND = {COMMAND: "write-three-folded-trajectory-with-polarity"}


def _canonical_command(command: str) -> str:
    return resolve_command_name(
        command, ALL_COMMANDS, aliases={COMMAND: ALL_LEGACY_COMMANDS}
    )


def build_parser(parser: argparse.ArgumentParser, *, command: str) -> argparse.ArgumentParser:
    canonical = _canonical_command(command)
    parser.set_defaults(command=canonical, progress=True)
    parser.formatter_class = argparse.RawTextHelpFormatter
    parser.description = """Write an Extended XYZ trajectory with three-folded wurtzite polarity properties.

Atom names remain unchanged. Each site receives basal-only polarity, delta_eff,
eta_c, three-basal completeness, and polarity-consistent apical-presence fields.
Frame headers retain iteration, lattice, and periodic metadata and can include an
iteration-matched electric field. The workspace also receives reusable CSV tables.

Examples:
  1. Export an AlN slab with formal charges:
     reaxkit write-three-folded-trajectory-with-polarity --periodic xy --charge-source formal --formal-charge Al=3 N=-3 H=1

  2. Add the applied z field and create a named copy:
     reaxkit write-three-folded-trajectory-with-polarity --include-electric-field --field-direction z --output ./aln_surface.extxyz
"""
    add_input_arguments(parser)
    add_structure_arguments(parser, include_polarity=True)
    parser.add_argument(
        "--include-electric-field", action="store_true",
        help="Add an iteration-matched field to frame headers. Example: --include-electric-field, reads fort.78 and writes electric-field metadata.",
    )
    parser.add_argument(
        "--field-direction", choices=["x", "y", "z"], default="z",
        help="Choose the field component written to headers. Example: --field-direction z, writes the z component in MV/cm.",
    )
    parser.add_argument(
        "--precision", type=int, default=8,
        help="Set significant digits for real values. Example: --precision 12, writes coordinates and properties with 12 digits.",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Create an additional trajectory copy. Example: --output ./surface.extxyz, copies the workspace artifact to that file.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Create an additional copy in a directory. Example: --output-dir ./results, writes three_folded_trajectory_with_polarity.extxyz there.",
    )
    return parser


def build_request(args: argparse.Namespace) -> PolarityExtendedXYZRequest:
    return PolarityExtendedXYZRequest(
        **structural_request_kwargs(args),
        polarity_tolerance=float(args.polarity_tolerance),
        precision=int(args.precision),
        include_electric_field=bool(args.include_electric_field),
        field_direction=str(args.field_direction),
    )


REQUEST_BUILDERS = {COMMAND: build_request}


def _requested_output_path(args: argparse.Namespace) -> Path | None:
    default_name = "three_folded_trajectory_with_polarity.extxyz"
    if args.output is not None:
        output = Path(args.output)
        if output.suffix.lower() not in {".xyz", ".extxyz"}:
            output = output / default_name
        return output.resolve()
    if args.output_dir is not None:
        return (Path(args.output_dir) / default_name).resolve()
    return None


def _output_path(args: argparse.Namespace, command: str = COMMAND) -> Path:
    requested = _requested_output_path(args)
    name = requested.name if requested is not None else "three_folded_trajectory_with_polarity.extxyz"
    return (workspace_artifact_directory(args, command) / name).resolve()


def run_main(command: str, args: argparse.Namespace) -> int:
    canonical = _canonical_command(command)
    request = replace(
        REQUEST_BUILDERS[canonical](args),
        _output_path=str(_output_path(args, canonical)),
    )
    result = AnalysisExecutor().run(
        TASK_REGISTRY[TASK_KEY_BY_COMMAND[canonical]](),
        request,
        runtime_arguments(args),
    )
    workspace_output = Path(result.output_path)
    write_polarity_tables(
        result.polarity_result,
        workspace_output.parent,
        all_csvs_in_helpful_data=True,
        args=args,
    )
    requested_output = _requested_output_path(args)
    if requested_output is not None and requested_output != workspace_output:
        requested_output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(workspace_output, requested_output)
    args.suppress_table = True
    present_result(canonical, result, args)
    print(f"Wrote species-preserving three-folded polarity trajectory: {workspace_output}")
    print(f"Wrote reusable CSVs under {workspace_output.parent / 'other_helpful_data'}")
    if requested_output is not None and requested_output != workspace_output:
        print(f"Wrote requested trajectory copy: {requested_output}")
    return 0


__all__ = [
    "ALL_COMMANDS", "ALL_LEGACY_COMMANDS", "COMMAND", "REQUEST_BUILDERS",
    "TASK_KEY_BY_COMMAND", "build_parser", "build_request", "run_main",
]
