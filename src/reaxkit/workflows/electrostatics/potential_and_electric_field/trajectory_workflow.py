"""Documented CLI workflow for local ReaxFF electrostatics Extended XYZ export."""

from __future__ import annotations

import argparse
import shutil
from dataclasses import replace
from pathlib import Path

from reaxkit.analysis import electrostatics as _tasks  # noqa: F401
from reaxkit.analysis.electrostatics.potential_and_electric_field import PotentialElectricFieldTrajectoryRequest
from reaxkit.core.registry.analysis_task_registry import TASK_REGISTRY
from reaxkit.core.runtime.analysis_executor import AnalysisExecutor
from reaxkit.presentation.dispatcher import present_result
from .artifacts import write_tables
from .common import add_input_arguments, artifact_directory, attach_energylog_reference, request_kwargs, runtime_arguments

COMMAND = "write-trajectory-with-potential-and-electric-field"
ALL_COMMANDS = (COMMAND,)
ALL_LEGACY_COMMANDS = ("write_trajectory_with_potential_and_electric_field", "local-field-extxyz")


def build_parser(parser: argparse.ArgumentParser, *, command: str) -> argparse.ArgumentParser:
    """Configure the local-electrostatics trajectory command-line parser."""
    if command not in (*ALL_COMMANDS, *ALL_LEGACY_COMMANDS): raise KeyError(command)
    parser.set_defaults(command=COMMAND, progress=True)
    parser.formatter_class = argparse.RawTextHelpFormatter
    parser.description = """Write an Extended XYZ trajectory with internal, external, and total local electrostatic properties.

Use this command when local electrostatic values must remain aligned with atom species,
coordinates, charges, frame indices, and iterations for OVITO or another trajectory tool.
Each frame includes internal, external, and total local values, plus separate internal
and total values for every selected +1e probe species. Applied fields come from fort.78.
The command calculates and exports data; it does not run ReaxFF.

Examples:
  1. Write every twentieth frame with automatically detected probe species:
     reaxkit write-trajectory-with-potential-and-electric-field --run-dir ./run --frames ::20 --periodic xyz

  2. Write Al- and N-probe properties to a named trajectory copy:
     reaxkit write-trajectory-with-potential-and-electric-field --run-dir ./run --probe-elements Al N --output ./results/local_electrostatics.extxyz
"""
    add_input_arguments(parser)
    parser.add_argument(
        "--precision", type=int, default=8,
        help="Set significant digits for coordinates and real-valued properties. Example: --precision 12, writes twelve significant digits to the Extended XYZ file.",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Choose an additional Extended XYZ destination. Example: --output ./results/local_field.extxyz, copies the workspace trajectory to that file.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Choose the artifact directory and default trajectory-copy location. Example: --output-dir ./results, writes trajectory_with_local_electrostatics.extxyz and CSV files there.",
    )
    return parser


def build_request(args) -> PotentialElectricFieldTrajectoryRequest:
    return PotentialElectricFieldTrajectoryRequest(**request_kwargs(args), precision=int(args.precision))


def _requested(args) -> Path | None:
    if args.output is not None:
        path = Path(args.output); return (path if path.suffix.lower() in {".xyz", ".extxyz"} else path / "trajectory_with_local_electrostatics.extxyz").resolve()
    if args.output_dir is not None: return (Path(args.output_dir) / "trajectory_with_local_electrostatics.extxyz").resolve()
    return None


def run_main(command: str, args: argparse.Namespace) -> int:
    output_dir = artifact_directory(args, COMMAND)
    canonical = (output_dir / "trajectory_with_local_electrostatics.extxyz").resolve()
    request = replace(build_request(args), _output_path=str(canonical))
    result = AnalysisExecutor().run(TASK_REGISTRY[COMMAND](), request, runtime_arguments(args))
    attach_energylog_reference(result.electrostatics_result, args)
    write_tables(result.electrostatics_result, output_dir)
    requested = _requested(args)
    if requested is not None and requested != canonical:
        requested.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(canonical, requested)
    args.suppress_table = True; present_result(COMMAND, result, args)
    print(f"Wrote local electrostatics trajectory: {canonical}")
    if requested is not None and requested != canonical: print(f"Wrote requested trajectory copy: {requested}")
    return 0


__all__ = ["ALL_COMMANDS", "ALL_LEGACY_COMMANDS", "COMMAND", "build_parser", "build_request", "run_main"]
