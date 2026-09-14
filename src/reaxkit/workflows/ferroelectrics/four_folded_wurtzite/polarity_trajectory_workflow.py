"""CLI workflow for species-preserving polarity Extended XYZ trajectories."""

from __future__ import annotations

import argparse
import shutil
from dataclasses import replace
from pathlib import Path

from reaxkit.analysis import ferroelectrics as _ferroelectric_tasks  # noqa: F401
from reaxkit.analysis.ferroelectrics.four_folded_wurtzite.trajectory import (
    PolarityExtendedXYZRequest,
)
from reaxkit.core.registry.analysis_task_registry import TASK_REGISTRY
from reaxkit.core.runtime.analysis_executor import AnalysisExecutor
from reaxkit.engine.reaxff.adapter_parts.io_paths import _quick_n_frames_from_control
from reaxkit.presentation.dispatcher import present_result
from reaxkit.workflows.ferroelectrics.four_folded_wurtzite.artifacts import (
    write_polarity_tables,
)
from reaxkit.workflows.ferroelectrics.four_folded_wurtzite.common import (
    add_input_arguments,
    add_structure_arguments,
    runtime_arguments,
    structural_request_kwargs,
    workspace_artifact_directory,
)

COMMAND = "write-trajectory-with-polarity"
ALL_COMMANDS = (COMMAND,)
ALL_LEGACY_COMMANDS = ("write_trajectory_with_polarity", "polarity-extxyz")


def build_parser(parser: argparse.ArgumentParser, *, command: str) -> argparse.ArgumentParser:
    if command not in (*ALL_COMMANDS, *ALL_LEGACY_COMMANDS):
        raise KeyError(f"Unsupported polarity-trajectory command: {command}")
    parser.set_defaults(command=COMMAND, progress=True)
    parser.formatter_class = argparse.RawTextHelpFormatter
    parser.description = """Write a species-preserving Extended XYZ trajectory with local polarity properties.

Unlike the original relabeling script, this command keeps every atom name unchanged.
It adds atom_number, charge, polarity, eta_c, delta_eff, is_polarity_site,
has_four_neighbors, and has_proton_within_cutoff columns. Each frame header includes
frame and iter, plus lattice/PBC metadata and an iteration-matched electric field when
requested. ReaxFF charges are streamed through the lightweight charge-only reader.
The canonical trajectory and the neighbor/polarity CSV tables are always written under
reaxkit_workspace. --output optionally creates a second copy at another destination.
  
Examples:
  1. Write an AlN polarity trajectory with automatically assigned charges and electric field profile:
     reaxkit write-trajectory-with-polarity --include-electric-field --field-direction z --charge-source formal --formal-charge Al=3 N=-3 H=1
"""
    add_input_arguments(parser)
    add_structure_arguments(parser, include_polarity=True)
    parser.add_argument(
        "--include-electric-field",
        action="store_true",
        help="Add an iteration-matched field to frame headers. Example: --include-electric-field, reads fort.78 and writes field metadata.",
    )
    parser.add_argument(
        "--field-direction",
        choices=["x", "y", "z"],
        default="z",
        help="Choose the field component written to headers. Example: --field-direction z, writes the z component in MV/cm.",
    )
    parser.add_argument(
        "--precision",
        type=int,
        default=8,
        help="Set significant digits for real values. Example: --precision 12, writes coordinates and properties with 12 digits.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Create an additional Extended XYZ copy outside the workspace. Example: "
            "--output ./polarity.extxyz, writes both the workspace artifact and that file."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Create an additional trajectory copy in a directory. Example: --output-dir ./results, "
            "also writes ./results/trajectory_with_polarity.extxyz. Prefer --output for a custom name."
        ),
    )
    parser.add_argument(
        "--control",
        default="control",
        help="Select the control file used for frame-count progress. Example: --control ./run/control, reads progress metadata there.",
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


def _requested_output_path(args: argparse.Namespace) -> Path | None:
    if args.output is not None:
        output = Path(args.output)
        if output.suffix.lower() not in {".xyz", ".extxyz"}:
            output = output / "trajectory_with_polarity.extxyz"
        return output.resolve()
    if args.output_dir is not None:
        return (Path(args.output_dir) / "trajectory_with_polarity.extxyz").resolve()
    return None


def _output_path(args: argparse.Namespace) -> Path:
    """Return the canonical trajectory path inside the ReaxKit workspace."""

    requested = _requested_output_path(args)
    name = requested.name if requested is not None else "trajectory_with_polarity.extxyz"
    return (workspace_artifact_directory(args, COMMAND) / name).resolve()


def _control_file(args: argparse.Namespace) -> Path:
    configured = Path(str(args.control))
    if configured != Path("control"):
        return configured
    for name in ("fort7", "xmolout", "input", "run_dir"):
        source = Path(str(getattr(args, name, "") or ""))
        directory = source if source.is_dir() else source.parent
        candidate = directory / "control"
        if candidate.is_file():
            return candidate
    return configured


def run_main(command: str, args: argparse.Namespace) -> int:
    base_request = build_request(args)
    expected = (
        len(set([*(base_request.frames or ()), base_request.reference_frame]))
        if base_request.frames is not None
        else _quick_n_frames_from_control(_control_file(args))
    )
    request = replace(
        base_request,
        _output_path=str(_output_path(args)),
        _expected_frames=expected,
    )
    result = AnalysisExecutor().run(
        TASK_REGISTRY[COMMAND](),
        request,
        runtime_arguments(args),
    )
    workspace_output = Path(result.output_path)
    write_polarity_tables(
        result.polarity_result,
        workspace_output.parent,
        all_csvs_in_helpful_data=True,
    )
    requested_output = _requested_output_path(args)
    if requested_output is not None and requested_output != workspace_output:
        requested_output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(workspace_output, requested_output)
    args.suppress_table = True
    present_result(COMMAND, result, args)
    print(f"Wrote species-preserving polarity trajectory: {workspace_output}")
    print(
        "Wrote reusable neighbor and polarity CSVs under "
        f"{workspace_output.parent / 'other_helpful_data'}"
    )
    if requested_output is not None and requested_output != workspace_output:
        print(f"Wrote requested trajectory copy: {requested_output}")
    return 0


__all__ = [
    "ALL_COMMANDS",
    "ALL_LEGACY_COMMANDS",
    "COMMAND",
    "build_parser",
    "build_request",
    "run_main",
]
