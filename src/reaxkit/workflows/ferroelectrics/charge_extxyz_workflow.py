"""CLI workflow for OVITO-ready charge Extended XYZ trajectories."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from reaxkit.analysis import ferroelectrics as _ferroelectric_tasks  # noqa: F401
from reaxkit.analysis.ferroelectrics.charge_extxyz import ChargeExtendedXYZRequest
from reaxkit.core.registry.analysis_task_registry import TASK_REGISTRY
from reaxkit.core.runtime.analysis_executor import AnalysisExecutor
from reaxkit.core.storage.storage_layout import ReaxkitStorageLayout, add_storage_cli_arguments
from reaxkit.core.utils.frame_utils import parse_frame_indices
from reaxkit.engine.reaxff.adapter_parts.io_paths import _quick_n_frames_from_control
from reaxkit.presentation.dispatcher import present_result

COMMAND = "write_trajectory_with_charges"
ALL_COMMANDS = (COMMAND,)


def build_parser(parser: argparse.ArgumentParser, *, command: str) -> argparse.ArgumentParser:
    parser.set_defaults(command=COMMAND, progress=True)
    parser.description = (
        "Generate an OVITO-compatible Extended XYZ trajectory with atom_number, "
        "charge, and delta_charge particle properties."
    )
    parser.add_argument("--engine", choices=["reaxff", "ams", "lammps"], default=None)
    parser.add_argument("--input", default=".", help="Input path used for engine detection.")
    parser.add_argument("--run-dir", default=".", help="Fallback simulation directory.")
    parser.add_argument("--fort7", default="fort.7", help="Dynamic-charge source.")
    parser.add_argument("--fort78", default="fort.78", help="Applied electric-field source.")
    parser.add_argument("--xmolout", default="xmolout", help="Coordinate and atom-type source.")
    parser.add_argument("--summary", default=None, help="Optional summary.txt metadata source.")
    parser.add_argument("--frames", nargs="*", default=None, help="Frames, e.g. 0:101:10.")
    parser.add_argument("--every", type=int, default=1, help="Keep every Nth selected frame.")
    parser.add_argument(
        "--include-electric-field",
        action="store_true",
        help="Add an iteration-matched electric_field attribute to every frame header.",
    )
    parser.add_argument(
        "--field-direction",
        choices=["x", "y", "z"],
        default="z",
        help="Electric-field component written with --include-electric-field (default: z).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Extended XYZ destination; overrides --output-dir.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Directory where charges_delta_charges.extxyz will be written."
        ),
    )
    parser.add_argument(
        "--precision",
        type=int,
        default=8,
        help="Significant digits for coordinates and floating-point properties.",
    )
    parser.add_argument("--control", default="control", help="Control file used for frame progress.")
    parser.add_argument("--log", choices=["verbose", "quiet"], default="quiet")
    add_storage_cli_arguments(parser)
    return parser


def build_request(args: argparse.Namespace) -> ChargeExtendedXYZRequest:
    return ChargeExtendedXYZRequest(
        frames=parse_frame_indices(args.frames),
        every=int(args.every),
        precision=int(args.precision),
        include_electric_field=bool(args.include_electric_field),
        field_direction=str(args.field_direction),
    )


def _artifact_directory(args: argparse.Namespace) -> Path:
    if args.output_dir is not None:
        return Path(args.output_dir).resolve()
    analysis_id = args.analysis_id or args.run_id or getattr(args, "_analysis_id", None) or "analysis"
    layout = ReaxkitStorageLayout(project_root=Path(args.project_root))
    return layout.analysis_root / COMMAND / str(analysis_id)


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


def _output_path(args: argparse.Namespace) -> Path:
    if args.output is not None:
        output = Path(args.output)
        if output.suffix.lower() not in {".xyz", ".extxyz"}:
            output = output / "charges_delta_charges.extxyz"
        return output.resolve()
    return (_artifact_directory(args) / "charges_delta_charges.extxyz").resolve()


def run_main(command: str, args: argparse.Namespace) -> int:
    base_request = build_request(args)
    expected_frames = (
        len(set([*base_request.frames, 0]))
        if base_request.frames is not None
        else _quick_n_frames_from_control(_control_file(args))
    )
    request = replace(
        base_request,
        _output_path=str(_output_path(args)),
        _expected_frames=expected_frames,
    )
    runtime_args = vars(args).copy()
    runtime_args["scope"] = "total"
    runtime_args["cache"] = False
    result = AnalysisExecutor().run(
        TASK_REGISTRY[COMMAND](),
        request,
        runtime_args,
    )
    present_result(COMMAND, result, args)
    print(f"Wrote OVITO Extended XYZ trajectory: {result.output_path}")
    return 0


__all__ = ["ALL_COMMANDS", "COMMAND", "build_parser", "build_request", "run_main"]
