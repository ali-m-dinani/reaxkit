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
    parser.formatter_class = argparse.RawTextHelpFormatter
    parser.description = (
        "Write an OVITO-compatible trajectory with atomic charges and charge changes.\n"
        "\n"
        "Use existing coordinates and charges to visualize charge transfer in OVITO.\n"
        "The Extended XYZ output includes atom_number, charge, and delta_charge properties.\n"
        "Optional electric-field values are matched by iteration.\n"
        "\n"
        "Examples:\n"
        "  1. Export charges:\n"
        "     reaxkit write-trajectory-with-charges --fort7 fort.7 --xmolout xmolout --output charges.extxyz\n"
        "\n"
        "  2. Include the applied field:\n"
        "     reaxkit write-trajectory-with-charges --fort7 fort.7 --xmolout xmolout --fort78 fort.78 --include-electric-field --field-direction z --output charges_field.extxyz"
    )
    parser.add_argument("--engine", choices=["reaxff", "ams", "lammps"], default=None, help="Engine used to load simulation inputs. Example: --engine reaxff, which selects ReaxFF input readers.")
    parser.add_argument("--input", default=".", help="Input path used for engine detection. Example: --input runs/heating, which detects the engine from that run.")
    parser.add_argument("--run-dir", default=".", help="Fallback simulation directory. Example: --run-dir runs/heating, which uses that directory for fallback discovery.")
    parser.add_argument("--fort7", default="fort.7", help="Dynamic-charge source. Example: --fort7 runs/heating/fort.7, which reads atomic charges and connectivity from that file.")
    parser.add_argument("--fort78", default="fort.78", help="Applied electric-field source. Example: --fort78 runs/heating/fort.78, which reads electric-field data from that file.")
    parser.add_argument("--xmolout", default="xmolout", help="Coordinate and atom-type source. Example: --xmolout runs/heating/xmolout, which reads trajectory coordinates from that file.")
    parser.add_argument("--summary", default=None, help="Optional summary.txt metadata source. Example: --summary runs/heating/summary.txt, which reads simulation summary values from that file.")
    parser.add_argument("--frames", nargs="*", default=None, help="Frames, e.g. 0:101:10. Example: --frames 0:20:2, which includes source frames 0, 2, ..., 18.")
    parser.add_argument("--every", type=int, default=1, help="Keep every Nth selected frame. Example: --every 5, which keeps every fifth selected frame.")
    parser.add_argument(
        "--include-electric-field",
        action="store_true",
        help="Add an iteration-matched electric_field attribute to every frame header. Example: --include-electric-field, which adds iteration-matched field data to exported frames.",
    )
    parser.add_argument(
        "--field-direction",
        choices=["x", "y", "z"],
        default="z",
        help="Electric-field component written with --include-electric-field (default: z). Example: --field-direction z, which selects the z-directed electric-field component.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Extended XYZ destination; overrides --output-dir. Example: --output charges.extxyz, which writes generated output to charges.extxyz.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Directory where charges_delta_charges.extxyz will be written. Example: --output-dir analysis/charges, which writes generated artifacts beneath that directory."
        ),
    )
    parser.add_argument(
        "--precision",
        type=int,
        default=8,
        help="Significant digits for coordinates and floating-point properties. Example: --precision 8, which writes floating-point values with eight significant digits.",
    )
    parser.add_argument("--control", default="control", help="Control file used for frame progress. Example: --control runs/heating/control, which reads simulation cadence and timestep metadata.")
    parser.add_argument("--log", choices=["verbose", "quiet"], default="quiet", help="Runtime logging verbosity. Example: --log verbose, which prints detailed execution messages.")
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
