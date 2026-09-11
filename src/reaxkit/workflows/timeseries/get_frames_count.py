"""Dedicated workflow for counting frames in canonical trajectory data."""

from __future__ import annotations

import argparse

from reaxkit.analysis.trajectory.frame_count import (
    FramesCountRequest,
    FramesCountResult,
    FramesCountTask,
)
from reaxkit.core.platform.engine_resolver import resolve_engine
from reaxkit.core.runtime.analysis_executor import AnalysisExecutor
from reaxkit.core.storage.storage_layout import add_storage_cli_arguments

COMMAND = "get_frames_count"


def build_parser(parser: argparse.ArgumentParser, *, command: str) -> argparse.ArgumentParser:
    """Configure the ``get_frames_count`` command parser."""
    parser.set_defaults(command=command, progress=False)
    parser.formatter_class = argparse.RawTextHelpFormatter
    parser.description = (
        "Print the number of frames in trajectory data loaded by any supported engine.\n\n"
        "Examples:\n"
        "  reaxkit get-frames-count --input runs/reaxff/xmolout\n"
        "  reaxkit get-frames-count --engine lammps --input dump.lammpstrj\n"
        "  reaxkit get-frames-count --engine ams --input reaxout.rkf"
    )
    parser.add_argument(
        "trajectory",
        nargs="?",
        default=None,
        help="Trajectory file or run directory used for engine detection.",
    )
    parser.add_argument(
        "--input",
        "--file",
        dest="input",
        default=".",
        help="Trajectory file or run directory (alternative to the positional path).",
    )
    parser.add_argument("--engine", choices=["reaxff", "ams", "lammps"], default=None)
    parser.add_argument("--run-dir", default=".", help="Run directory used for trajectory discovery.")
    parser.add_argument("--xmolout", default=None, help="Explicit ReaxFF xmolout path.")
    parser.add_argument("--dump", default=None, help="Explicit LAMMPS dump path.")
    parser.add_argument("--rkf", default=None, help="Explicit AMS RKF/KF path.")
    parser.add_argument("--log", choices=["verbose", "quiet"], default="quiet")
    add_storage_cli_arguments(parser)
    return parser


def build_request(args: argparse.Namespace) -> FramesCountRequest:
    """Build a frame-count request from parsed CLI arguments."""
    _ = args
    return FramesCountRequest()


def _normalize_trajectory_source(args: argparse.Namespace) -> None:
    """Expose a general trajectory path through each adapter's path hint."""
    input_value = getattr(args, "input", None)
    input_source = input_value if input_value and str(input_value) != "." else None
    general_source = input_source or getattr(args, "trajectory", None)
    explicit_source = (
        general_source
        or getattr(args, "xmolout", None)
        or getattr(args, "dump", None)
        or getattr(args, "rkf", None)
    )
    if explicit_source:
        args.input = str(explicit_source)
    if general_source:
        for name in ("xmolout", "dump", "rkf"):
            if not getattr(args, name, None):
                setattr(args, name, str(general_source))


def _quick_frames_count(args: argparse.Namespace) -> int | None:
    """Ask the resolved engine for a format-specific metadata/frame probe."""
    import reaxkit.engine  # noqa: F401  (register engine adapters)

    input_value = str(getattr(args, "input", ".") or ".")
    detection_path = input_value if input_value != "." else str(getattr(args, "run_dir", ".") or ".")
    adapter = resolve_engine(detection_path, engine=getattr(args, "engine", None))
    probe = getattr(adapter, "quick_n_frames", None)
    if not callable(probe):
        return None
    count = probe(vars(args))
    return None if count is None else int(count)


def run_main(command: str, args: argparse.Namespace) -> int:
    """Use an engine fast probe, falling back to canonical trajectory loading."""
    _ = command
    _normalize_trajectory_source(args)
    count = _quick_frames_count(args)
    if count is None:
        request = build_request(args)
        result = AnalysisExecutor().run(FramesCountTask(), request, vars(args))
        count = result.count
    print(count)
    return 0


__all__ = [
    "COMMAND",
    "FramesCountRequest",
    "FramesCountResult",
    "build_parser",
    "build_request",
    "run_main",
]
