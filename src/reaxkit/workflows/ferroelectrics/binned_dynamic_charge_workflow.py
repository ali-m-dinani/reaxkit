"""CLI workflow for spatially binned dynamic charges."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from reaxkit.analysis import ferroelectrics as _ferroelectric_tasks  # noqa: F401
from reaxkit.analysis.ferroelectrics.binned_dynamic_charge import (
    BinnedDynamicChargeRequest,
    generate_binned_charge_heatmaps,
)
from reaxkit.core.registry.analysis_task_registry import TASK_REGISTRY
from reaxkit.core.runtime.analysis_executor import AnalysisExecutor
from reaxkit.core.storage.storage_layout import ReaxkitStorageLayout, add_storage_cli_arguments
from reaxkit.core.utils.frame_utils import parse_frame_indices
from reaxkit.engine.reaxff.adapter_parts.io_paths import _quick_n_frames_from_control
from reaxkit.presentation.dispatcher import present_result

COMMAND = "get_binned_dynamic_charges"
ALL_COMMANDS = (COMMAND,)


def build_parser(parser: argparse.ArgumentParser, *, command: str) -> argparse.ArgumentParser:
    """Configure fixed-frame-zero charge binning and heatmap output."""

    parser.set_defaults(command=COMMAND, progress=True)
    parser.description = (
        "Project atoms onto xy, xz, or yz using frame 0, sum charge and "
        "delta_charge through the perpendicular direction, and write globally "
        "scaled 2-D heatmaps."
    )
    parser.add_argument("--engine", choices=["reaxff", "ams", "lammps"], default=None)
    parser.add_argument("--input", default=".", help="Input path used for engine detection.")
    parser.add_argument("--run-dir", default=".", help="Fallback simulation directory.")
    parser.add_argument("--fort7", default="fort.7", help="Dynamic-charge source.")
    parser.add_argument("--xmolout", default="xmolout", help="Coordinate and atom-id source.")
    parser.add_argument("--summary", default=None, help="Optional summary.txt metadata source.")
    parser.add_argument(
        "--plane", choices=["xy", "xz", "yz"], default="xz", help="Heatmap plane."
    )
    parser.add_argument("--bins-x", type=int, default=None, help="Number of bins along x.")
    parser.add_argument("--bins-y", type=int, default=None, help="Number of bins along y.")
    parser.add_argument("--bins-z", type=int, default=None, help="Number of bins along z.")
    parser.add_argument("--frames", nargs="*", default=None, help="Frames, e.g. 0:101:10.")
    parser.add_argument("--every", type=int, default=1, help="Keep every Nth selected frame.")
    parser.add_argument("--dpi", type=int, default=180, help="Heatmap resolution in dots per inch.")
    parser.add_argument(
        "--average",
        action="store_true",
        help=(
            "Add average_charge and average_delta_charge to the CSV and plot "
            "those per-particle averages instead of bin sums."
        ),
    )
    parser.add_argument("--control", default="control", help="Control file used for frame progress.")
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Write the bin CSV without generating per-frame heatmaps.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output root for binned_dynamic_charge.csv and heatmaps/.",
    )
    parser.add_argument("--log", choices=["verbose", "quiet"], default="quiet")
    add_storage_cli_arguments(parser)
    return parser


def build_request(args: argparse.Namespace) -> BinnedDynamicChargeRequest:
    return BinnedDynamicChargeRequest(
        plane=str(args.plane),
        bins_x=None if args.bins_x is None else int(args.bins_x),
        bins_y=None if args.bins_y is None else int(args.bins_y),
        bins_z=None if args.bins_z is None else int(args.bins_z),
        selected_frames=parse_frame_indices(args.frames),
        every=int(args.every),
        average=bool(args.average),
    )


def _artifact_directory(args: argparse.Namespace) -> Path:
    if args.output_dir is not None:
        return Path(args.output_dir).resolve()
    analysis_id = (
        args.analysis_id
        or args.run_id
        or getattr(args, "_analysis_id", None)
        or "analysis"
    )
    layout = ReaxkitStorageLayout(project_root=Path(args.project_root))
    return layout.analysis_root / COMMAND / str(analysis_id)


def _control_file(args: argparse.Namespace) -> Path:
    configured = Path(str(getattr(args, "control", "control")))
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
    """Run fixed-bin aggregation and write its table and heatmaps."""

    output = _artifact_directory(args)
    output.mkdir(parents=True, exist_ok=True)
    request = replace(
        build_request(args),
        _expected_frames=_quick_n_frames_from_control(_control_file(args)),
    )
    runtime_args = vars(args).copy()
    runtime_args["frames"] = None  # Frame 0 is always needed to define the grid.
    runtime_args["scope"] = "total"
    runtime_args["_quick_charge_only"] = True
    runtime_args["cache"] = False
    result = AnalysisExecutor().run(
        TASK_REGISTRY[COMMAND](),
        request,
        runtime_args,
    )
    csv_path = output / "binned_dynamic_charge.csv"
    result.table.to_csv(csv_path, index=False)
    args.suppress_table = True
    present_result(COMMAND, result, args)

    written = []
    if not args.skip_plots:
        written = generate_binned_charge_heatmaps(
            result,
            output,
            dpi=int(args.dpi),
            progress=bool(args.progress),
        )
    print(f"Wrote binned charge table: {csv_path}")
    if written:
        print(f"Wrote {len(written):,} globally scaled heatmaps under {output / 'heatmaps'}")
    return 0


__all__ = ["ALL_COMMANDS", "COMMAND", "build_parser", "build_request", "run_main"]
