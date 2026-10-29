"""CLI workflow for spatially binned dynamic charges."""

from __future__ import annotations

from reaxkit.presentation.workflow_artifacts import write_workflow_tables

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
    parser.formatter_class = argparse.RawTextHelpFormatter
    parser.description = (
        "Project dynamic charges into fixed spatial bins and write charge heatmaps.\n"
        "\n"
        "Use frame-zero positions to compare charge changes in the same spatial regions.\n"
        "Values are summed through the omitted direction; --average reports per-particle averages.\n"
        "This analyzes existing trajectory and charge files.\n"
        "\n"
        "Examples:\n"
        "  1. Summed charges in the x-z plane:\n"
        "     reaxkit get-binned-dynamic-charges --fort7 fort.7 --xmolout xmolout --plane xz --bins-x 40 --bins-z 40\n"
        "\n"
        "  2. Per-particle averages without plots:\n"
        "     reaxkit get-binned-dynamic-charges --fort7 fort.7 --xmolout xmolout --average --skip-plots"
    )
    parser.add_argument("--engine", choices=["reaxff", "ams", "lammps"], default=None, help="Engine used to load simulation inputs. Example: --engine reaxff, which selects ReaxFF input readers.")
    parser.add_argument("--input", default=".", help="Input path used for engine detection. Example: --input runs/heating, which detects the engine from that run.")
    parser.add_argument("--run-dir", default=".", help="Fallback simulation directory. Example: --run-dir runs/heating, which uses that directory for fallback discovery.")
    parser.add_argument("--fort7", default="fort.7", help="Dynamic-charge source. Example: --fort7 runs/heating/fort.7, which reads atomic charges and connectivity from that file.")
    parser.add_argument("--xmolout", default="xmolout", help="Coordinate and atom-id source. Example: --xmolout runs/heating/xmolout, which reads trajectory coordinates from that file.")
    parser.add_argument("--summary", default=None, help="Optional summary.txt metadata source. Example: --summary runs/heating/summary.txt, which reads simulation summary values from that file.")
    parser.add_argument(
        "--plane", choices=["xy", "xz", "yz"], default="xz", help="Heatmap plane. Example: --plane xz, which projects onto the x-z plane."
    )
    parser.add_argument("--bins-x", type=int, default=None, help="Number of bins along x. Example: --bins-x 40, which splits the x extent into 40 bins.")
    parser.add_argument("--bins-y", type=int, default=None, help="Number of bins along y. Example: --bins-y 40, which splits the y extent into 40 bins.")
    parser.add_argument("--bins-z", type=int, default=None, help="Number of bins along z. Example: --bins-z 40, which splits the z extent into 40 bins.")
    parser.add_argument("--frames", nargs="*", default=None, help="Frames, e.g. 0:101:10. Example: --frames 0:20:2, which includes source frames 0, 2, ..., 18.")
    parser.add_argument("--every", type=int, default=1, help="Keep every Nth selected frame. Example: --every 5, which keeps every fifth selected frame.")
    parser.add_argument("--dpi", type=int, default=180, help="Heatmap resolution in dots per inch. Example: --dpi 300, which writes figures at 300 dots per inch.")
    parser.add_argument(
        "--average",
        action="store_true",
        help=(
            "Add average_charge and average_delta_charge to the CSV and plot those per-particle averages instead of bin sums. Example: --average, which plots per-particle averages instead of summed bin charges."
        ),
    )
    parser.add_argument("--control", default="control", help="Control file used for frame progress. Example: --control runs/heating/control, which reads simulation cadence and timestep metadata.")
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Write the bin CSV without generating per-frame heatmaps. Example: --skip-plots, which exports the data without generating heatmaps.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output root for binned_dynamic_charge.csv and heatmaps/. Example: --output-dir analysis/charges, which writes generated artifacts beneath that directory.",
    )
    parser.add_argument("--log", choices=["verbose", "quiet"], default="quiet", help="Runtime logging verbosity. Example: --log verbose, which prints detailed execution messages.")
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
    write_workflow_tables({csv_path: result.table}, args=args)
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
