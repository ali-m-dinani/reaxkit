"""CLI workflow for per-atom dynamic charge changes and summaries."""

from __future__ import annotations

import argparse
import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from reaxkit.analysis import ferroelectrics as _ferroelectric_tasks  # noqa: F401
from reaxkit.analysis.ferroelectrics.dynamic_charge import (
    DynamicChargeChangeRequest,
    DynamicChargeChangeResult,
)
from reaxkit.core.registry.analysis_task_registry import TASK_REGISTRY
from reaxkit.core.runtime.analysis_executor import AnalysisExecutor
from reaxkit.core.storage.storage_layout import ReaxkitStorageLayout, add_storage_cli_arguments
from reaxkit.core.utils.frame_utils import parse_frame_indices
from reaxkit.engine.reaxff.adapter_parts.io_paths import _quick_n_frames_from_control
from reaxkit.presentation.convert import convert_xaxis
from reaxkit.presentation.dispatcher import present_result

COMMAND = "get_dynamic_charge_changes"
ALL_COMMANDS = (COMMAND,)


def build_parser(parser: argparse.ArgumentParser, *, command: str) -> argparse.ArgumentParser:
    """Configure the dynamic-charge analysis command."""

    parser.set_defaults(command=COMMAND, progress=True)
    parser.formatter_class = argparse.RawTextHelpFormatter
    parser.description = (
        "Compare each atom's dynamic charge with its charge at frame zero.\n"
        "Writes per-atom and per-frame summaries plus charges.csv unless "
        "--skip-detailed-csv is supplied; --gen-plots writes one charge and "
        "one delta-charge trace per atom.\n\n"
        "Example:\n"
        "  reaxkit get_dynamic_charge_changes --fort7 fort.7 --xmolout xmolout "
        "--gen-plots --x-axis time"
    )
    parser.add_argument("--engine", choices=["reaxff", "ams", "lammps"], default=None)
    parser.add_argument("--input", default=".", help="Input path used for engine detection.")
    parser.add_argument("--run-dir", default=".", help="Fallback simulation directory.")
    parser.add_argument("--fort7", default="fort.7", help="Dynamic charge input for ReaxFF.")
    parser.add_argument("--xmolout", default="xmolout", help="Optional atom identity/time source.")
    parser.add_argument("--summary", default=None, help="Optional summary.txt metadata source.")
    parser.add_argument("--atom-numbers", "--atom-ids", type=int, nargs="+", default=None)
    parser.add_argument("--frames", nargs="*", default=None, help="Frames, e.g. 0:101:10.")
    parser.add_argument("--every", type=int, default=1, help="Keep every Nth selected frame.")
    parser.add_argument("--gen-plots", action="store_true", help="Generate per-atom PNG plots.")
    parser.add_argument(
        "--global-y-axis",
        action="store_true",
        help=(
            "Use one shared charge y-axis across atoms and one shared "
            "delta-charge y-axis across atoms."
        ),
    )
    parser.add_argument(
        "--skip-detailed-csv",
        action="store_true",
        help="Do not create the potentially very large charges.csv file.",
    )
    parser.add_argument(
        "--x-axis", "--xaxis", dest="x_axis", choices=["frame", "time", "iter"],
        default="frame", help="Horizontal axis for generated plots.",
    )
    parser.add_argument("--control", default="control", help="Control file used to derive time.")
    parser.add_argument("--dpi", type=int, default=180, help="Plot resolution in dots per inch.")
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Optional output root for charges.csv, the three summary CSVs, and plots/.",
    )
    parser.add_argument(
        "--export", default=None,
        help="Optional CSV destination; all enabled CSV outputs are written beside it.",
    )
    parser.add_argument("--log", choices=["verbose", "quiet"], default="quiet")
    add_storage_cli_arguments(parser)
    return parser


def build_request(args: argparse.Namespace) -> DynamicChargeChangeRequest:
    return DynamicChargeChangeRequest(
        atom_numbers=tuple(args.atom_numbers) if args.atom_numbers else None,
        selected_frames=parse_frame_indices(args.frames),
        every=int(args.every),
    )


def _artifact_directory(args: argparse.Namespace) -> Path:
    if args.output_dir is not None:
        return Path(args.output_dir).resolve()
    analysis_id = args.analysis_id or args.run_id or getattr(args, "_analysis_id", None) or "analysis"
    layout = ReaxkitStorageLayout(project_root=Path(args.project_root))
    return layout.analysis_root / COMMAND / str(analysis_id)


def _write_requested_output(result: DynamicChargeChangeResult, args: argparse.Namespace) -> None:
    output = _artifact_directory(args)
    output.mkdir(parents=True, exist_ok=True)
    if not bool(getattr(args, "skip_detailed_csv", False)):
        detail_path = output / "charges.csv"
        existing = Path(result.detail_csv_path) if result.detail_csv_path else None
        if existing is None or not existing.is_file():
            result.charges.to_csv(detail_path, index=False)
            result.detail_csv_path = str(detail_path)
        elif existing.resolve() != detail_path.resolve():
            shutil.copyfile(existing, detail_path)
            result.detail_csv_path = str(detail_path)
    result.summary.to_csv(output / "summary_per_atom.csv", index=False)
    result.summary_per_frame_for_all_atoms.to_csv(
        output / "summary_per_frame_for_all_atoms.csv",
        index=False,
    )
    result.summary_per_frame_per_atom_type.to_csv(
        output / "summary_per_frame_per_atom_type.csv",
        index=False,
    )


def _export_csvs(_command: str, result: DynamicChargeChangeResult, args: argparse.Namespace) -> list[Path]:
    detail_path = Path(args.export).resolve()
    if not detail_path.suffix:
        detail_path = detail_path / "charges.csv"
    detail_path.parent.mkdir(parents=True, exist_ok=True)
    if not bool(getattr(args, "skip_detailed_csv", False)):
        existing = Path(result.detail_csv_path) if result.detail_csv_path else None
        if existing is not None and existing.is_file():
            if existing.resolve() != detail_path.resolve():
                shutil.copyfile(existing, detail_path)
        else:
            result.charges.to_csv(detail_path, index=False)
    result.summary.to_csv(detail_path.with_name("summary_per_atom.csv"), index=False)
    result.summary_per_frame_for_all_atoms.to_csv(
        detail_path.with_name("summary_per_frame_for_all_atoms.csv"),
        index=False,
    )
    result.summary_per_frame_per_atom_type.to_csv(
        detail_path.with_name("summary_per_frame_per_atom_type.csv"),
        index=False,
    )
    return [detail_path.parent]


def _plot_x_values(
        result: DynamicChargeChangeResult,
        args: argparse.Namespace,
) -> tuple[dict[int, float], str]:
    frames = np.asarray(result.frame_indices, dtype=int)
    if args.x_axis == "frame":
        return dict(zip(frames.tolist(), frames.astype(float).tolist())), "frame"
    iterations = np.asarray(result.iterations, dtype=int)
    if args.x_axis == "iter":
        return dict(zip(frames.tolist(), iterations.astype(float).tolist())), "iteration"
    if result.time_values is not None:
        times = np.asarray(result.time_values, dtype=float)
        return dict(zip(frames.tolist(), times.tolist())), "time"
    times, label = convert_xaxis(iterations, "time", control_file=str(args.control))
    return dict(zip(frames.tolist(), np.asarray(times, dtype=float).tolist())), label


def _axis_control_file(args: argparse.Namespace) -> str:
    """Resolve the default control file beside the simulation inputs."""

    configured = Path(str(args.control))
    if configured != Path("control"):
        return str(configured)
    for name in ("fort7", "xmolout", "summary", "input", "run_dir"):
        raw = getattr(args, name, None)
        if not raw:
            continue
        source = Path(str(raw))
        directory = source if source.is_dir() else source.parent
        candidate = directory / "control"
        if candidate.is_file():
            return str(candidate)
    return str(configured)


def generate_atom_charge_plots(
        result: DynamicChargeChangeResult,
        output_root: Path,
        *,
        x_axis: str,
        control_file: str = "control",
        dpi: int = 180,
        progress: bool = True,
        global_y_axis: bool = False,
) -> list[Path]:
    """Write charge and delta-charge time-series plots for every selected atom."""

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("Plot generation requires matplotlib; install reaxkit[plot].") from exc

    plotting_args = argparse.Namespace(x_axis=x_axis, control=control_file)
    x_by_frame, xlabel = _plot_x_values(result, plotting_args)
    plot_root = Path(output_root) / "plots"
    charge_dir = plot_root / "charges"
    delta_dir = plot_root / "delta_charge"
    charge_dir.mkdir(parents=True, exist_ok=True)
    delta_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    charge_ylim = (
        _global_y_limits(result.summary, "min(charge)", "max(charge)")
        if global_y_axis
        else None
    )
    delta_ylim = (
        _global_y_limits(result.summary, "min(delta_charge)", "max(delta_charge)")
        if global_y_axis
        else None
    )

    if result.charge_matrix_path and result.charge_matrix_shape:
        matrix = np.memmap(
            result.charge_matrix_path,
            mode="r",
            dtype=np.float64,
            shape=tuple(result.charge_matrix_shape),
        )
        x = np.asarray([x_by_frame[int(frame)] for frame in result.frame_indices], dtype=float)
        records = zip(
            result.atom_numbers.tolist(),
            result.atom_types,
            result.matrix_columns.tolist(),
            result.baseline_charges.tolist(),
            strict=False,
        )
        if progress:
            from tqdm.auto import tqdm

            records = tqdm(
                records,
                total=len(result.atom_numbers),
                desc="plots: generating per-atom charge plots",
                unit="atom",
                dynamic_ncols=True,
            )
        for atom_number, atom_type, matrix_column, baseline in records:
            charge_values = np.asarray(matrix[:, int(matrix_column)], dtype=float)
            written.extend(
                _write_atom_plot_pair(
                    plt,
                    x,
                    charge_values,
                    charge_values - baseline,
                    int(atom_number),
                    str(atom_type),
                    charge_dir,
                    delta_dir,
                    xlabel=xlabel,
                    dpi=dpi,
                    charge_ylim=charge_ylim,
                    delta_ylim=delta_ylim,
                )
            )
        del matrix
        return written

    groups = result.charges.groupby("atom_number", sort=False)
    if progress:
        from tqdm.auto import tqdm

        groups = tqdm(
            groups,
            total=result.charges["atom_number"].nunique(),
            desc="plots: generating per-atom charge plots",
            unit="atom",
            dynamic_ncols=True,
        )
    for atom_number, group in groups:
        group = group.sort_values("frame", kind="stable")
        x = group["frame"].map(x_by_frame).to_numpy(dtype=float)
        atom_type = str(group["atom_type"].iloc[0])
        written.extend(
            _write_atom_plot_pair(
                plt,
                x,
                pd.to_numeric(group["charge"], errors="coerce").to_numpy(dtype=float),
                pd.to_numeric(group["delta_charge"], errors="coerce").to_numpy(dtype=float),
                int(atom_number),
                atom_type,
                charge_dir,
                delta_dir,
                xlabel=xlabel,
                dpi=dpi,
                charge_ylim=charge_ylim,
                delta_ylim=delta_ylim,
            )
        )
    return written


def _global_y_limits(
        summary: pd.DataFrame,
        minimum_column: str,
        maximum_column: str,
) -> tuple[float, float] | None:
    """Return padded finite limits spanning all plotted atoms."""

    minimum = pd.to_numeric(summary.get(minimum_column), errors="coerce").to_numpy(dtype=float)
    maximum = pd.to_numeric(summary.get(maximum_column), errors="coerce").to_numpy(dtype=float)
    finite_minimum = minimum[np.isfinite(minimum)]
    finite_maximum = maximum[np.isfinite(maximum)]
    if finite_minimum.size == 0 or finite_maximum.size == 0:
        return None
    lower = float(np.min(finite_minimum))
    upper = float(np.max(finite_maximum))
    span = upper - lower
    padding = 0.05 * span if span > 0.0 else max(0.05 * abs(lower), 0.05)
    return lower - padding, upper + padding


def _write_atom_plot_pair(
        plt,
        x: np.ndarray,
        charges: np.ndarray,
        delta_charges: np.ndarray,
        atom_number: int,
        atom_type: str,
        charge_dir: Path,
        delta_dir: Path,
        *,
        xlabel: str,
        dpi: int,
        charge_ylim: tuple[float, float] | None = None,
        delta_ylim: tuple[float, float] | None = None,
) -> list[Path]:
    """Render one atom at a time so plot memory stays bounded."""

    safe_type = "".join(c if c.isalnum() or c in {"-", "_"} else "_" for c in atom_type)
    suffix = f"_{safe_type}" if safe_type else ""
    filename = f"atom_{atom_number}{suffix}.png"
    outputs: list[Path] = []
    for values, destination, ylabel, ylim in (
            (charges, charge_dir / filename, "charge", charge_ylim),
            (
                    delta_charges,
                    delta_dir / filename,
                    "delta charge from frame 0",
                    delta_ylim,
            ),
    ):
        figure, axis = plt.subplots(figsize=(7.0, 4.2))
        axis.plot(x, values, linewidth=1.5)
        axis.set_title(f"Atom {atom_number}{f' ({atom_type})' if atom_type else ''}")
        axis.set_xlabel(xlabel)
        axis.set_ylabel(ylabel)
        if ylim is not None:
            axis.set_ylim(*ylim)
        axis.grid(alpha=0.3)
        figure.tight_layout()
        figure.savefig(destination, dpi=int(dpi), bbox_inches="tight")
        plt.close(figure)
        outputs.append(destination)
    return outputs


def run_main(command: str, args: argparse.Namespace) -> int:
    """Run charge-change analysis, persist two CSVs, and optionally plot."""

    runtime_args = vars(args).copy()
    runtime_args["frames"] = None
    runtime_args["_quick_charge_only"] = True
    runtime_args["cache"] = False
    output_dir = _artifact_directory(args)
    output_dir.mkdir(parents=True, exist_ok=True)
    expected_frames = _quick_n_frames_from_control(Path(_axis_control_file(args)))
    request = replace(
        build_request(args),
        _detail_csv_path=(
            None if args.skip_detailed_csv else str(output_dir / "charges.csv")
        ),
        _matrix_path=str(output_dir / ".dynamic_charge_matrix.dat"),
        _expected_frames=expected_frames,
        _retain_matrix=bool(args.gen_plots),
    )
    result = AnalysisExecutor().run(
        TASK_REGISTRY[COMMAND](),
        request,
        runtime_args,
    )
    _write_requested_output(result, args)
    args.suppress_table = True
    present_result(COMMAND, result, args, export_handler=_export_csvs)
    if args.gen_plots:
        try:
            written = generate_atom_charge_plots(
                result,
                output_dir,
                x_axis=args.x_axis,
                control_file=_axis_control_file(args),
                dpi=args.dpi,
                progress=bool(args.progress),
                global_y_axis=bool(getattr(args, "global_y_axis", False)),
            )
        finally:
            if result.charge_matrix_path:
                Path(result.charge_matrix_path).unlink(missing_ok=True)
        print(f"Wrote {len(written):,} atom charge plots under {_artifact_directory(args) / 'plots'}")
    return 0


__all__ = [
    "ALL_COMMANDS",
    "COMMAND",
    "build_parser",
    "build_request",
    "generate_atom_charge_plots",
    "run_main",
]
