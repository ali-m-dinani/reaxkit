"""CLI workflow for selected-atom charges versus applied electric field."""

from __future__ import annotations

from reaxkit.presentation.workflow_artifacts import write_workflow_csv

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np

from reaxkit.analysis import ferroelectrics as _ferroelectric_tasks  # noqa: F401
from reaxkit.analysis.ferroelectrics.charge_field import (
    ChargeFieldRequest,
    ChargeFieldResult,
    ChargeFieldTask,
)
from reaxkit.core.platform.engine_resolver import resolve_engine
from reaxkit.core.registry.analysis_task_registry import TASK_REGISTRY
from reaxkit.core.runtime.analysis_executor import AnalysisExecutor
from reaxkit.core.runtime.progress import resolve_reporter
from reaxkit.core.storage.storage_layout import ReaxkitStorageLayout, add_storage_cli_arguments
from reaxkit.core.utils.frame_utils import parse_frame_indices
from reaxkit.domain.data_models import ElectricFieldData
from reaxkit.engine.reaxff.adapter_parts.io_paths import _quick_n_frames_from_control
from reaxkit.presentation.convert import convert_xaxis
from reaxkit.presentation.dispatcher import present_result

COMMAND = "get_charge_vs_electric_field"
ALL_COMMANDS = (COMMAND,)


def build_parser(parser: argparse.ArgumentParser, *, command: str) -> argparse.ArgumentParser:
    parser.set_defaults(command=COMMAND, progress=True)
    parser.formatter_class = argparse.RawTextHelpFormatter
    parser.description = (
        "Export and plot selected-atom charges alongside the applied electric field.\n"
        "\n"
        "Use this to compare charge response with iteration-matched field values in an existing run.\n"
        "Select atom IDs and the field component; time axes require control-file metadata.\n"
        "\n"
        "Examples:\n"
        "  1. Compare charge and field along z:\n"
        "     reaxkit get-charge-vs-electric-field --atom-numbers 1 2 --fort7 fort.7 --fort78 fort.78 --xmolout xmolout --field-direction z\n"
        "\n"
        "  2. Export values on a time axis:\n"
        "     reaxkit get-charge-vs-electric-field --atom-numbers 1 2 --x-axis time --control control --export charge_field.csv"
    )
    parser.add_argument("--engine", choices=["reaxff", "ams", "lammps"], default=None, help="Engine used to load simulation inputs. Example: --engine reaxff, which selects ReaxFF input readers.")
    parser.add_argument("--input", default=".", help="Input path used for engine detection. Example: --input runs/heating, which detects the engine from that run.")
    parser.add_argument("--run-dir", default=".", help="Fallback simulation directory. Example: --run-dir runs/heating, which uses that directory for fallback discovery.")
    parser.add_argument("--fort7", default="fort.7", help="Dynamic-charge source. Example: --fort7 runs/heating/fort.7, which reads atomic charges and connectivity from that file.")
    parser.add_argument("--fort78", default="fort.78", help="Applied electric-field source. Example: --fort78 runs/heating/fort.78, which reads electric-field data from that file.")
    parser.add_argument("--xmolout", default="xmolout", help="Per-frame atom identity source. Example: --xmolout runs/heating/xmolout, which reads trajectory coordinates from that file.")
    parser.add_argument("--atom-numbers", "--atom-ids", type=int, nargs="+", required=True, help="One-based atom identifiers to include. Example: --atom-numbers 1 2, which includes only atoms 1 and 2.")
    parser.add_argument("--frames", nargs="*", default=None, help="Frames, e.g. 0:101:10. Example: --frames 0:20:2, which includes source frames 0, 2, ..., 18.")
    parser.add_argument("--every", type=int, default=1, help="Keep every Nth frame. Example: --every 5, which keeps every fifth selected frame.")
    parser.add_argument(
        "--field-direction",
        choices=["x", "y", "z"],
        default="z",
        help="Applied-field component to correlate (default: z). Example: --field-direction z, which selects the z-directed electric-field component.",
    )
    parser.add_argument(
        "--x-axis", "--xaxis", dest="x_axis", choices=["frame", "iter", "time"],
        default="frame", help="Horizontal plot axis. Example: --x-axis time, which shows physical time on the horizontal axis.",
    )
    parser.add_argument("--control", default="control", help="Control file used to derive time. Example: --control runs/heating/control, which reads simulation cadence and timestep metadata.")
    parser.add_argument("--dpi", type=int, default=180, help="Plot resolution in dots per inch. Example: --dpi 300, which writes figures at 300 dots per inch.")
    parser.add_argument("--output-dir", type=Path, default=None, help="CSV and plots output directory. Example: --output-dir analysis/charges, which writes generated artifacts beneath that directory.")
    parser.add_argument("--export", default=None, help="Optional CSV output path. Example: --export analysis.csv, which writes the result table for further analysis.")
    parser.add_argument("--log", choices=["verbose", "quiet"], default="quiet", help="Runtime logging verbosity. Example: --log verbose, which prints detailed execution messages.")
    add_storage_cli_arguments(parser)
    return parser


def build_request(args: argparse.Namespace) -> ChargeFieldRequest:
    return ChargeFieldRequest(
        atom_numbers=tuple(args.atom_numbers),
        frames=parse_frame_indices(args.frames),
        every=int(args.every),
        field_direction=str(args.field_direction),
    )


def _artifact_directory(args: argparse.Namespace) -> Path:
    if args.output_dir is not None:
        return Path(args.output_dir).resolve()
    analysis_id = args.analysis_id or args.run_id or getattr(args, "_analysis_id", None) or "analysis"
    layout = ReaxkitStorageLayout(project_root=Path(args.project_root))
    return layout.analysis_root / COMMAND / str(analysis_id)


def _axis_control_file(args: argparse.Namespace) -> str:
    configured = Path(str(args.control))
    if configured != Path("control"):
        return str(configured)
    for name in ("fort7", "fort78", "xmolout", "input", "run_dir"):
        source = Path(str(getattr(args, name, "") or ""))
        directory = source if source.is_dir() else source.parent
        candidate = directory / "control"
        if candidate.is_file():
            return str(candidate)
    return str(configured)


def _x_values(result: ChargeFieldResult, x_axis: str, control_file: str) -> tuple[np.ndarray, str]:
    if x_axis == "frame":
        return np.asarray(result.frame_indices, dtype=float), "frame"
    if x_axis == "iter":
        return np.asarray(result.iterations, dtype=float), "iteration"
    if result.time_values is not None:
        return np.asarray(result.time_values, dtype=float), "time"
    values, label = convert_xaxis(result.iterations, "time", control_file=control_file)
    return np.asarray(values, dtype=float), label


def generate_charge_field_plots(
        result: ChargeFieldResult,
        output_root: Path,
        *,
        x_axis: str,
        control_file: str = "control",
        dpi: int = 180,
        progress: bool = True,
) -> list[Path]:
    """Generate one bounded-memory dual-y-axis plot per selected atom."""

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Plot generation requires matplotlib; install reaxkit[plot].") from exc

    plot_dir = Path(output_root) / "plots"
    charge_dir = plot_dir / "charges"
    delta_dir = plot_dir / "delta_charge"
    charge_dir.mkdir(parents=True, exist_ok=True)
    delta_dir.mkdir(parents=True, exist_ok=True)
    frame_x, xlabel = _x_values(result, x_axis, control_file)
    x_by_frame = dict(zip(result.frame_indices.tolist(), frame_x.tolist()))
    groups = result.table.groupby("atom_number", sort=False)
    if progress:
        from tqdm.auto import tqdm

        groups = tqdm(
            groups,
            total=result.table["atom_number"].nunique(),
            desc="plots: generating charge-field plots",
            unit="atom",
            dynamic_ncols=True,
        )

    written: list[Path] = []
    for atom_number, group in groups:
        group = group.sort_values("frame", kind="stable")
        atom_type = str(group["atom_type"].iloc[0])
        x = group["frame"].map(x_by_frame).to_numpy(dtype=float)
        charge = group["charge"].to_numpy(dtype=float)
        delta_charge = group["delta_charge"].to_numpy(dtype=float)
        field = group["electric_field"].to_numpy(dtype=float)
        safe_type = "".join(c if c.isalnum() or c in {"-", "_"} else "_" for c in atom_type)
        suffix = f"_{safe_type}" if safe_type else ""
        filename = f"atom_{int(atom_number)}{suffix}.png"
        written.append(
            _write_dual_axis_plot(
                plt,
                x,
                charge,
                field,
                charge_dir / filename,
                atom_number=int(atom_number),
                atom_type=atom_type,
                value_label="charge",
                field_direction=result.request.field_direction,
                xlabel=xlabel,
                dpi=dpi,
            )
        )
        written.append(
            _write_dual_axis_plot(
                plt,
                x,
                delta_charge,
                field,
                delta_dir / filename,
                atom_number=int(atom_number),
                atom_type=atom_type,
                value_label="delta charge from frame 0",
                field_direction=result.request.field_direction,
                xlabel=xlabel,
                dpi=dpi,
            )
        )
    return written


def _write_dual_axis_plot(
        plt,
        x: np.ndarray,
        values: np.ndarray,
        field: np.ndarray,
        destination: Path,
        *,
        atom_number: int,
        atom_type: str,
        value_label: str,
        field_direction: str,
        xlabel: str,
        dpi: int,
) -> Path:
    """Write one charge-like series with electric field on the second y-axis."""

    figure, value_axis = plt.subplots(figsize=(7.2, 4.4))
    field_axis = value_axis.twinx()
    value_line = value_axis.plot(x, values, color="tab:blue", label=value_label)[0]
    field_line = field_axis.plot(
        x,
        field,
        color="tab:red",
        alpha=0.8,
        label=f"electric field ({field_direction})",
    )[0]
    value_axis.set_xlabel(xlabel)
    value_axis.set_ylabel(value_label, color="tab:blue")
    field_axis.set_ylabel("electric field (MV/cm)", color="tab:red")
    value_axis.tick_params(axis="y", labelcolor="tab:blue")
    field_axis.tick_params(axis="y", labelcolor="tab:red")
    value_axis.grid(alpha=0.25)
    value_axis.set_title(
        f"Atom {atom_number}{f' ({atom_type})' if atom_type else ''}: {value_label}"
    )
    value_axis.legend(handles=[value_line, field_line], loc="best")
    figure.tight_layout()
    figure.savefig(destination, dpi=int(dpi), bbox_inches="tight")
    plt.close(figure)
    return destination


def _export_csv(_command: str, result: ChargeFieldResult, args: argparse.Namespace) -> list[Path]:
    destination = Path(args.export).resolve()
    if not destination.suffix:
        destination = destination / "charge_vs_electric_field.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    write_workflow_csv(result.table, destination, index=False)
    return [destination]


def run_main(command: str, args: argparse.Namespace) -> int:
    runtime_args = vars(args).copy()
    runtime_args["_quick_charge_only"] = True
    runtime_args["cache"] = False
    control_file = _axis_control_file(args)
    base_request = build_request(args)
    expected_frames = (
        len(set([0, *(int(frame) for frame in base_request.frames)]))
        if base_request.frames is not None
        else _quick_n_frames_from_control(Path(control_file))
    )
    request = replace(base_request, _expected_frames=expected_frames)

    detection_path = args.input if Path(str(args.input)).exists() else args.run_dir
    adapter = resolve_engine(detection_path, engine=args.engine)
    field_data = adapter.load(
        ElectricFieldData,
        runtime_args,
        reporter=resolve_reporter(runtime_args),
    )
    task = TASK_REGISTRY[COMMAND]()
    if not isinstance(task, ChargeFieldTask):
        raise TypeError(f"Unexpected task registered for {COMMAND}: {type(task).__name__}")
    task.electric_field = field_data
    result = AnalysisExecutor().run(task, request, runtime_args)

    output_dir = _artifact_directory(args)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "charge_vs_electric_field.csv"
    write_workflow_csv(result.table, csv_path, index=False)
    written = generate_charge_field_plots(
        result,
        output_dir,
        x_axis=args.x_axis,
        control_file=control_file,
        dpi=args.dpi,
        progress=bool(args.progress),
    )
    args.suppress_table = True
    present_result(COMMAND, result, args, export_handler=_export_csv)
    print(f"Wrote {csv_path} and {len(written):,} charge-field plot(s) under {output_dir / 'plots'}")
    return 0


__all__ = [
    "ALL_COMMANDS",
    "COMMAND",
    "build_parser",
    "build_request",
    "generate_charge_field_plots",
    "run_main",
]
