"""Direct command workflow for connectivity analyses."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import reaxkit.engine  # noqa: F401

from reaxkit.analysis import connectivity as _connectivity_tasks  # noqa: F401
from reaxkit.analysis import trajectory as _trajectory_tasks  # noqa: F401
from reaxkit.analysis.connectivity.connectivity import (
    BondEventsRequest,
    BondTimeseriesRequest,
    ConnectionListRequest,
    ConnectionStatsRequest,
    ConnectionTableRequest,
)
from reaxkit.analysis.connectivity.coordination import CoordinationStatusRequest
from reaxkit.analysis.connectivity.hybridization import HybridizationStatusRequest
from reaxkit.analysis.trajectory.relabel import TrajectoryRelabelByCoordinationRequest
from reaxkit.core.analysis_executor import AnalysisExecutor
from reaxkit.core.engine_registry import resolve_engine
from reaxkit.core.analysis_task_registry import TASK_REGISTRY
from reaxkit.core.command_alias_resolver import resolve_command_name
from reaxkit.core.storage_layout import add_storage_cli_arguments, normalize_storage_args
from reaxkit.domain.data_models import ConnectivityTrajectoryData, ForceFieldParametersData
from reaxkit.presentation.convert import convert_xaxis
from reaxkit.presentation.dispatcher import export_result_csv, present_result

CONNECTIVITY_COMMANDS = (
    "connection_list",
    "connection_table",
    "connection_stats",
    "bond_timeseries",
    "bond_events",
    "coordination",
    "coordination_relabel",
    "hybridization",
)


def _parse_frames(values: list[int] | None) -> list[int] | None:
    return None if values is None else [int(v) for v in values]


def _parse_kv_map(spec: str | None, *, value_cast=float) -> dict[str, float]:
    if not spec:
        return {}
    out: dict[str, float] = {}
    for item in spec.split(","):
        token = item.strip()
        if not token:
            continue
        if "=" not in token:
            raise ValueError(f"Invalid mapping entry {token!r}; use key=value.")
        key, value = token.split("=", 1)
        out[key.strip()] = value_cast(value.strip())
    return out


def _parse_element_hybridizations(spec: str | None) -> dict[str, dict[str, float]]:
    if not spec:
        return {}
    out: dict[str, dict[str, float]] = {}
    for block in spec.split(";"):
        token = block.strip()
        if not token:
            continue
        if ":" not in token:
            raise ValueError(f"Invalid element hybridization block {token!r}; use C:sp=1,sp2=2.")
        element, mapping = token.split(":", 1)
        out[element.strip()] = _parse_kv_map(mapping, value_cast=float)
    return out


def _parse_status_labels(spec: str | None) -> dict[int, str]:
    out = {-1: "U", 0: "C", 1: "O"}
    if not spec:
        return out
    for item in spec.split(","):
        token = item.strip()
        if not token:
            continue
        if "=" not in token:
            raise ValueError(f"Invalid label entry {token!r}; use -1=U,0=C,1=O.")
        key, value = token.split("=", 1)
        ikey = int(key.strip())
        if ikey not in (-1, 0, 1):
            raise ValueError("Status keys must be -1, 0, or 1.")
        out[ikey] = value.strip()
    return out


def _add_runtime_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--engine", choices=["reaxff", "ams", "lammps"], default=None)
    parser.add_argument("--input", default=".", help="Input file or directory for engine resolution")
    parser.add_argument("--run-dir", "--dir", dest="run_dir", default=".", help="Run directory fallback for engine detection")
    parser.add_argument("--fort7", default="fort.7", help="Path to fort.7")
    parser.add_argument("--xmolout", default="xmolout", help="Path to xmolout")
    parser.add_argument("--summary", default=None, help="Optional summary.txt path")
    parser.add_argument("--log", choices=["verbose", "quiet"], default=None, help="Logging level")
    add_storage_cli_arguments(parser)


def _add_presentation_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--plot", choices=["single", "subplot"], default=None, help="Render a plot")
    parser.add_argument("--show", action="store_true", help="Show the generated plot window")
    parser.add_argument("--save", default=None, help="Save the generated plot to a file path")
    parser.add_argument("--export", default=None, help="Write the result table to CSV")
    parser.add_argument("--grid", default=None, help="Subplot grid like 2x2 or 2*2")
    parser.add_argument("--xaxis", choices=["iter", "frame", "time"], default="iter", help="Quantity on x-axis")
    parser.add_argument("--control", default="control", help="Control file for time-axis conversion")


def _add_frame_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--frames", type=int, nargs="*", default=None, help="Selected frame indices")
    parser.add_argument("--every", type=int, default=1, help="Use every Nth selected frame")


def _maybe_load_force_field(args: argparse.Namespace) -> ForceFieldParametersData | None:
    raw = getattr(args, "ffield", None)
    if not raw:
        return None
    path = Path(raw)
    if not path.exists():
        return None
    adapter = resolve_engine(str(path), engine=getattr(args, "engine", None))
    return adapter.load(ForceFieldParametersData, vars(args))


def _build_connection_list_request(args: argparse.Namespace) -> ConnectionListRequest:
    return ConnectionListRequest(
        frames=_parse_frames(args.frames),
        every=args.every,
        min_bo=args.min_bo,
        undirected=args.undirected,
        aggregate=args.aggregate,
        include_self=args.include_self,
    )


def _build_connection_table_request(args: argparse.Namespace) -> ConnectionTableRequest:
    return ConnectionTableRequest(
        frame=args.frame,
        min_bo=args.min_bo,
        undirected=args.undirected,
        fill_value=args.fill_value,
    )


def _build_connection_stats_request(args: argparse.Namespace) -> ConnectionStatsRequest:
    return ConnectionStatsRequest(
        frames=_parse_frames(args.frames),
        every=args.every,
        min_bo=args.min_bo,
        undirected=args.undirected,
        how=args.how,
    )


def _build_bond_timeseries_request(args: argparse.Namespace) -> BondTimeseriesRequest:
    return BondTimeseriesRequest(
        frames=_parse_frames(args.frames),
        every=args.every,
        undirected=args.undirected,
        bo_threshold=args.bo_threshold,
        as_wide=args.as_wide,
    )


def _build_bond_events_request(args: argparse.Namespace) -> BondEventsRequest:
    return BondEventsRequest(
        frames=_parse_frames(args.frames),
        every=args.every,
        src=args.src,
        dst=args.dst,
        threshold=args.threshold,
        hysteresis=args.hysteresis,
        smooth=args.smooth,
        window=args.window,
        ema_alpha=args.ema_alpha,
        min_run=args.min_run,
        xaxis=args.xaxis,
        undirected=args.undirected,
    )


def _build_coordination_request(args: argparse.Namespace) -> CoordinationStatusRequest:
    valences = _parse_kv_map(args.valences, value_cast=float) if args.valences else None
    force_field = None if valences else _maybe_load_force_field(args)
    return CoordinationStatusRequest(
        valences=valences,
        force_field=force_field,
        valence_key=args.valence_key,
        threshold=args.threshold,
        frames=_parse_frames(args.frames),
        every=args.every,
        require_all_valences=not args.allow_missing_valences,
    )


def _build_hybridization_request(args: argparse.Namespace) -> HybridizationStatusRequest:
    global_map = _parse_kv_map(args.hybridizations, value_cast=float) if args.hybridizations else None
    element_map = _parse_element_hybridizations(args.element_hybridizations) if args.element_hybridizations else None
    return HybridizationStatusRequest(
        hybridizations=global_map,
        element_hybridizations=element_map,
        target_elements=tuple(args.target_elements) if args.target_elements else None,
        target_atom_ids=tuple(args.target_atom_ids) if args.target_atom_ids else None,
        threshold=args.threshold,
        frames=_parse_frames(args.frames),
        every=args.every,
        require_defined_hybridization=not args.allow_undefined_hybridization,
    )


def _build_coordination_relabel_request(
    args: argparse.Namespace,
    coordination_table: pd.DataFrame,
) -> TrajectoryRelabelByCoordinationRequest:
    return TrajectoryRelabelByCoordinationRequest(
        coordination_table=coordination_table,
        labels=_parse_status_labels(args.labels),
        mode=args.mode,
        keep_coord_original=args.keep_coord_original,
        frames=_parse_frames(args.frames),
        every=args.every,
    )


REQUEST_BUILDERS: dict[str, Callable[[argparse.Namespace], object]] = {
    "connection_list": _build_connection_list_request,
    "connection_table": _build_connection_table_request,
    "connection_stats": _build_connection_stats_request,
    "bond_timeseries": _build_bond_timeseries_request,
    "bond_events": _build_bond_events_request,
    "coordination": _build_coordination_request,
    "hybridization": _build_hybridization_request,
}


def build_parser(parser: argparse.ArgumentParser, *, command: str) -> argparse.ArgumentParser:
    canonical = resolve_command_name(command, task_names=CONNECTIVITY_COMMANDS)
    parser.set_defaults(command=canonical)
    parser.formatter_class = argparse.RawTextHelpFormatter

    _add_runtime_arguments(parser)
    _add_presentation_arguments(parser)

    if canonical == "connection_list":
        parser.description = (
            "List connections from bond-order frames.\n\n"
            "Examples:\n"
            "  reaxkit connection_list --fort7 fort.7 --frames 0 1 2 --export connections.csv\n"
            "  reaxkit connection_list --fort7 fort.7 --min-bo 0.3 --undirected\n"
            "  reaxkit connection_list --fort7 fort.7 --include-self --aggregate mean"
        )
        _add_frame_arguments(parser)
        parser.add_argument("--min-bo", type=float, default=0.0, help="Minimum bond order")
        parser.add_argument("--undirected", action=argparse.BooleanOptionalAction, default=True, help="Collapse i-j and j-i")
        parser.add_argument("--aggregate", choices=["max", "mean"], default="max", help="How to aggregate undirected pairs")
        parser.add_argument("--include-self", action="store_true", help="Include self connections")
    elif canonical == "connection_table":
        parser.description = (
            "Build a single-frame connectivity matrix.\n\n"
            "Examples:\n"
            "  reaxkit connection_table --fort7 fort.7 --frame 0 --export connection_table.csv\n"
            "  reaxkit connection_table --fort7 fort.7 --frame 10 --min-bo 0.3\n"
            "  reaxkit connection_table --fort7 fort.7 --frame 5 --fill-value -1"
        )
        parser.add_argument("--frame", type=int, default=0, help="Frame index to extract")
        parser.add_argument("--min-bo", type=float, default=0.0, help="Minimum bond order")
        parser.add_argument("--undirected", action=argparse.BooleanOptionalAction, default=True, help="Collapse i-j and j-i")
        parser.add_argument("--fill-value", type=float, default=0.0, help="Fill value for missing entries")
    elif canonical == "connection_stats":
        parser.description = (
            "Aggregate connectivity statistics across frames.\n\n"
            "Examples:\n"
            "  reaxkit connection_stats --fort7 fort.7 --how mean --export connection_stats.csv\n"
            "  reaxkit connection_stats --fort7 fort.7 --frames 0 10 20 --how count\n"
            "  reaxkit connection_stats --fort7 fort.7 --min-bo 0.3 --how max"
        )
        _add_frame_arguments(parser)
        parser.add_argument("--min-bo", type=float, default=0.0, help="Minimum bond order")
        parser.add_argument("--undirected", action=argparse.BooleanOptionalAction, default=True, help="Collapse i-j and j-i")
        parser.add_argument("--how", choices=["mean", "max", "count"], default="mean", help="Statistic to compute")
    elif canonical == "bond_timeseries":
        parser.description = (
            "Build bond-order time series for all observed bonds.\n\n"
            "Examples:\n"
            "  reaxkit bond_timeseries --fort7 fort.7 --plot single\n"
            "  reaxkit bond_timeseries --fort7 fort.7 --bo-threshold 0.3 --export bond_ts.csv\n"
            "  reaxkit bond_timeseries --fort7 fort.7 --as-wide --export bond_ts_wide.csv"
        )
        _add_frame_arguments(parser)
        parser.add_argument("--undirected", action=argparse.BooleanOptionalAction, default=True, help="Collapse i-j and j-i")
        parser.add_argument("--bo-threshold", type=float, default=0.0, help="Zero-out values below this threshold")
        parser.add_argument("--as-wide", action="store_true", help="Return wide one-column-per-bond output")
    elif canonical == "bond_events":
        parser.description = (
            "Detect bond formation and breakage events.\n\n"
            "Examples:\n"
            "  reaxkit bond_events --fort7 fort.7 --src 1 --dst 2 --export bond_events.csv\n"
            "  reaxkit bond_events --fort7 fort.7 --threshold 0.35 --hysteresis 0.05 --plot single\n"
            "  reaxkit bond_events --fort7 fort.7 --smooth ema --window 9 --xaxis frame"
        )
        _add_frame_arguments(parser)
        parser.add_argument("--src", type=int, default=None, help="Source atom id filter")
        parser.add_argument("--dst", type=int, default=None, help="Destination atom id filter")
        parser.add_argument("--threshold", type=float, default=0.35, help="Schmitt threshold")
        parser.add_argument("--hysteresis", type=float, default=0.05, help="Schmitt hysteresis width")
        parser.add_argument("--smooth", choices=["ma", "ema"], default="ma", help="Smoothing method")
        parser.add_argument("--window", type=int, default=7, help="Smoothing window")
        parser.add_argument("--ema-alpha", type=float, default=None, help="Optional EMA alpha")
        parser.add_argument("--min-run", type=int, default=3, help="Minimum run length after flicker cleanup")
        parser.add_argument("--undirected", action=argparse.BooleanOptionalAction, default=True, help="Collapse i-j and j-i")
    elif canonical == "coordination":
        parser.description = (
            "Classify atoms as under-, coordinated, or over-coordinated.\n\n"
            "Examples:\n"
            "  reaxkit coordination --fort7 fort.7 --xmolout xmolout --valences Mg=2,O=2 --export coordination.csv\n"
            "  reaxkit coordination --fort7 fort.7 --xmolout xmolout --ffield ffield --frames 0 10 20\n"
            "  reaxkit coordination --fort7 fort.7 --xmolout xmolout --threshold 0.2 --plot single"
        )
        _add_frame_arguments(parser)
        parser.add_argument("--valences", default=None, help="Explicit valence map like Mg=2,O=2")
        parser.add_argument("--ffield", default=None, help="Optional force-field file to infer valences")
        parser.add_argument("--valence-key", default="valency", help="Force-field atom parameter column for valence")
        parser.add_argument("--threshold", type=float, default=0.9, help="Tolerance around target valence")
        parser.add_argument("--allow-missing-valences", action="store_true", help="Do not fail on missing valences")
    elif canonical == "coordination_relabel":
        parser.description = (
            "Relabel trajectory atom labels from coordination status and write engine-specific output.\n\n"
            "Examples:\n"
            "  reaxkit coordination_relabel --fort7 fort.7 --xmolout xmolout --output xmolout_relabeled\n"
            "  reaxkit coordination_relabel --valences Mg=2,O=2 --mode by_type --keep-coord-original --output relabeled.xyz\n"
            "  reaxkit coordination_relabel --ffield ffield --frames 0 10 20 --labels=-1=U,0=C,1=O --export coordination.csv --output relabeled.xmolout"
        )
        _add_frame_arguments(parser)
        parser.add_argument("--valences", default=None, help="Explicit valence map like Mg=2,O=2")
        parser.add_argument("--ffield", default=None, help="Optional force-field file to infer valences")
        parser.add_argument("--valence-key", default="valency", help="Force-field atom parameter column for valence")
        parser.add_argument("--threshold", type=float, default=0.9, help="Tolerance around target valence")
        parser.add_argument("--allow-missing-valences", action="store_true", help="Do not fail on missing valences")
        parser.add_argument("--output", required=True, help="Output trajectory path")
        parser.add_argument("--mode", choices=["global", "by_type"], default="global", help="Relabeling mode")
        parser.add_argument("--labels", default=None, help="Status tag map like -1=U,0=C,1=O")
        parser.add_argument("--keep-coord-original", action="store_true", help="Keep original label when status is coordinated in by_type mode")
        parser.add_argument("--precision", type=int, default=6, help="Writer precision when supported by the engine")
        parser.add_argument("--simulation", default=None, help="Optional trajectory writer simulation label")
    elif canonical == "hybridization":
        parser.description = (
            "Classify atoms against target hybridization bond-order sums.\n\n"
            "Examples:\n"
            "  reaxkit hybridization --fort7 fort.7 --xmolout xmolout --hybridizations sp=1,sp2=2,sp3=3 --export hyb.csv\n"
            "  reaxkit hybridization --fort7 fort.7 --xmolout xmolout --element-hybridizations \"C:sp=1,sp2=2,sp3=3;N:sp2=2,sp3=3\"\n"
            "  reaxkit hybridization --fort7 fort.7 --xmolout xmolout --target-elements C O --threshold 0.2"
        )
        _add_frame_arguments(parser)
        parser.add_argument("--hybridizations", default=None, help="Global map like sp=1,sp2=2,sp3=3")
        parser.add_argument("--element-hybridizations", default=None, help="Per-element map like C:sp=1,sp2=2;N:sp2=2,sp3=3")
        parser.add_argument("--target-elements", nargs="*", default=None, help="Restrict to selected elements")
        parser.add_argument("--target-atom-ids", type=int, nargs="*", default=None, help="Restrict to selected atom ids")
        parser.add_argument("--threshold", type=float, default=0.3, help="Tolerance around target BO sum")
        parser.add_argument("--allow-undefined-hybridization", action="store_true", help="Do not fail on missing mappings")
    else:
        raise KeyError(f"Unsupported connectivity command '{canonical}'.")

    return parser


def _prepare_result_table(command: str, result, args: argparse.Namespace) -> None:
    if command in {"connection_table", "bond_timeseries"} and isinstance(result.table, pd.DataFrame):
        if not isinstance(result.table.index, pd.RangeIndex):
            result.table = result.table.reset_index()


def _plot_payload(command: str, result, args: argparse.Namespace) -> dict[str, object] | None:
    table = result.table
    if not isinstance(table, pd.DataFrame) or table.empty:
        return None

    if command == "connection_list":
        count_col = "frame_idx" if "frame_idx" in table.columns else ("frame_index" if "frame_index" in table.columns else None)
        if count_col is None:
            return None
        counts = table.groupby(count_col, as_index=False).size().rename(columns={"size": "edges"})
        if counts.empty:
            return None

        if args.xaxis == "iter" and "iter" in table.columns:
            iter_map = table.drop_duplicates(count_col)[[count_col, "iter"]]
            counts = counts.merge(iter_map, on=count_col, how="left")
            xvals = counts["iter"].tolist()
            xlabel = "Iteration"
        elif args.xaxis == "time" and "iter" in table.columns:
            iter_map = table.drop_duplicates(count_col)[[count_col, "iter"]]
            counts = counts.merge(iter_map, on=count_col, how="left")
            converted, xlabel = convert_xaxis(counts["iter"].to_numpy(dtype=int), "time", control_file=args.control)
            xvals = np.asarray(converted).tolist()
        else:
            xvals = counts[count_col].tolist()
            xlabel = "Frame Index"

        return {
            "plot_type": "single_plot",
            "x": xvals,
            "y": counts["edges"].tolist(),
            "xlabel": xlabel,
            "ylabel": "Edge Count",
            "title": "Connection Count Per Frame",
        }

    if command == "bond_timeseries":
        if args.as_wide:
            base_x_col = "iter" if "iter" in table.columns else "frame_idx"
            series = []
            for col in table.columns:
                if col in {"frame_idx", "iter"}:
                    continue
                xvals = table[base_x_col].to_numpy(dtype=int if base_x_col == "iter" else float)
                xlabel = "Iteration" if base_x_col == "iter" else "Frame Index"
                if args.xaxis == "time" and "iter" in table.columns:
                    converted, xlabel = convert_xaxis(table["iter"].to_numpy(dtype=int), "time", control_file=args.control)
                    xvals = np.asarray(converted)
                elif args.xaxis == "frame" and "frame_idx" in table.columns:
                    xvals = table["frame_idx"].to_numpy(dtype=float)
                    xlabel = "Frame Index"
                elif args.xaxis == "iter" and "iter" in table.columns:
                    xvals = table["iter"].to_numpy(dtype=int)
                    xlabel = "Iteration"
                series.append({"x": xvals.tolist(), "y": pd.to_numeric(table[col], errors="coerce").tolist(), "label": str(col)})
        else:
            series = []
            for (src, dst), group in table.groupby(["src", "dst"], sort=True):
                group = group.sort_values("frame_idx")
                if args.xaxis == "time" and "iter" in group.columns:
                    converted, xlabel = convert_xaxis(group["iter"].to_numpy(dtype=int), "time", control_file=args.control)
                    xvals = np.asarray(converted).tolist()
                elif args.xaxis == "frame":
                    xvals = group["frame_idx"].tolist()
                    xlabel = "Frame Index"
                else:
                    xvals = group["iter"].tolist() if "iter" in group.columns else group["frame_idx"].tolist()
                    xlabel = "Iteration" if "iter" in group.columns else "Frame Index"
                series.append(
                    {
                        "x": xvals,
                        "y": pd.to_numeric(group["bo"], errors="coerce").tolist(),
                        "label": f"{src}-{dst}",
                    }
                )
        if not series:
            return None
        if getattr(args, "plot", None) == "subplot":
            return {
                "plot_type": "multi_subplots",
                "subplots": [[s] for s in series],
                "xlabel": xlabel,
                "ylabel": "Bond Order",
                "title": "Bond Timeseries",
                "legend": False,
                "grid": getattr(args, "grid", None),
            }
        return {
            "plot_type": "single_plot",
            "series": series,
            "xlabel": xlabel,
            "ylabel": "Bond Order",
            "title": "Bond Timeseries",
            "legend": True,
        }

    if command == "bond_events":
        series = []
        for event_name, group in table.groupby("event", sort=True):
            if args.xaxis == "time" and "iter" in group.columns:
                converted, xlabel = convert_xaxis(group["iter"].to_numpy(dtype=int), "time", control_file=args.control)
                xvals = np.asarray(converted).tolist()
            else:
                x_col = "x_axis" if "x_axis" in table.columns else ("iter" if args.xaxis == "iter" and "iter" in table.columns else "frame_idx")
                xvals = group[x_col].tolist()
                xlabel = "Iteration" if x_col == "iter" else "Frame Index"
            series.append(
                {
                    "x": xvals,
                    "y": pd.to_numeric(group["bo_at_event"], errors="coerce").tolist(),
                    "label": str(event_name),
                }
            )
        if not series:
            return None
        return {
            "plot_type": "single_plot",
            "series": series,
            "xlabel": xlabel,
            "ylabel": "Bond Order At Event",
            "title": "Bond Events",
            "legend": True,
            "plot_type_style": "scatter",
        }

    if command in {"coordination", "hybridization"} and "frame_index" in table.columns:
        label_col = "status_label" if "status_label" in table.columns else None
        if label_col is None:
            return None
        counts = table.groupby(["frame_index", "iter", label_col], as_index=False).size().rename(columns={"size": "count"})
        x_col = "iter" if args.xaxis == "iter" and "iter" in counts.columns else "frame_index"
        series = []
        for label, group in counts.groupby(label_col, sort=True):
            group = group.sort_values("frame_index")
            series.append({"x": group[x_col].tolist(), "y": group["count"].tolist(), "label": str(label)})
        if not series:
            return None
        return {
            "plot_type": "single_plot",
            "series": series,
            "xlabel": "Iteration" if args.xaxis == "iter" else "Frame Index",
            "ylabel": "Atom Count",
            "title": command.replace("_", " ").title(),
            "legend": True,
        }

    return None


def run_main(command: str, args: argparse.Namespace) -> int:
    canonical = resolve_command_name(command, task_names=CONNECTIVITY_COMMANDS)
    if canonical == "coordination_relabel":
        normalized = normalize_storage_args(vars(args))
        adapter = resolve_engine(
            normalized.get("input") or normalized.get("run_dir") or normalized.get("xmolout") or ".",
            engine=getattr(args, "engine", None),
        )
        composite = adapter.load(ConnectivityTrajectoryData, normalized)

        coordination_task_cls = TASK_REGISTRY["coordination"]
        coordination_request = _build_coordination_request(args)
        coordination_result = coordination_task_cls().run(composite.connectivity, coordination_request)

        if args.export:
            export_result_csv(coordination_result, args.export)

        relabel_task_cls = TASK_REGISTRY["trajectory_relabel_by_coordination"]
        relabel_request = _build_coordination_relabel_request(args, coordination_result.table)
        relabel_result = relabel_task_cls().run(composite, relabel_request)

        out_path = adapter.write(relabel_result.trajectory, args.output, vars(args))
        print(f"Wrote relabeled trajectory to {out_path}")
        return 0

    task_cls = TASK_REGISTRY[canonical]
    request = REQUEST_BUILDERS[canonical](args)

    executor = AnalysisExecutor()
    result = executor.run(task_cls(), request, vars(args))
    _prepare_result_table(canonical, result, args)
    present_result(canonical, result, args, plot_payload_builder=_plot_payload)
    return 0
