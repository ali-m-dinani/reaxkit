"""Direct command workflow for connectivity analyses.

This module implements CLI workflow orchestration for its command family, including argument parsing, request construction, execution dispatch, and result presentation handoff.

**Usage context**

- Command routing: Resolve CLI aliases and normalized command names.
- Task execution: Build request objects and invoke registered tasks.
- Output handling: Forward results to table, plot, export, or report flows.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import reaxkit.engine  # noqa: F401

from reaxkit.analysis import connectivity as _connectivity_tasks  # noqa: F401
from reaxkit.analysis import trajectory as _trajectory_tasks  # noqa: F401
from reaxkit.cli.path import resolve_output_path
from reaxkit.analysis.connectivity.connectivity import (
    BondEventsRequest,
    ConnectionListRequest,
    ConnectionStatsRequest,
    ConnectionTableRequest,
)
from reaxkit.analysis.connectivity.coordination import CoordinationStatusRequest
from reaxkit.analysis.connectivity.hybridization import HybridizationStatusRequest
from reaxkit.analysis.trajectory.relabel import TrajectoryRelabelByCoordinationRequest
from reaxkit.core.runtime.analysis_executor import AnalysisExecutor
from reaxkit.core.platform.engine_resolver import resolve_engine
from reaxkit.core.registry.analysis_task_registry import TASK_REGISTRY
from reaxkit.core.resolve.command_alias_resolver import resolve_command_name
from reaxkit.core.utils.frame_utils import parse_frame_indices
from reaxkit.core.storage.storage_layout import add_storage_cli_arguments, normalize_storage_args
from reaxkit.domain.data_models import ConnectivityTrajectoryData
from reaxkit.presentation.convert import convert_xaxis
from reaxkit.presentation.dispatcher import export_result_csv, present_result

ALL_COMMANDS = (
    "get_connection_list",
    "get_connection_table",
    "get_connection_stats",
    "get_bond_events",
    "get_coordination",
    "relabel_traj_using_coordination",
    "get_hybridization",
)
ALL_LEGACY_COMMANDS = (
    "connection_list",
    "connection_table",
    "connection_stats",
    "bond_events",
    "coordination",
    "coordination_relabel",
    "hybridization",
    "get-connection-list",
    "get-connection-table",
    "get-connection-stats",
    "get-bond-events",
    "get-coordination",
    "get-hybridization",
    "relabel-traj-using-coordination",
)


def _parse_frames(values) -> list[int] | None:
    """Parse frames."""
    return parse_frame_indices(values)


def _parse_kv_map(spec: str | None, *, value_cast=float) -> dict[str, float]:
    """Parse kv map."""
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
    """Parse element hybridizations."""
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
    """Parse status labels."""
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
    """Add runtime arguments."""
    parser.add_argument("--engine", choices=["reaxff", "ams", "lammps"], default=None, help="Engine override. Example: --engine reaxff, which forces ReaxFF file parsing rules.")
    parser.add_argument("--input", default=".", help="Input file or directory for engine resolution. Example: --input runs/job1, which points loader context to that run.")
    parser.add_argument("--run-dir", "--dir", dest="run_dir", default=".", help="Run directory fallback for engine detection. Example: --run-dir runs/job1, which acts as backup search location.")
    parser.add_argument("--fort7", default="fort.7", help="Path to fort.7. Example: --fort7 runs/job1/fort.7, which uses that bond-order trajectory file.")
    parser.add_argument("--xmolout", default="xmolout", help="Path to xmolout. Example: --xmolout runs/job1/xmolout, which supplies atom/trajectory metadata.")
    parser.add_argument("--summary", default=None, help="Optional summary.txt path. Example: --summary runs/job1/summary.txt, which provides auxiliary timeline data when needed.")
    parser.add_argument("--log", choices=["verbose", "quiet"], default=None, help="Logging level. Example: --log verbose, which prints more processing details.")
    add_storage_cli_arguments(parser)


def _add_presentation_arguments(parser: argparse.ArgumentParser) -> None:
    """Add presentation arguments."""
    parser.add_argument("--plot", choices=["single", "subplot"], default=None, help="Render a plot. Example: --plot single, which generates a single-panel figure.")
    parser.add_argument("--show", action="store_true", help="Show the generated plot window. Example: --show, which opens the figure interactively.")
    parser.add_argument("--save", default=None, help="Save the generated plot to a file path. Example: --save figures/conn.png, which writes the plot image to that path.")
    parser.add_argument("--export", default=None, help="Write the result table to CSV. Example: --export connectivity.csv, which saves tabular results for post-processing.")
    parser.add_argument("--grid", default=None, help="Subplot grid like 2x2 or 2*2. Example: --grid 2x2, which arranges subplot panels in a 2-by-2 layout.")
    parser.add_argument("--xaxis", choices=["iter", "frame", "time"], default="iter", help="Quantity on x-axis. Example: --xaxis time, which converts iteration axis to physical time when possible.")
    parser.add_argument("--control", default="control", help="Control file for time-axis conversion. Example: --control control, which supplies timestep settings for time conversion.")


def _add_frame_arguments(parser: argparse.ArgumentParser) -> None:
    """Add frame arguments."""
    parser.add_argument(
        "--frames",
        nargs="*",
        default=None,
        help='Frames selection syntax. Example: --frames 0:20:2, which selects frames 0,2,4,...,20.',
    )
    parser.add_argument("--every", type=int, default=1, help="Use every Nth selected frame. Example: --every 5, which subsamples selected frames by a factor of 5.")


def _build_connection_list_request(args: argparse.Namespace) -> ConnectionListRequest:
    """Build connection list request."""
    return ConnectionListRequest(
        frames=_parse_frames(args.frames),
        every=args.every,
        min_bo=args.min_bo,
        undirected=args.undirected,
        include_self=args.include_self,
    )


def _build_connection_table_request(args: argparse.Namespace) -> ConnectionTableRequest:
    """Build connection table request."""
    selected = _selected_connection_table_frames(args)
    return ConnectionTableRequest(
        frame=int(selected[0]),
        min_bo=args.min_bo,
        undirected=args.undirected,
        fill_value=args.fill_value,
    )


def _build_connection_stats_request(args: argparse.Namespace) -> ConnectionStatsRequest:
    """Build connection stats request."""
    return ConnectionStatsRequest(
        frames=_parse_frames(args.frames),
        every=args.every,
        min_bo=args.min_bo,
        undirected=args.undirected,
        how=args.how,
    )


def _build_bond_events_request(args: argparse.Namespace) -> BondEventsRequest:
    """Build bond events request."""
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
        undirected=args.undirected,
    )


def _build_coordination_request(args: argparse.Namespace) -> CoordinationStatusRequest:
    """Build coordination request."""
    valences = _parse_kv_map(args.valences, value_cast=float) if args.valences else None
    return CoordinationStatusRequest(
        valences=valences,
        threshold=args.threshold,
        frames=_parse_frames(args.frames),
        every=args.every,
        require_all_valences=not args.allow_missing_valences,
    )


def _build_hybridization_request(args: argparse.Namespace) -> HybridizationStatusRequest:
    """Build hybridization request."""
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
) -> TrajectoryRelabelByCoordinationRequest:
    """Build coordination relabel request."""
    valences = _parse_kv_map(args.valences, value_cast=float) if args.valences else None
    return TrajectoryRelabelByCoordinationRequest(
        labels=_parse_status_labels(args.labels),
        mode=args.mode,
        keep_coord_original=args.keep_coord_original,
        frames=_parse_frames(args.frames),
        every=args.every,
        valences=valences,
        threshold=args.threshold,
        require_all_valences=not args.allow_missing_valences,
    )


REQUEST_BUILDERS: dict[str, Callable[[argparse.Namespace], object]] = {
    "get_connection_list": _build_connection_list_request,
    "get_connection_table": _build_connection_table_request,
    "get_connection_stats": _build_connection_stats_request,
    "get_bond_events": _build_bond_events_request,
    "get_coordination": _build_coordination_request,
    "get_hybridization": _build_hybridization_request,
}


def _selected_connection_table_frames(args: argparse.Namespace) -> list[int]:
    """Selected connection table frames."""
    raw_frames = _parse_frames(getattr(args, "frames", None))
    if raw_frames is None or len(raw_frames) == 0:
        return [0]
    every = max(1, int(getattr(args, "every", 1)))
    return [int(v) for v in raw_frames][::every]


def build_parser(parser: argparse.ArgumentParser, *, command: str) -> argparse.ArgumentParser:
    """Build parser.

    Execute the workflow function for this command path and return the
    computed result for downstream CLI handling.

    Parameters
    -----
    parser : Any
        Function argument.
    command : Any
        Function argument.

    Returns
    -----
    argparse.ArgumentParser
        Function return value.

    Examples
    -----
    >>> # See workflow CLI usage for concrete examples.
    """
    canonical = resolve_command_name(command, task_names=ALL_COMMANDS)
    parser.set_defaults(command=canonical)
    parser.set_defaults(progress=True)
    parser.formatter_class = argparse.RawTextHelpFormatter

    _add_runtime_arguments(parser)
    _add_presentation_arguments(parser)

    if canonical == "get_connection_list":
        parser.description = (
            "List atom-to-atom connections extracted from bond-order frames.\n"
            "Use this command to inspect which atom pairs are connected at selected frames,\n"
            "with optional bond-order thresholding and direction collapsing.\n\n"
            "Examples:\n"
            "  1. Export connections for selected frames:\n"
            "   reaxkit get_connection_list --fort7 fort.7 --frames 0 1 2 --export connections.csv\n\n"
            "  2. Keep only edges above a BO threshold and collapse i-j/j-i duplicates:\n"
            "   reaxkit get_connection_list --fort7 fort.7 --min-bo 0.3 --undirected\n\n"
            "  3. Include self-connections in output:\n"
            "   reaxkit get_connection_list --fort7 fort.7 --include-self"
        )
        _add_frame_arguments(parser)
        parser.add_argument("--min-bo", type=float, default=0.0, help="Minimum bond order. Example: --min-bo 0.3, which filters out weaker bonds below 0.3.")
        parser.add_argument("--undirected", action=argparse.BooleanOptionalAction, default=True, help="Collapse i-j and j-i. Example: --no-undirected, which keeps directed pair ordering.")
        parser.add_argument("--include-self", action="store_true", help="Include self connections. Example: --include-self, which keeps i->i entries when present.")
    elif canonical == "get_connection_table":
        parser.description = (
            "Build frame-wise connectivity matrices from bond-order data.\n"
            "For one frame, the command returns a single matrix. For multiple frames, export mode\n"
            "writes one CSV per frame.\n\n"
            "Examples:\n"
            "  1. Export a single-frame connectivity table:\n"
            "   reaxkit get_connection_table --fort7 fort.7 --frames 0 --export connection_table.csv\n\n"
            "  2. Export connectivity tables for multiple frames:\n"
            "   reaxkit get_connection_table --fort7 fort.7 --frames 0 10 20 --export connection_table.csv\n\n"
            "  3. Apply bond-order threshold and custom fill value:\n"
            "   reaxkit get_connection_table --fort7 fort.7 --frames 5 --min-bo 0.3 --fill-value -1"
        )
        _add_frame_arguments(parser)
        parser.add_argument("--min-bo", type=float, default=0.0, help="Minimum bond order. Example: --min-bo 0.3, which keeps only stronger connections.")
        parser.add_argument("--undirected", action=argparse.BooleanOptionalAction, default=True, help="Collapse i-j and j-i. Example: --no-undirected, which keeps direction-specific matrix entries.")
        parser.add_argument("--fill-value", type=float, default=0.0, help="Fill value for missing entries. Example: --fill-value -1, which marks absent entries explicitly as -1.")
    elif canonical == "get_connection_stats":
        parser.description = (
            "Aggregate connectivity statistics across selected frames.\n"
            "Use this command to summarize connectivity behavior with mean/max/count aggregations.\n\n"
            "Examples:\n"
            "  1. Export mean connectivity statistics:\n"
            "   reaxkit get_connection_stats --fort7 fort.7 --how mean --export connection_stats.csv\n\n"
            "  2. Compute edge counts on specific frames:\n"
            "   reaxkit get_connection_stats --fort7 fort.7 --frames 0 10 20 --how count\n\n"
            "  3. Use thresholded max aggregation:\n"
            "   reaxkit get_connection_stats --fort7 fort.7 --min-bo 0.3 --how max"
        )
        _add_frame_arguments(parser)
        parser.add_argument("--min-bo", type=float, default=0.0, help="Minimum bond order. Example: --min-bo 0.3, which removes weak edges before statistics.")
        parser.add_argument("--undirected", action=argparse.BooleanOptionalAction, default=True, help="Collapse i-j and j-i. Example: --no-undirected, which treats reverse directions separately.")
        parser.add_argument("--how", choices=["mean", "max", "count"], default="mean", help="Statistic to compute. Example: --how count, which reports occurrence counts instead of mean/max BO.")
    elif canonical == "get_bond_events":
        parser.description = (
            "Detect bond formation and breakage events over time.\n"
            "The command applies threshold/hysteresis logic and optional smoothing to bond-order\n"
            "signals, then reports event points.\n\n"
            "Examples:\n"
            "  1. Detect events for one atom pair and export:\n"
            "   reaxkit get_bond_events --fort7 fort.7 --src 1 --dst 2 --export bond_events.csv\n\n"
            "  2. Tune threshold/hysteresis and plot events:\n"
            "   reaxkit get_bond_events --fort7 fort.7 --threshold 0.35 --hysteresis 0.05 --plot single\n\n"
            "  3. Use EMA smoothing and frame axis for plotting:\n"
            "   reaxkit get_bond_events --fort7 fort.7 --smooth ema --window 9 --xaxis frame"
        )
        _add_frame_arguments(parser)
        parser.add_argument("--src", type=int, default=None, help="Source atom-id filter. Example: --src 1, which keeps events where source atom id is 1.")
        parser.add_argument("--dst", type=int, default=None, help="Destination atom-id filter. Example: --dst 2, which keeps events where destination atom id is 2.")
        parser.add_argument("--threshold", type=float, default=0.35, help="Schmitt threshold. Example: --threshold 0.4, which raises event trigger level.")
        parser.add_argument("--hysteresis", type=float, default=0.05, help="Schmitt hysteresis width. Example: --hysteresis 0.05, which adds separation between open/close transitions.")
        parser.add_argument("--smooth", choices=["ma", "ema"], default="ma", help="Smoothing method. Example: --smooth ema, which applies exponential moving average.")
        parser.add_argument("--window", type=int, default=7, help="Smoothing window. Example: --window 9, which increases smoothing span.")
        parser.add_argument("--ema-alpha", type=float, default=None, help="Optional EMA alpha. Example: --ema-alpha 0.3, which controls EMA responsiveness.")
        parser.add_argument("--min-run", type=int, default=3, help="Minimum run length after flicker cleanup. Example: --min-run 5, which suppresses short-lived toggles.")
        parser.add_argument("--undirected", action=argparse.BooleanOptionalAction, default=True, help="Collapse i-j and j-i. Example: --no-undirected, which keeps direction-specific events.")
    elif canonical == "get_coordination":
        parser.description = (
            "Classify atoms as under-, coordinated-, or over-coordinated.\n"
            "Classification compares bond-order totals against target valences from explicit maps\n"
            "or inferred values.\n"
            "For example, if an atom's valence is 3 and the threshold is 0.5, then:\n"
            " - sum_BOs < 2.5 -> under-coordinated\n"
            " - 2.5 <= sum_BOs <= 3.5 -> coordinated\n"
            " - sum_BOs > 3.5 -> over-coordinated\n\n"
            "Examples:\n"
            "  1. Classify using explicit valence map and export:\n"
            "   reaxkit get_coordination --fort7 fort.7 --xmolout xmolout --valences Mg=2,O=2 --export coordination.csv\n\n"
            "  2. Classify using valences inferred from force field:\n"
            "   reaxkit get_coordination --fort7 fort.7 --xmolout xmolout --ffield ffield --frames 0 10 20\n\n"
            "  3. Adjust tolerance and plot results:\n"
            "   reaxkit get_coordination --fort7 fort.7 --xmolout xmolout --threshold 0.2 --plot single"
        )
        _add_frame_arguments(parser)
        parser.add_argument("--valences", default=None, help="Explicit valence map like Mg=2,O=2. Example: --valences Mg=2,O=2, which sets target valences directly.")
        parser.add_argument("--ffield", default=None, help="Optional force-field file to infer valences. Example: --ffield ffield, which derives valence targets from that force field.")
        parser.add_argument("--threshold", type=float, default=0.9, help="Tolerance around target valence. Example: --threshold 0.2, which tightens classification around target BO sums.")
        parser.add_argument("--allow-missing-valences", action="store_true", help="Do not fail on missing valences. Example: --allow-missing-valences, which skips strict failure when some mappings are absent.")
    elif canonical == "relabel_traj_using_coordination":
        parser.description = (
            "Relabel trajectory atom labels based on coordination status.\n"
            "This command computes coordination classes and writes a relabeled trajectory using\n"
            "engine-specific output formatting.\n\n"
            "Examples:\n"
            "  1. Relabel using defaults and write output trajectory:\n"
            "   reaxkit relabel_traj_using_coordination --fort7 fort.7 --xmolout xmolout --output xmolout_relabeled\n\n"
            "  2. Use explicit valences and type-aware relabeling:\n"
            "   reaxkit relabel_traj_using_coordination --valences Mg=2,O=2 --mode by_type --keep-coord-original --output relabeled.xyz\n\n"
            "  3. Use inferred valences, custom status labels, and export status table:\n"
            "   reaxkit relabel_traj_using_coordination --ffield ffield --frames 0 10 20 --labels=-1=U,0=C,1=O --export coordination.csv --output relabeled.xmolout"
        )
        _add_frame_arguments(parser)
        parser.add_argument("--valences", default=None, help="Explicit valence map like Mg=2,O=2. Example: --valences Mg=2,O=2, which sets coordination targets directly.")
        parser.add_argument("--ffield", default=None, help="Optional force-field file to infer valences. Example: --ffield ffield, which derives target valences automatically.")
        parser.add_argument("--threshold", type=float, default=0.9, help="Tolerance around target valence. Example: --threshold 0.2, which makes status classification stricter.")
        parser.add_argument("--allow-missing-valences", action="store_true", help="Do not fail on missing valences. Example: --allow-missing-valences, which permits partial valence definitions.")
        parser.add_argument("--output", required=True, help="Output trajectory path. Example: --output relabeled.xmolout, which writes relabeled trajectory to that file.")
        parser.add_argument("--mode", choices=["global", "by_type"], default="global", help="Relabeling mode. Example: --mode by_type, which applies status labels per atom type context.")
        parser.add_argument("--labels", default=None, help="Status tag map like -1=U,0=C,1=O. Example: --labels=-1=U,0=C,1=O, which customizes output status tokens.")
        parser.add_argument("--keep-coord-original", action="store_true", help="Keep original label when status is coordinated in by_type mode. Example: --keep-coord-original, which preserves original labels for coordinated atoms.")
        parser.add_argument("--precision", type=int, default=6, help="Writer precision when supported by the engine. Example: --precision 8, which writes numeric coordinates with higher decimal precision.")
        parser.add_argument("--simulation", default=None, help="Optional trajectory writer simulation label. Example: --simulation run_01, which tags output with that simulation name when supported.")
    elif canonical == "get_hybridization":
        parser.description = (
            "Classify atoms against target hybridization bond-order sums.\n"
            "You can define global hybridization targets or per-element target maps, then restrict\n"
            "classification to specific elements or atom ids.\n\n"
            "Examples:\n"
            "  1. Use global hybridization targets and export:\n"
            "   reaxkit get_hybridization --fort7 fort.7 --xmolout xmolout --hybridizations sp=1,sp2=2,sp3=3 --export hyb.csv\n\n"
            "  2. Use element-specific hybridization targets:\n"
            "   reaxkit get_hybridization --fort7 fort.7 --xmolout xmolout --element-hybridizations \"C:sp=1,sp2=2,sp3=3;N:sp2=2,sp3=3\"\n\n"
            "  3. Restrict to selected elements and tighten tolerance:\n"
            "   reaxkit get_hybridization --fort7 fort.7 --xmolout xmolout --target-elements C O --threshold 0.2"
        )
        _add_frame_arguments(parser)
        parser.add_argument("--hybridizations", default=None, help="Global map like sp=1,sp2=2,sp3=3. Example: --hybridizations sp=1,sp2=2,sp3=3, which applies one map to all elements.")
        parser.add_argument("--element-hybridizations", default=None, help="Per-element map like C:sp=1,sp2=2;N:sp2=2,sp3=3. Example: --element-hybridizations \"C:sp=1,sp2=2,sp3=3\", which customizes targets for specific elements.")
        parser.add_argument("--target-elements", nargs="*", default=None, help="Restrict to selected elements. Example: --target-elements C O, which evaluates only carbon and oxygen atoms.")
        parser.add_argument("--target-atom-ids", type=int, nargs="*", default=None, help="Restrict to selected atom ids. Example: --target-atom-ids 1 2 5, which evaluates only those atom indices.")
        parser.add_argument("--threshold", type=float, default=0.3, help="Tolerance around target BO sum. Example: --threshold 0.2, which tightens hybridization matching tolerance.")
        parser.add_argument("--allow-undefined-hybridization", action="store_true", help="Do not fail on missing mappings. Example: --allow-undefined-hybridization, which allows output even when some atoms have no configured target.")
    else:
        raise KeyError(f"Unsupported connectivity command '{canonical}'.")

    return parser


def _prepare_result_table(command: str, result, args: argparse.Namespace) -> None:
    """Prepare result table."""
    if command in {"get_connection_table"} and isinstance(result.table, pd.DataFrame):
        if not isinstance(result.table.index, pd.RangeIndex):
            result.table = result.table.reset_index()


def _plot_payload(command: str, result, args: argparse.Namespace) -> dict[str, object] | None:
    """Plot payload."""
    table = result.table
    if not isinstance(table, pd.DataFrame) or table.empty:
        return None

    if command == "get_connection_list":
        count_col = "frame_idx" if "frame_idx" in table.columns else ("frame_index" if "frame_index" in table.columns else None)
        if count_col is None:
            return None
        counts = table.groupby(count_col, as_index=False).size().rename(columns={"size": "edges"})
        if counts.empty:
            return None

        if args.xaxis == "iter" and "iteration" in table.columns:
            iter_map = table.drop_duplicates(count_col)[[count_col, "iteration"]]
            counts = counts.merge(iter_map, on=count_col, how="left")
            xvals = counts["iteration"].tolist()
            xlabel = "Iteration"
        elif args.xaxis == "time" and "iteration" in table.columns:
            iter_map = table.drop_duplicates(count_col)[[count_col, "iteration"]]
            counts = counts.merge(iter_map, on=count_col, how="left")
            converted, xlabel = convert_xaxis(counts["iteration"].to_numpy(dtype=int), "time", control_file=args.control)
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

    if command == "get_bond_events":
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

    if command in {"get_coordination", "get_hybridization"} and "frame_index" in table.columns:
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
    """Run main.

    Execute the workflow function for this command path and return the
    computed result for downstream CLI handling.

    Parameters
    -----
    command : Any
        Function argument.
    args : Any
        Function argument.

    Returns
    -----
    int
        Function return value.

    Examples
    -----
    >>> # See workflow CLI usage for concrete examples.
    """
    canonical = resolve_command_name(command, task_names=ALL_COMMANDS)
    if canonical == "relabel_traj_using_coordination":
        normalized = normalize_storage_args(vars(args))
        adapter = resolve_engine(
            normalized.get("input") or normalized.get("run_dir") or normalized.get("xmolout") or ".",
            engine=getattr(args, "engine", None),
        )
        composite = adapter.load(ConnectivityTrajectoryData, normalized)

        relabel_task_cls = TASK_REGISTRY["trajectory_relabel_by_coordination"]
        relabel_request = _build_coordination_relabel_request(args)
        relabel_result = relabel_task_cls().run(composite, relabel_request)
        if args.export:
            out = resolve_output_path(
                args.export,
                canonical,
                run_id=getattr(args, "run_id", None),
                project_root=getattr(args, "project_root", "."),
                analysis_id=getattr(args, "analysis_id", None),
            )
            export_result_csv(relabel_result, str(out))
            print(f"[Done] Exported data to {out}")

        out_path = adapter.write(relabel_result.trajectory, args.output, vars(args))
        print(f"Wrote relabeled trajectory to {out_path}")
        return 0

    if canonical == "get_connection_table":
        frames = _selected_connection_table_frames(args)
        if len(frames) > 1:
            if getattr(args, "plot", None) or getattr(args, "save", None) or getattr(args, "show", False):
                raise ValueError("Plotting for get_connection_table with multiple frames is not supported. Export CSVs instead.")
            if not getattr(args, "export", None):
                raise ValueError("Multiple frames require --export so one CSV per frame can be written.")

            task_cls = TASK_REGISTRY[canonical]
            executor = AnalysisExecutor()
            base_out = resolve_output_path(
                args.export,
                canonical,
                run_id=getattr(args, "run_id", None),
                project_root=getattr(args, "project_root", "."),
                analysis_id=getattr(args, "analysis_id", None),
            )
            stem = base_out.stem if base_out.suffix else base_out.name
            suffix = base_out.suffix if base_out.suffix else ".csv"
            written: list[Path] = []

            for frame in frames:
                req = ConnectionTableRequest(
                    frame=int(frame),
                    min_bo=float(args.min_bo),
                    undirected=bool(args.undirected),
                    fill_value=float(args.fill_value),
                )
                result = executor.run(task_cls(), req, vars(args))
                _prepare_result_table(canonical, result, args)
                out_path = base_out.parent / f"{stem}_frame_{int(frame)}{suffix}"
                export_result_csv(result, str(out_path))
                written.append(out_path)

            print("Results saved in:")
            print(f"  {base_out.parent.resolve()}")
            return 0

    task_cls = TASK_REGISTRY[canonical]
    request = REQUEST_BUILDERS[canonical](args)

    executor = AnalysisExecutor()
    result = executor.run(task_cls(), request, vars(args))
    _prepare_result_table(canonical, result, args)
    present_result(canonical, result, args, plot_payload_builder=_plot_payload)
    return 0
