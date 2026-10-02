"""Compatibility layer for the former nested ``fort7`` command workflow."""

from __future__ import annotations

from reaxkit.presentation.workflow_artifacts import write_workflow_csv

import argparse
from collections.abc import Sequence

import pandas as pd

from reaxkit.cli.path import resolve_output_path
from reaxkit.engine.reaxff.io.fort7_handler import Fort7Handler


def _legacy_analyzer_removed(*_args, **_kwargs):
    raise RuntimeError(
        "The handler-based fort7 analyzer API has been retired; use the direct "
        "get_connection_* and get_bond_events commands."
    )


# Kept as module attributes so applications that monkeypatch or wrap the old
# workflow continue to work while production callers receive a clear message.
get_fort7_data_per_atom = _legacy_analyzer_removed
get_fort7_data_summaries = _legacy_analyzer_removed
connection_list = _legacy_analyzer_removed
connection_stats_over_frames = _legacy_analyzer_removed
bond_timeseries = _legacy_analyzer_removed
bond_events = _legacy_analyzer_removed


def _parse_frames(value: str | Sequence[int] | None):
    if value is None or isinstance(value, (list, tuple)):
        return value
    token = str(value).strip()
    if not token:
        return None
    if ":" in token:
        parts = [int(part) if part else None for part in token.split(":")]
        while len(parts) < 3:
            parts.append(None)
        return slice(parts[0], parts[1], parts[2])
    return [int(part) for part in token.split(",") if part.strip()]


def _export(table: pd.DataFrame, args: argparse.Namespace) -> None:
    if getattr(args, "export", None):
        output = resolve_output_path(args.export, getattr(args, "kind", "fort7"))
        write_workflow_csv(table, output, index=False)


def _task_get(args: argparse.Namespace) -> int:
    handler = Fort7Handler(args.file)
    frames = _parse_frames(getattr(args, "frames", None))
    atom = getattr(args, "atom", None)
    if atom is None:
        table = get_fort7_data_summaries(
            handler,
            args.yaxis,
            frames=frames,
            regex=bool(getattr(args, "regex", False)),
            add_index_cols=True,
        )
    else:
        table = get_fort7_data_per_atom(
            handler,
            args.yaxis,
            atom=atom,
            frames=frames,
            regex=bool(getattr(args, "regex", False)),
            add_index_cols=True,
        )
    if getattr(args, "export", None):
        xaxis = getattr(args, "xaxis", "iter")
        xcol = "frame_idx" if xaxis == "frame" else "iter"
        selected = [column for column in (xcol, args.yaxis) if column in table.columns]
        output = table[selected] if selected else table
        if xaxis == "frame" and "frame_idx" in output.columns:
            output = output.rename(columns={"frame_idx": "frame"})
        _export(output, args)
    return 0


def _task_edges(args: argparse.Namespace) -> int:
    table = connection_list(
        Fort7Handler(args.file),
        frames=_parse_frames(getattr(args, "frames", None)),
        iterations=None,
        min_bo=args.min_bo,
        undirected=not args.directed,
        aggregate=args.aggregate,
        include_self=args.include_self,
    )
    _export(table, args)
    return 0


def _task_constats(args: argparse.Namespace) -> int:
    table = connection_stats_over_frames(
        Fort7Handler(args.file),
        frames=_parse_frames(getattr(args, "frames", None)),
        iterations=None,
        min_bo=args.min_bo,
        undirected=not args.directed,
        how=args.how,
    )
    _export(table, args)
    return 0


def _task_bond_ts(args: argparse.Namespace) -> int:
    table = bond_timeseries(
        Fort7Handler(args.file),
        frames=_parse_frames(getattr(args, "frames", None)),
        iterations=None,
        undirected=not args.directed,
        bo_threshold=args.bo_threshold,
        as_wide=args.wide,
    )
    if table is not None and not table.empty and getattr(args, "export", None):
        output = resolve_output_path(args.export, getattr(args, "kind", "fort7"))
        write_workflow_csv(table, output, index=bool(args.wide))
    return 0


def _task_bond_events(args: argparse.Namespace) -> int:
    table = bond_events(
        Fort7Handler(args.file),
        frames=_parse_frames(getattr(args, "frames", None)),
        iterations=None,
        src=args.src,
        dst=args.dst,
        threshold=args.threshold,
        hysteresis=args.hysteresis,
        smooth=args.smooth,
        window=args.window,
        ema_alpha=args.ema_alpha,
        min_run=args.min_run,
        xaxis=args.xaxis,
        undirected=not args.directed,
    )
    _export(table, args)
    return 0


def _add_output_arguments(parser: argparse.ArgumentParser, *, plot: bool = False) -> None:
    parser.add_argument("--export", default=None, help="CSV destination for extracted data. Example: --export analysis.csv, which writes the result table for further analysis.")
    parser.add_argument("--save", default=None, help="Image destination for generated plots. Example: --save analysis.png, which writes the generated figure to that image.")
    if plot:
        parser.add_argument("--plot", action="store_true", help="Generate a plot of the extracted data. Example: --plot, which enables plotting of the extracted data.")


def register_tasks(subparsers: argparse._SubParsersAction) -> None:
    get_parser = subparsers.add_parser(
        "get",
        formatter_class=argparse.RawTextHelpFormatter,
        description=(
            "Inspect the legacy fort7 interface for atomic-property extraction.\n"
            "\n"
            "This parser is retained for compatibility; its production analyzer has been retired.\n"
            "Use direct get-charge, get_connection_* or get_bond_events commands for analysis.\n"
            "\n"
            "Examples:\n"
            "  1. Inspect the retained options:\n"
            "     reaxkit fort7 get --help"
        ),
    )
    get_parser.add_argument("--file", default="fort.7", help="Input fort.7 file containing charges and bond orders. Example: --file fort.7, which reads the structure or data from fort.7.")
    get_parser.add_argument("--yaxis", required=True, help="Atomic property to extract from fort.7. Example: --yaxis charge, which extracts atomic charge as the plotted quantity.")
    get_parser.add_argument("--atom", default=None, help="Atom identifier used to select an atomic series. Example: --atom 1, which selects atom 1.")
    get_parser.add_argument("--frames", default=None, help="Zero-based source frames to include. Example: --frames 0:20:2, which includes source frames 0, 2, ..., 18.")
    get_parser.add_argument("--xaxis", choices=("iter", "frame", "time"), default="iter", help="Horizontal axis for the extracted series. Example: --xaxis frame, which labels the series by trajectory-frame position.")
    get_parser.add_argument("--control", default="control", help="Control file supplying timestep and output cadence. Example: --control runs/heating/control, which reads simulation cadence and timestep metadata.")
    get_parser.add_argument("--regex", action="store_true", help="Interpret the atomic-property selector as a regular expression. Example: --regex, which interprets the property selector as a regular expression.")
    _add_output_arguments(get_parser, plot=True)
    get_parser.set_defaults(_run=_task_get, kind="fort7")

    edges = subparsers.add_parser(
        "edges",
        formatter_class=argparse.RawTextHelpFormatter,
        description=(
            "Inspect the legacy fort7 interface for bond-edge extraction.\n"
            "\n"
            "This parser is retained for compatibility; its production analyzer has been retired.\n"
            "Use direct get-charge, get_connection_* or get_bond_events commands for analysis.\n"
            "\n"
            "Examples:\n"
            "  1. Inspect the retained options:\n"
            "     reaxkit fort7 edges --help"
        ),
    )
    edges.add_argument("--file", default="fort.7", help="Input fort.7 file containing charges and bond orders. Example: --file fort.7, which reads the structure or data from fort.7.")
    edges.add_argument("--frames", default=None, help="Zero-based source frames to include. Example: --frames 0:20:2, which includes source frames 0, 2, ..., 18.")
    edges.add_argument("--min-bo", type=float, default=0.0, dest="min_bo", help="Minimum bond order to retain. Example: --min-bo 0.3, which excludes bonds below order 0.3.")
    edges.add_argument("--directed", action="store_true", help="Keep directed atom pairs instead of merging opposite directions. Example: --directed, which keeps source-to-target bond direction.")
    edges.add_argument("--aggregate", choices=("max", "mean"), default="max", help="Reduction applied to repeated bond entries. Example: --aggregate mean, which averages repeated bond entries.")
    edges.add_argument("--include-self", action="store_true", dest="include_self", help="Retain edges whose source and destination are the same atom. Example: --include-self, which retains self-pairs in bond data.")
    edges.add_argument("--xaxis", choices=("iter", "frame", "time"), default="frame", help="Horizontal axis for the extracted series. Example: --xaxis frame, which labels the series by trajectory-frame position.")
    edges.add_argument("--control", default="control", help="Control file supplying timestep and output cadence. Example: --control runs/heating/control, which reads simulation cadence and timestep metadata.")
    _add_output_arguments(edges, plot=True)
    edges.set_defaults(_run=_task_edges, kind="fort7")

    stats = subparsers.add_parser(
        "constats",
        formatter_class=argparse.RawTextHelpFormatter,
        description=(
            "Inspect the legacy fort7 interface for connectivity statistics.\n"
            "\n"
            "This parser is retained for compatibility; its production analyzer has been retired.\n"
            "Use direct get-charge, get_connection_* or get_bond_events commands for analysis.\n"
            "\n"
            "Examples:\n"
            "  1. Inspect the retained options:\n"
            "     reaxkit fort7 constats --help"
        ),
    )
    stats.add_argument("--file", default="fort.7", help="Input fort.7 file containing charges and bond orders. Example: --file fort.7, which reads the structure or data from fort.7.")
    stats.add_argument("--frames", default=None, help="Zero-based source frames to include. Example: --frames 0:20:2, which includes source frames 0, 2, ..., 18.")
    stats.add_argument("--min-bo", type=float, default=0.0, dest="min_bo", help="Minimum bond order to retain. Example: --min-bo 0.3, which excludes bonds below order 0.3.")
    stats.add_argument("--directed", action="store_true", help="Keep directed atom pairs instead of merging opposite directions. Example: --directed, which keeps source-to-target bond direction.")
    stats.add_argument("--how", choices=("mean", "max", "count"), default="mean", help="Reduction used to summarize connectivity. Example: --how mean, which averages bond values in the aggregate.")
    _add_output_arguments(stats)
    stats.set_defaults(_run=_task_constats, kind="fort7")

    timeseries = subparsers.add_parser(
        "bond-ts",
        formatter_class=argparse.RawTextHelpFormatter,
        description=(
            "Inspect the legacy fort7 interface for bond histories.\n"
            "\n"
            "This parser is retained for compatibility; its production analyzer has been retired.\n"
            "Use direct get-charge, get_connection_* or get_bond_events commands for analysis.\n"
            "\n"
            "Examples:\n"
            "  1. Inspect the retained options:\n"
            "     reaxkit fort7 bond-ts --help"
        ),
    )
    timeseries.add_argument("--file", default="fort.7", help="Input fort.7 file containing charges and bond orders. Example: --file fort.7, which reads the structure or data from fort.7.")
    timeseries.add_argument("--frames", default=None, help="Zero-based source frames to include. Example: --frames 0:20:2, which includes source frames 0, 2, ..., 18.")
    timeseries.add_argument("--directed", action="store_true", help="Keep directed atom pairs instead of merging opposite directions. Example: --directed, which keeps source-to-target bond direction.")
    timeseries.add_argument("--bo-threshold", type=float, default=0.0, dest="bo_threshold", help="Minimum bond order to include in the series. Example: --bo-threshold 0.4, which requires a bond order of at least 0.4 for retained connectivity.")
    timeseries.add_argument("--wide", action="store_true", help="Return one column per bond instead of a long table. Example: --wide, which places series in separate table columns.")
    timeseries.add_argument("--xaxis", choices=("iter", "frame", "time"), default="iter", help="Horizontal axis for the extracted series. Example: --xaxis frame, which labels the series by trajectory-frame position.")
    timeseries.add_argument("--control", default="control", help="Control file supplying timestep and output cadence. Example: --control runs/heating/control, which reads simulation cadence and timestep metadata.")
    timeseries.add_argument("--src", type=int, default=None, help="Source atom identifier for bond selection. Example: --src 1, which restricts bonds to source atom 1.")
    timeseries.add_argument("--dst", type=int, default=None, help="Destination atom identifier for bond selection. Example: --dst 2, which restricts bonds to destination atom 2.")
    _add_output_arguments(timeseries, plot=True)
    timeseries.set_defaults(_run=_task_bond_ts, kind="fort7")

    events = subparsers.add_parser(
        "bond-events",
        formatter_class=argparse.RawTextHelpFormatter,
        description=(
            "Inspect the legacy fort7 interface for bond-event detection.\n"
            "\n"
            "This parser is retained for compatibility; its production analyzer has been retired.\n"
            "Use direct get-charge, get_connection_* or get_bond_events commands for analysis.\n"
            "\n"
            "Examples:\n"
            "  1. Inspect the retained options:\n"
            "     reaxkit fort7 bond-events --help"
        ),
    )
    events.add_argument("--file", default="fort.7", help="Input fort.7 file containing charges and bond orders. Example: --file fort.7, which reads the structure or data from fort.7.")
    events.add_argument("--frames", default=None, help="Zero-based source frames to include. Example: --frames 0:20:2, which includes source frames 0, 2, ..., 18.")
    events.add_argument("--src", type=int, default=None, help="Source atom identifier for bond selection. Example: --src 1, which restricts bonds to source atom 1.")
    events.add_argument("--dst", type=int, default=None, help="Destination atom identifier for bond selection. Example: --dst 2, which restricts bonds to destination atom 2.")
    events.add_argument("--threshold", type=float, default=0.35, help="Bond-order threshold for bond-state detection. Example: --threshold 0.35, which centers bond-state detection on bond order 0.35.")
    events.add_argument("--hysteresis", type=float, default=0.05, help="Bond-order margin around the state threshold. Example: --hysteresis 0.05, which uses a 0.05 bond-order hysteresis margin.")
    events.add_argument("--smooth", choices=("ma", "ema", "none"), default="ma", help="Smoothing method applied before bond-event detection. Example: --smooth ma, which applies moving-average smoothing.")
    events.add_argument("--window", type=int, default=7, help="Number of samples in the moving-average window. Example: --window 7, which smooths over seven samples.")
    events.add_argument("--ema-alpha", type=float, default=None, dest="ema_alpha", help="New-sample weight for exponential moving-average smoothing. Example: --ema-alpha 0.2, which weights each new sample by 0.2 in exponential smoothing.")
    events.add_argument("--min-run", type=int, default=3, dest="min_run", help="Minimum number of consecutive samples for an accepted state. Example: --min-run 3, which requires a state to persist for three samples.")
    events.add_argument("--xaxis", choices=("iter", "frame"), default="iter", help="Horizontal axis for the extracted series. Example: --xaxis frame, which labels the series by trajectory-frame position.")
    events.add_argument("--directed", action="store_true", help="Keep directed atom pairs instead of merging opposite directions. Example: --directed, which keeps source-to-target bond direction.")
    _add_output_arguments(events)
    events.set_defaults(_run=_task_bond_events, kind="fort7")


__all__ = [
    "register_tasks",
    "_task_get",
    "_task_edges",
    "_task_constats",
    "_task_bond_ts",
    "_task_bond_events",
]
