"""Compatibility layer for the former nested ``fort7`` command workflow."""

from __future__ import annotations

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
        table.to_csv(output, index=False)


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
        table.to_csv(output, index=bool(args.wide))
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
    parser.add_argument("--export", default=None)
    parser.add_argument("--save", default=None)
    if plot:
        parser.add_argument("--plot", action="store_true")


def register_tasks(subparsers: argparse._SubParsersAction) -> None:
    get_parser = subparsers.add_parser("get")
    get_parser.add_argument("--file", default="fort.7")
    get_parser.add_argument("--yaxis", required=True)
    get_parser.add_argument("--atom", default=None)
    get_parser.add_argument("--frames", default=None)
    get_parser.add_argument("--xaxis", choices=("iter", "frame", "time"), default="iter")
    get_parser.add_argument("--control", default="control")
    get_parser.add_argument("--regex", action="store_true")
    _add_output_arguments(get_parser, plot=True)
    get_parser.set_defaults(_run=_task_get, kind="fort7")

    edges = subparsers.add_parser("edges")
    edges.add_argument("--file", default="fort.7")
    edges.add_argument("--frames", default=None)
    edges.add_argument("--min-bo", type=float, default=0.0, dest="min_bo")
    edges.add_argument("--directed", action="store_true")
    edges.add_argument("--aggregate", choices=("max", "mean"), default="max")
    edges.add_argument("--include-self", action="store_true", dest="include_self")
    edges.add_argument("--xaxis", choices=("iter", "frame", "time"), default="frame")
    edges.add_argument("--control", default="control")
    _add_output_arguments(edges, plot=True)
    edges.set_defaults(_run=_task_edges, kind="fort7")

    stats = subparsers.add_parser("constats")
    stats.add_argument("--file", default="fort.7")
    stats.add_argument("--frames", default=None)
    stats.add_argument("--min-bo", type=float, default=0.0, dest="min_bo")
    stats.add_argument("--directed", action="store_true")
    stats.add_argument("--how", choices=("mean", "max", "count"), default="mean")
    _add_output_arguments(stats)
    stats.set_defaults(_run=_task_constats, kind="fort7")

    timeseries = subparsers.add_parser("bond-ts")
    timeseries.add_argument("--file", default="fort.7")
    timeseries.add_argument("--frames", default=None)
    timeseries.add_argument("--directed", action="store_true")
    timeseries.add_argument("--bo-threshold", type=float, default=0.0, dest="bo_threshold")
    timeseries.add_argument("--wide", action="store_true")
    timeseries.add_argument("--xaxis", choices=("iter", "frame", "time"), default="iter")
    timeseries.add_argument("--control", default="control")
    timeseries.add_argument("--src", type=int, default=None)
    timeseries.add_argument("--dst", type=int, default=None)
    _add_output_arguments(timeseries, plot=True)
    timeseries.set_defaults(_run=_task_bond_ts, kind="fort7")

    events = subparsers.add_parser("bond-events")
    events.add_argument("--file", default="fort.7")
    events.add_argument("--frames", default=None)
    events.add_argument("--src", type=int, default=None)
    events.add_argument("--dst", type=int, default=None)
    events.add_argument("--threshold", type=float, default=0.35)
    events.add_argument("--hysteresis", type=float, default=0.05)
    events.add_argument("--smooth", choices=("ma", "ema", "none"), default="ma")
    events.add_argument("--window", type=int, default=7)
    events.add_argument("--ema-alpha", type=float, default=None, dest="ema_alpha")
    events.add_argument("--min-run", type=int, default=3, dest="min_run")
    events.add_argument("--xaxis", choices=("iter", "frame"), default="iter")
    events.add_argument("--directed", action="store_true")
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
