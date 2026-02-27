"""Top-level time-series dispatcher workflow."""

from __future__ import annotations

import argparse
import re

import numpy as np
import pandas as pd

from reaxkit.analysis import timeseries as _timeseries_tasks  # noqa: F401
from reaxkit.analysis.timeseries.timeseries import (
    SimulationScalarSeriesRequest,
    SimulationScalarSeriesTask,
    TimeSeriesResult,
    TrajectoryCoordinateSeriesRequest,
    TrajectoryCoordinateSeriesTask,
)
from reaxkit.cli.path import resolve_output_path
from reaxkit.core.analysis_executor import AnalysisExecutor
from reaxkit.core.frame_utils import parse_frames
from reaxkit.presentation.convert import convert_xaxis
from reaxkit.presentation.plot import single_plot

_ATOM_FIELD_RE = re.compile(r"^atom\[(?P<ids>[0-9,\s]+)\]\.(?P<axis>[xyzXYZ])$")

_SIM_FIELDS = {
    "potential_energy": "potential_energy",
    "num_of_atoms": "num_of_atoms",
    "a": "a",
    "b": "b",
    "c": "c",
    "alpha": "alpha",
    "beta": "beta",
    "gamma": "gamma",
}


def _parse_frame_selector(spec: str | None) -> list[int] | None:
    if not spec:
        return None
    sel = parse_frames(spec)
    if sel is None:
        return None
    if isinstance(sel, slice):
        start = 0 if sel.start is None else int(sel.start)
        stop = int(sel.stop) if sel.stop is not None else start
        step = 1 if sel.step is None else int(sel.step)
        return list(range(start, stop, step))
    return [int(v) for v in sel]


def _resolve_task_and_request(args: argparse.Namespace):
    field = str(args.field).strip()
    match = _ATOM_FIELD_RE.match(field)
    frames = _parse_frame_selector(args.frames)
    if match:
        atom_ids = [int(tok) for tok in re.split(r"[\s,]+", match.group("ids").strip()) if tok]
        return (
            TrajectoryCoordinateSeriesTask(),
            TrajectoryCoordinateSeriesRequest(
                atom_ids=atom_ids,
                dims=(match.group("axis").lower(),),
                frames=frames,
                every=int(args.every),
            ),
            {
                "engine": args.engine,
                "input": args.input,
                "run_dir": args.run_dir,
                "xmolout": args.xmolout,
            },
        )

    sim_field = _SIM_FIELDS.get(field.lower())
    if sim_field is not None:
        return (
            SimulationScalarSeriesTask(),
            SimulationScalarSeriesRequest(field=sim_field, frames=frames, every=int(args.every)),
            {
                "engine": args.engine,
                "input": args.input,
                "run_dir": args.run_dir,
                "xmolout": args.xmolout,
            },
        )

    raise ValueError(
        f"Unsupported field {field!r}. "
        "Examples: potential_energy, num_of_atoms, a, alpha, atom[1].x, atom[1,2,3].z"
    )


def _series_with_xaxis(result: TimeSeriesResult, args: argparse.Namespace):
    meta = result.metadata or {}
    iterations = np.asarray(meta.get("iterations", np.empty((0,), dtype=int)), dtype=int)
    xvals, xlabel = convert_xaxis(iterations, args.xaxis, control_file=args.control)
    out = []
    for series in result.series:
        out.append({"x": xvals, "y": np.asarray(series.y, dtype=float), "label": series.label})
    return out, xlabel


def _result_to_frame(result: TimeSeriesResult, xlabel: str, xvals: np.ndarray) -> pd.DataFrame:
    if not result.series:
        return pd.DataFrame(columns=[xlabel])

    if len(result.series) == 1:
        s = result.series[0]
        return pd.DataFrame({xlabel: xvals, s.label: np.asarray(s.y, dtype=float)})

    same_x = all(np.array_equal(np.asarray(s.x), np.asarray(result.series[0].x)) for s in result.series)
    if same_x:
        out = pd.DataFrame({xlabel: xvals})
        for s in result.series:
            out[s.label] = np.asarray(s.y, dtype=float)
        return out

    rows = []
    for s in result.series:
        for xv, yv in zip(np.asarray(s.x), np.asarray(s.y)):
            rows.append({xlabel: xv, "value": yv, "label": s.label})
    return pd.DataFrame(rows)


def build_parser(p: argparse.ArgumentParser) -> None:
    p.add_argument("--field", required=True, help="Series field, e.g. potential_energy or atom[1,2,3].x")
    p.add_argument("--engine", choices=["reaxff", "ams", "lammps"], default=None)
    p.add_argument("--input", default=".", help="Input file or directory for engine resolution")
    p.add_argument("--run-dir", default=".", help="Run directory fallback for engine detection")
    p.add_argument("--xmolout", default="xmolout", help="Path to xmolout")
    p.add_argument("--control", default="control", help="Path to control file for time-axis conversion")
    p.add_argument("--xaxis", choices=["iter", "frame", "time"], default="iter", help="X-axis domain")
    p.add_argument("--frames", default=None, help="Frame selector: start:stop[:step] or i,j,k")
    p.add_argument("--every", type=int, default=1, help="Use every Nth selected frame")
    p.add_argument("--plot", action="store_true", help="Show plot interactively")
    p.add_argument("--save", default=None, help="Save plot to file")
    p.add_argument("--export", default=None, help="Export time-series table to CSV")


def run_main(args: argparse.Namespace) -> int:
    task, request, load_args = _resolve_task_and_request(args)
    executor = AnalysisExecutor()
    result = executor.run(task, request, load_args)
    if not result.series:
        print("No time-series data produced for the requested field.")
        return 1

    plot_series, xlabel = _series_with_xaxis(result, args)
    xvals = np.asarray(plot_series[0]["x"], dtype=float) if plot_series else np.empty((0,), dtype=float)
    df = _result_to_frame(result, xlabel, xvals)

    if args.export:
        out = resolve_output_path(args.export, "timeseries")
        df.to_csv(out, index=False)
        print(f"[Done] Exported time-series data to {out}")

    if args.save:
        out = resolve_output_path(args.save, "timeseries")
        single_plot(
            series=plot_series,
            title=f"{result.y_label} vs {xlabel}",
            xlabel=xlabel,
            ylabel=result.y_label,
            legend=len(plot_series) > 1,
            save=out,
        )

    if args.plot:
        single_plot(
            series=plot_series,
            title=f"{result.y_label} vs {xlabel}",
            xlabel=xlabel,
            ylabel=result.y_label,
            legend=len(plot_series) > 1,
            save=None,
        )

    if not args.export and not args.save and not args.plot:
        print(df.head(20).to_string(index=False))

    return 0


def register_tasks(subparsers: argparse._SubParsersAction) -> None:
    _ = subparsers
    return
