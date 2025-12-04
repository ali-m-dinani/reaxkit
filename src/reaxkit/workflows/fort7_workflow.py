"""used to read a fort.7 file, getting its data, or calculating bond events and bond connectivities across selected frames."""
from __future__ import annotations
import argparse
import pandas as pd
from typing import Optional, Sequence, Union

# --- Direct imports (no _import_or_die, no fallbacks) ---
from reaxkit.utils.path import resolve_output_path
from reaxkit.io.fort7_handler import Fort7Handler
from reaxkit.analysis.fort7_analyzer import get_features_atom, get_features_summary
from reaxkit.analysis.connectivity_analyzer import (
    connection_list,
    connection_stats_over_frames,
    bond_timeseries,
    bond_events,
    debug_bond_trace_overlay,
)
from reaxkit.analysis.plotter import single_plot
from reaxkit.utils.frame_utils import parse_frames
from reaxkit.utils.alias import normalize_choice
from reaxkit.utils.convert import convert_xaxis

FramesT = Optional[Union[slice, Sequence[int]]]


# ==========================================================
# Task: GET  (atomic or summary feature)
# ==========================================================
def _task_get(args: argparse.Namespace) -> int:
    h = Fort7Handler(args.file)

    frames_sel = parse_frames(args.frames)
    feat = (args.feature or "").strip()
    feat = normalize_choice(feat, domain="feature")
    use_regex = bool(args.regex)
    is_atom_scope = args.atom is not None

    if is_atom_scope:
        if args.atom is None:
            raise SystemExit("❌ For atom-level features, provide --atom <1-based atom_num>.")
        df = get_features_atom(h, feat, frames=frames_sel, regex=use_regex, add_index_cols=True)
        if df.empty:
            raise SystemExit("❌ No atom-level rows matched your request.")
        if "atom_num" not in df.columns:
            needed = {"frame_idx", "atom_idx"}
            if not needed.issubset(df.columns):
                raise SystemExit("❌ Internal: missing frame_idx/atom_idx to recover atom numbers.")
            rows = []
            for fi, g in df.groupby("frame_idx"):
                fr = h._frames[int(fi)]
                take = g.copy()
                take["atom_num"] = fr.loc[take["atom_idx"].values, "atom_num"].to_numpy()
                rows.append(take)
            df = pd.concat(rows, ignore_index=True)
        df = df[df["atom_num"] == int(args.atom)].copy()
        if df.empty:
            raise SystemExit(f"❌ Atom with atom_num={args.atom} not found in selected frames.")

        keep_meta = {"frame_idx", "iter", "atom_idx", "atom_num"}
        ycols = [c for c in df.columns if c not in keep_meta]
        if not ycols:
            raise SystemExit("❌ No feature columns found after selection.")
        if len(ycols) == 1:
            ylab = ycols[0]
            y = df[ylab].to_numpy()
        else:
            ylab = f"mean({feat})"
            y = df[ycols].mean(axis=1).to_numpy()

        xchoice = normalize_choice(args.xaxis, domain="xaxis")
        if xchoice == "frame":
            x_raw = df["frame_idx"].to_numpy(); xlabel = "frame"
        else:
            x_raw = df["iter"].to_numpy(); xlabel = "iter"
        if xchoice == "time":
            x, xlabel = convert_xaxis(x_raw, "time", control_file=args.control)
        else:
            x = x_raw

        out_df = pd.DataFrame({xlabel: x, ylab: y})
        out_df.insert(0, "atom_num", int(args.atom))
    else:
        df = get_features_summary(h, feat, frames=frames_sel, regex=use_regex, add_index_cols=True)
        if df.empty:
            raise SystemExit("❌ No summary rows matched your request.")
        keep_meta = {"frame_idx", "iter"}
        ycols = [c for c in df.columns if c not in keep_meta]
        if not ycols:
            raise SystemExit("❌ No summary feature columns found.")
        if len(ycols) == 1:
            ylab = ycols[0]
            y = df[ylab].to_numpy()
        else:
            ylab = f"mean({feat})"
            y = df[ycols].mean(axis=1).to_numpy()

        xchoice = normalize_choice(args.xaxis, domain="xaxis")
        if xchoice == "frame":
            x_raw = df["frame_idx"].to_numpy(); xlabel = "frame"
        else:
            x_raw = df["iter"].to_numpy(); xlabel = "iter"
        if xchoice == "time":
            x, xlabel = convert_xaxis(x_raw, "time", control_file=args.control)
        else:
            x = x_raw

        out_df = pd.DataFrame({xlabel: x, ylab: y})

    workflow_name = args.kind
    if args.export:
        out = resolve_output_path(args.export, workflow_name)
        out_df.to_csv(out, index=False)
        print(f"[Done] Exported data to {out}")

    if args.save or args.plot:
        series = [{'x': out_df.iloc[:, 1].to_numpy() if out_df.columns[0] == "atom_num" else out_df.iloc[:, 0].to_numpy(),
                   'y': out_df.iloc[:, -1].to_numpy(),
                   'label': ylab}]
        # fix X for atom_num case
        xlabel = out_df.columns[1] if out_df.columns[0] == "atom_num" else out_df.columns[0]
        if args.save:
            out = resolve_output_path(args.save, workflow_name)
            single_plot(series=series, title=f"{ylab} vs {xlabel}", xlabel=xlabel, ylabel=ylab, save=out,
                        legend=True)
        elif args.plot:
            single_plot(series=series, title=f"{ylab} vs {xlabel}", xlabel=xlabel, ylabel=ylab, save=None,
                        legend=True)

    if not (args.export or args.plot or args.save):
        print("ℹ️ No action selected. Use one or more of --plot, --save, --export.")
    return 0


# ==========================================================
# Task: EDGES  (connection_list → tidy edge table)
# ==========================================================
def _task_edges(args: argparse.Namespace) -> int:
    h = Fort7Handler(args.file)
    frames_sel = parse_frames(args.frames)
    edges = connection_list(
        h,
        frames=frames_sel,
        iterations=None,
        min_bo=args.min_bo,
        undirected=not args.directed,
        aggregate=args.aggregate,
        include_self=args.include_self,
    )
    if edges.empty:
        print("ℹ️ No edges found for the given selection.")

    workflow_name = args.kind
    if args.export:
        out = resolve_output_path(args.export, workflow_name)
        edges.to_csv(out, index=False)
        print(f"[Done] Exported edges to {out}")

    if args.save or args.plot:
        if edges.empty:
            print("ℹ️ Nothing to plot.")
        else:
            counts = edges.groupby("frame_idx", as_index=False).size().rename(columns={"size": "edges"})
            xchoice = normalize_choice(args.xaxis, domain="xaxis")
            if xchoice == "iter":
                f2i = edges.drop_duplicates("frame_idx")[["frame_idx", "iter"]]
                counts = counts.merge(f2i, on="frame_idx", how="left")
                x_raw, xlabel = counts["iter"].to_numpy(), "iter"
            elif xchoice == "time":
                f2i = edges.drop_duplicates("frame_idx")[["frame_idx", "iter"]]
                counts = counts.merge(f2i, on="frame_idx", how="left")
                x_raw, xlabel = convert_xaxis(counts["iter"].to_numpy(), "time", control_file=args.control)
            else:
                x_raw, xlabel = counts["frame_idx"].to_numpy(), "frame"
            y = counts["edges"].to_numpy()
            series = [{'x': x_raw, 'y': y, 'label': '#edges'}]
            if args.save:
                out = resolve_output_path(args.save, workflow_name)
                single_plot(series=series, title=f"#edges vs {xlabel}", xlabel=xlabel, ylabel="#edges",
                            save=out, legend=False)
            elif args.plot:
                single_plot(series=series, title=f"#edges vs {xlabel}", xlabel=xlabel, ylabel="#edges",
                            save=None, legend=False)

    if not (args.export or args.plot or args.save):
        print("ℹ️ No action selected. Use --export to save CSV or --plot/--save to visualize edge counts.")
    return 0


# ==========================================================
# Task: CONSTATs (connection_stats_over_frames)
# ==========================================================
def _task_constats(args: argparse.Namespace) -> int:
    h = Fort7Handler(args.file)
    frames_sel = parse_frames(args.frames)
    stats = connection_stats_over_frames(
        h,
        frames=frames_sel,
        iterations=None,
        min_bo=args.min_bo,
        undirected=not args.directed,
        how=args.how,
    )
    if stats.empty:
        print("ℹ️ No connection stats for the given selection.")
    workflow_name = args.kind
    if args.export:
        out = resolve_output_path(args.export, workflow_name)
        stats.to_csv(out, index=False)
        print(f"[Done] Exported stats to {out}")
    else:
        print("ℹ️ Use --export to save stats (src, dst, value).")
    return 0


# ==========================================================
# Task: BOND-TS (bond_timeseries)
# ==========================================================
def _task_bond_ts(args: argparse.Namespace) -> int:
    h = Fort7Handler(args.file)
    frames_sel = parse_frames(args.frames)
    ts = bond_timeseries(
        h,
        frames=frames_sel,
        iterations=None,
        undirected=not args.directed,
        bo_threshold=args.bo_threshold,
        as_wide=args.wide,
    )
    if ts is None or getattr(ts, "empty", False):
        print("ℹ️ No bond time series produced.")
        return 0

    workflow_name = args.kind
    if args.export:
        out = resolve_output_path(args.export, workflow_name)
        ts.to_csv(out, index=True if args.wide else False)
        print(f"[Done] Exported bond time series to {out}")

    if (args.save or args.plot) and (args.src and args.dst) and not args.wide:
        a, b = int(args.src), int(args.dst)
        if not args.directed and a > b:
            a, b = b, a
        g = ts[(ts["src"] == a) & (ts["dst"] == b)].copy()
        if g.empty:
            print(f"ℹ️ Bond {a}-{b} not found in the selection.")
        else:
            xchoice = normalize_choice(args.xaxis, domain="xaxis")
            if xchoice == "frame":
                x_raw, xlabel = g["frame_idx"].to_numpy(), "frame"
            elif xchoice == "time":
                x_raw, xlabel = convert_xaxis(g["iter"].to_numpy(), "time", control_file=args.control)
            else:
                x_raw, xlabel = g["iter"].to_numpy(), "iter"
            y = g["bo"].to_numpy()
            if args.save:
                out = resolve_output_path(args.save, workflow_name)
                single_plot(series=[{'x': x_raw, 'y': y, 'label': f'BO {a}-{b}'}],
                        title=f"BO({a}-{b}) vs {xlabel}", xlabel=xlabel, ylabel="BO", save=out, legend=False)
            elif args.plot:
                single_plot(series=[{'x': x_raw, 'y': y, 'label': f'BO {a}-{b}'}],
                            title=f"BO({a}-{b}) vs {xlabel}", xlabel=xlabel, ylabel="BO", save=None, legend=False)
    elif (args.save or args.plot) and not (args.src and args.dst):
        print("ℹ️ Provide --src and --dst to plot a specific bond trace.")

    if not (args.export or args.plot or args.save):
        print("ℹ️ No action selected. Use --export to save CSV, or --plot/--save with --src/--dst to visualize a bond.")
    return 0


# ==========================================================
# Task: BOND-EVENTS (bond_events + optional overlay)
# ==========================================================
def _task_bond_events(args: argparse.Namespace) -> int:
    print('This is a time-consuming task; please be patient ...')
    h = Fort7Handler(args.file)
    frames_sel = parse_frames(args.frames)
    events = bond_events(
        h,
        frames=frames_sel,
        iterations=None,
        src=args.src,
        dst=args.dst,
        threshold=args.threshold,
        hysteresis=args.hysteresis,
        smooth=args.smooth,
        window=args.window,
        ema_alpha=args.ema_alpha,
        min_run=args.min_run,
        xaxis=args.xaxis if args.xaxis in ("iter", "frame") else "iter",
        undirected=not args.directed,
    )
    if events.empty:
        print("ℹ️ No bond formation/breakage events detected with the given settings.")

    workflow_name = args.kind
    if args.export:
        out = resolve_output_path(args.export, workflow_name)
        events.to_csv(out, index=False)
        print(f"[Done] Exported events to {out}")

    if args.save or args.plot and args.src and args.dst:
        out = resolve_output_path(args.save, workflow_name)
        if args.save:
            debug_bond_trace_overlay(
                h,
                src=int(args.src),
                dst=int(args.dst),
                smooth=("ema" if args.smooth == "ema" else "ma"),
                window=int(args.window),
                hysteresis=float(args.hysteresis),
                threshold=float(args.threshold),
                min_run=int(args.min_run or 0),
                xaxis=("iter" if args.xaxis == "iter" else "frame"),
                save=out,
            )
        else:
            debug_bond_trace_overlay(
                h,
                src=int(args.src),
                dst=int(args.dst),
                smooth=("ema" if args.smooth == "ema" else "ma"),
                window=int(args.window),
                hysteresis=float(args.hysteresis),
                threshold=float(args.threshold),
                min_run=int(args.min_run or 0),
                xaxis=("iter" if args.xaxis == "iter" else "frame"),
                save=None,
            )
    elif args.save:
        print("ℹ️ --save requires both --src and --dst to be set.")

    if not (args.export or args.save):
        print("ℹ️ No action selected. Use --export to save CSV or --save to write a debug figure.")
    return 0


# ==========================================================
# CLI wiring
# ==========================================================
import argparse

# ---------------------------------------------------------------------------
# Common argument helpers
# ---------------------------------------------------------------------------

def _add_common_fort7_file_arg(p: argparse.ArgumentParser) -> None:
    p.add_argument("--file", default="fort.7", help="Path to fort.7 file.")

def _add_common_io_args(
    p: argparse.ArgumentParser,
    *,
    include_plot: bool = False,
    save_help: str = "Path to save plot image.",
    export_help: str = "Path to export CSV data.",
) -> None:
    if include_plot:
        p.add_argument("--plot", action="store_true", help="Show plot interactively.")
    p.add_argument("--save", default=None, help=save_help)
    p.add_argument("--export", default=None, help=export_help)

# ---------------------------------------------------------------------------
# Wiring functions
# ---------------------------------------------------------------------------

def _wire_get(p: argparse.ArgumentParser) -> None:
    _add_common_fort7_file_arg(p)
    p.add_argument("--feature", required=True, help="Feature name or regex (with --regex).")
    p.add_argument("--atom", type=int, default=None, help="1-based atom index for atom-level features.")
    p.add_argument("--frames", default=None, help="Frame selection: 'a:b[:c]' or 'i,j,k'.")
    p.add_argument("--xaxis", default="iter", choices=["iter","frame","time"], help="X-axis mode.")
    p.add_argument("--control", default="control", help="Control file (for --xaxis time).")
    p.add_argument("--regex", action="store_true", help="Interpret --feature as regex.")
    _add_common_io_args(p, include_plot=True,
                        save_help="Save feature plot image.",
                        export_help="Export extracted feature(s) as CSV.")
    p.set_defaults(_run=_task_get)

def _wire_edges(p: argparse.ArgumentParser) -> None:
    _add_common_fort7_file_arg(p)
    p.add_argument("--frames", default=None, help="Frame selection.")
    p.add_argument("--min-bo", type=float, default=0.0, dest="min_bo", help="Minimum BO.")
    p.add_argument("--directed", action="store_true", help="Treat edges as directed.")
    p.add_argument("--aggregate", choices=["max","mean"], default="max",
                   help="Aggregation for undirected edges.")
    p.add_argument("--include-self", action="store_true", dest="include_self", help="Keep self-edges.")
    p.add_argument("--xaxis", default="frame", choices=["iter","frame","time"],
                   help="X-axis for quick plot.")
    p.add_argument("--control", default="control", help="Control file for --xaxis time.")
    _add_common_io_args(p, include_plot=True,
                        save_help="Save edge-count plot.",
                        export_help="Export edge list CSV.")
    p.set_defaults(_run=_task_edges)

def _wire_constats(p: argparse.ArgumentParser) -> None:
    _add_common_fort7_file_arg(p)
    p.add_argument("--frames", default=None, help="Frame selection.")
    p.add_argument("--min-bo", type=float, default=0.0, dest="min_bo", help="BO threshold before stats.")
    p.add_argument("--directed", action="store_true", help="Do not merge A–B with B–A.")
    p.add_argument("--how", choices=["mean","max","count"], default="mean", help="Statistic to compute.")
    _add_common_io_args(p, include_plot=False,
                        save_help="(Unused) No plot.",
                        export_help="Export connection stats as CSV.")
    p.set_defaults(_run=_task_constats)

def _wire_bond_ts(p: argparse.ArgumentParser) -> None:
    _add_common_fort7_file_arg(p)
    p.add_argument("--frames", default=None, help="Frame selection.")
    p.add_argument("--directed", action="store_true", help="Do not merge A–B with B–A.")
    p.add_argument("--bo-threshold", type=float, default=0.0, dest="bo_threshold",
                   help="Zero out BO below this.")
    p.add_argument("--wide", action="store_true", help="Return wide matrix (frames × bonds).")
    p.add_argument("--xaxis", default="iter", choices=["iter","frame","time"],
                   help="X-axis for quick plot.")
    p.add_argument("--control", default="control", help="Control file for --xaxis time.")
    p.add_argument("--src", type=int, help="Source atom for quick plot.")
    p.add_argument("--dst", type=int, help="Destination atom for quick plot.")
    _add_common_io_args(p, include_plot=True,
                        save_help="Save bond time-series plot.",
                        export_help="Export bond-order time series CSV.")
    p.set_defaults(_run=_task_bond_ts)

def _wire_bond_events(p: argparse.ArgumentParser) -> None:
    _add_common_fort7_file_arg(p)
    p.add_argument("--frames", default=None, help="Frame selection.")
    p.add_argument("--src", type=int, help="Source atom.")
    p.add_argument("--dst", type=int, help="Destination atom.")
    p.add_argument("--threshold", type=float, default=0.35, help="Schmitt trigger base threshold.")
    p.add_argument("--hysteresis", type=float, default=0.05,
                   help="Hysteresis width around threshold.")
    p.add_argument("--smooth", choices=["ma","ema","none"], default="ma", help="Smoothing method.")
    p.add_argument("--window", type=int, default=7, help="Window size for MA/EMA.")
    p.add_argument("--ema-alpha", type=float, default=None, dest="ema_alpha", help="Optional EMA alpha.")
    p.add_argument("--min-run", type=int, default=3, dest="min_run", help="Minimum consecutive points.")
    p.add_argument("--xaxis", default="iter", choices=["iter","frame"], help="Internal event x-axis.")
    p.add_argument("--directed", action="store_true", help="Do not merge A–B/B–A.")
    _add_common_io_args(p, include_plot=False,
                        save_help="Save debug overlay (requires --src --dst).",
                        export_help="Export detected events CSV.")
    p.set_defaults(_run=_task_bond_events)

# ---------------------------------------------------------------------------
# Task registration
# ---------------------------------------------------------------------------

def register_tasks(subparsers: argparse._SubParsersAction) -> None:
    """
    Register subcommands under the 'fort7' namespace.
    """

    # GET
    p_get = subparsers.add_parser(
        "get",
        help="Extract a feature and optionally plot/save/export.",
        description=(
            "Examples:\n"
            "  reaxkit fort7 get --feature charge --atom 1 --plot\n"
            "  reaxkit fort7 get --feature q_.* --regex --export charges.csv\n"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    _wire_get(p_get)

    # EDGES
    p_edges = subparsers.add_parser(
        "edges",
        help="Build a tidy edge list from fort.7.",
        description=(
            "Examples:\n"
            "  reaxkit fort7 edges --frames 0:1000:10 --min-bo 0.4 --export edges.csv\n"
            "  reaxkit fort7 edges --plot --min-bo 0.3\n"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    _wire_edges(p_edges)

    # CONSTATs
    p_constats = subparsers.add_parser(
        "constats",
        help="Aggregate connection statistics across frames.",
        description=(
            "Examples:\n"
            "  reaxkit fort7 constats --frames 0:1000 --how mean --export stats.csv\n"
            "  reaxkit fort7 constats --how count --min-bo 0.4 --export counts.csv\n"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    _wire_constats(p_constats)

    # BOND-TS
    p_bts = subparsers.add_parser(
        "bond-ts",
        help="Bond-order time series; optional quick plot.",
        description=(
            "Examples:\n"
            "  reaxkit fort7 bond-ts --frames 0:500 --export bo.csv\n"
            "  reaxkit fort7 bond-ts --src 1 --dst 19 --plot\n"
            "  reaxkit fort7 bond-ts --wide --bo-threshold 0.1 --export wide.csv\n"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    _wire_bond_ts(p_bts)

    # BOND-EVENTS
    p_bev = subparsers.add_parser(
        "bond-events",
        help="Detect bond formation/breakage events.",
        description=(
            "Examples:\n"
            "  reaxkit fort7 bond-events --export events.csv\n"
            "  reaxkit fort7 bond-events --src 1 --dst 19 --threshold 0.38 "
            "--hysteresis 0.10 --smooth ema --window 7 --min-run 4 "
            "--export events_1_19.csv --save overlay.png\n"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    _wire_bond_events(p_bev)

