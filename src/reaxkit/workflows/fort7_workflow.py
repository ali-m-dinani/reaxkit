"""used to read a fort.7 file, getting its data, or calculating bond events and bond connectivities across selected frames."""
from __future__ import annotations
import argparse
import pandas as pd
from typing import Optional, Sequence, Union

# --- Direct imports (no _import_or_die, no fallbacks) ---
from reaxkit.io.fort7_handler import Fort7Handler
from reaxkit.analysis.fort7_analyzer import features_atom, features_summary
from reaxkit.analysis.connectivity_analyzer import (
    connection_list,
    connection_stats_over_frames,
    bond_timeseries,
    bond_events,
    debug_bond_trace_overlay,
)
from reaxkit.analysis.plotter import single_plot
from reaxkit.utils.frame_utils import parse_frames
from reaxkit.utils.alias_utils import normalize_choice
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
        df = features_atom(h, feat, frames=frames_sel, regex=use_regex, add_index_cols=True)
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
        df = features_summary(h, feat, frames=frames_sel, regex=use_regex, add_index_cols=True)
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

    if args.export:
        out_df.to_csv(args.export, index=False)
        print(f"[Done] Exported data to {args.export}")

    if args.save or args.plot:
        series = [{'x': out_df.iloc[:, 1].to_numpy() if out_df.columns[0] == "atom_num" else out_df.iloc[:, 0].to_numpy(),
                   'y': out_df.iloc[:, -1].to_numpy(),
                   'label': ylab}]
        # fix X for atom_num case
        xlabel = out_df.columns[1] if out_df.columns[0] == "atom_num" else out_df.columns[0]
        single_plot(series=series, title=f"{ylab} vs {xlabel}", xlabel=xlabel, ylabel=ylab, save=args.save, legend=True)

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

    if args.export:
        edges.to_csv(args.export, index=False)
        print(f"[Done] Exported edges to {args.export}")

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
            single_plot(series=series, title=f"#edges vs {xlabel}", xlabel=xlabel, ylabel="#edges", save=args.save, legend=False)

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
    if args.export:
        stats.to_csv(args.export, index=False)
        print(f"[Done] Exported stats to {args.export}")
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

    if args.export:
        ts.to_csv(args.export, index=True if args.wide else False)
        print(f"[Done] Exported bond time series to {args.export}")

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
            single_plot(series=[{'x': x_raw, 'y': y, 'label': f'BO {a}-{b}'}],
                        title=f"BO({a}-{b}) vs {xlabel}", xlabel=xlabel, ylabel="BO", save=args.save, legend=False)
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
    if args.export:
        events.to_csv(args.export, index=False)
        print(f"[Done] Exported events to {args.export}")

    if args.save and args.src and args.dst:
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
            save=args.save,
        )
    elif args.save:
        print("ℹ️ --save requires both --src and --dst to be set.")

    if not (args.export or args.save):
        print("ℹ️ No action selected. Use --export to save CSV or --save to write a debug figure.")
    return 0


# ==========================================================
# CLI wiring
# ==========================================================
def _wire_get(p: argparse.ArgumentParser) -> None:
    p.add_argument('--file', default='fort.7', help='Path to fort.7 file')
    p.add_argument('--feature', required=True,
                   help="Feature to extract (atom or summary). Regex allowed with --regex.")
    p.add_argument('--atom', type=int, default=None,
                   help='1-based atom_num for atom-level features (required for atom scope).')
    p.add_argument('--frames', default=None,
                   help="Frames to select: 'start:stop[:step]' or 'i,j,k' (default: all).")
    p.add_argument('--xaxis', default='iter', choices=['iter', 'frame', 'time'],
                   help='X-axis for plotting/export.')
    p.add_argument('--control', default='control', help='Control file (for --xaxis time).')
    p.add_argument('--regex', action='store_true', help='Treat --feature as a regex.')
    p.add_argument('--plot', action='store_true', help='Show plot interactively.')
    p.add_argument('--save', default=None, help='Save plot to image path.')
    p.add_argument('--export', default=None, help='Export data to CSV (path).')
    p.set_defaults(_run=_task_get)

def _wire_edges(p: argparse.ArgumentParser) -> None:
    p.add_argument('--file', default='fort.7', help='Path to fort.7 file')
    p.add_argument('--frames', default=None, help="Frame selection: 'start:stop[:step]' or 'i,j,k'.")
    p.add_argument('--min-bo', type=float, default=0.0, dest='min_bo', help='Keep edges with BO ≥ this threshold.')
    p.add_argument('--directed', action='store_true', help='Treat bonds as directed (default undirected).')
    p.add_argument('--aggregate', choices=['max', 'mean'], default='max',
                   help='When undirected, aggregate duplicates by max or mean BO.')
    p.add_argument('--include-self', action='store_true', dest='include_self', help='Keep self-edges (default: drop).')
    p.add_argument('--xaxis', default='frame', choices=['iter', 'frame', 'time'], help='X-axis for quick count plot.')
    p.add_argument('--control', default='control', help='Control file (for --xaxis time).')
    p.add_argument('--plot', action='store_true', help='Quick plot: #edges vs x-axis.')
    p.add_argument('--save', default=None, help='Save quick plot image.')
    p.add_argument('--export', default=None, help='Export edge list CSV.')
    p.set_defaults(_run=_task_edges)

def _wire_constats(p: argparse.ArgumentParser) -> None:
    p.add_argument('--file', default='fort.7', help='Path to fort.7 file')
    p.add_argument('--frames', default=None, help="Frame selection.")
    p.add_argument('--min-bo', type=float, default=0.0, dest='min_bo', help='BO threshold before stats.')
    p.add_argument('--directed', action='store_true', help='Treat bonds as directed.')
    p.add_argument('--how', choices=['mean', 'max', 'count'], default='mean', help='Statistic across frames.')
    p.add_argument('--export', default=None, help='Export stats CSV (src, dst, value).')
    p.set_defaults(_run=_task_constats)

def _wire_bond_ts(p: argparse.ArgumentParser) -> None:
    p.add_argument('--file', default='fort.7', help='Path to fort.7 file')
    p.add_argument('--frames', default=None, help="Frame selection.")
    p.add_argument('--directed', action='store_true', help='Do not merge A–B with B–A.')
    p.add_argument('--bo-threshold', type=float, default=0.0, dest='bo_threshold',
                   help='Zero out BO values below this (noise floor).')
    p.add_argument('--wide', action='store_true', help='Return wide matrix (frames × bonds).')
    p.add_argument('--xaxis', default='iter', choices=['iter', 'frame', 'time'],
                   help='X-axis for single-bond quick plot.')
    p.add_argument('--control', default='control', help='Control file (for --xaxis time).')
    p.add_argument('--src', type=int, help='Source atom for quick plot.')
    p.add_argument('--dst', type=int, help='Destination atom for quick plot.')
    p.add_argument('--plot', action='store_true', help='Show quick plot for one bond.')
    p.add_argument('--save', default=None, help='Save quick plot image.')
    p.add_argument('--export', default=None, help='Export CSV.')
    p.set_defaults(_run=_task_bond_ts)

def _wire_bond_events(p: argparse.ArgumentParser) -> None:
    p.add_argument('--file', default='fort.7', help='Path to fort.7 file')
    p.add_argument('--frames', default=None, help="Frame selection.")
    p.add_argument('--src', type=int, help='Filter to a specific bond: source atom.')
    p.add_argument('--dst', type=int, help='Filter to a specific bond: destination atom.')
    p.add_argument('--threshold', type=float, default=0.35, help='Schmitt base threshold.')
    p.add_argument('--hysteresis', type=float, default=0.05, help='Schmitt hysteresis band.')
    p.add_argument('--smooth', choices=['ma', 'ema', 'none'], default='ma', help='Smoothing for event detection.')
    p.add_argument('--window', type=int, default=7, help='Window for MA/EMA.')
    p.add_argument('--ema-alpha', type=float, default=None, dest='ema_alpha', help='Alpha for EMA (optional).')
    p.add_argument('--min-run', type=int, default=3, dest='min_run', help='Minimum consecutive points for a state.')
    p.add_argument('--xaxis', default='iter', choices=['iter', 'frame'], help='Event x-axis (internal).')
    p.add_argument('--directed', action='store_true', help='Do not merge A–B with B–A.')
    p.add_argument('--export', default=None, help='Export detected events CSV.')
    p.add_argument('--save', default=None,
                   help='Additionally save a debug overlay figure for the single bond (requires --src --dst).')
    p.set_defaults(_run=_task_bond_events)


def register_tasks(subparsers: argparse._SubParsersAction) -> None:
    """
    Register subcommands under the 'fort7' namespace.
    """
    # GET
    p_get = subparsers.add_parser('get', help='Get a feature (atom or summary) and optionally plot/save/export. || '
                                              'reaxkit fort7 get --feature charge --atom 1 --plot || '
                                              'reaxkit fort7 get --feature total_charge --save tc.png || '
                                              'reaxkit fort7 get --feature "^BO\\d+$" --atom 12 --regex --export bo_a12.csv')
    _wire_get(p_get)

    # EDGES
    p_edges = subparsers.add_parser('edges', help='Build a tidy connection (edge) list from fort.7. || '
                                                  'reaxkit fort7 edges --frames 0:1000:10 --min-bo 0.4 --export edges.csv')
    _wire_edges(p_edges)

    # CONSTATs
    p_constats = subparsers.add_parser('constats', help='Aggregate connection stats across frames (mean/max/count). || '
                                                        'reaxkit fort7 constats --frames 0:1000 --how count --export stats.csv')
    _wire_constats(p_constats)

    # BOND-TS
    p_bts = subparsers.add_parser('bond-ts', help='Time series of bond orders; optional single-bond quick plot. || '
                                                  'reaxkit fort7 bond-ts --frames 0:500 --src 24 --dst 34 --save bo_between_24_34.png')
    _wire_bond_ts(p_bts)

    # BOND-EVENTS
    p_bev = subparsers.add_parser('bond-events', help='Detect bond formation/breakage events with hysteresis. || '
                                                      'reaxkit fort7 bond-events --src 24 --dst 11 --threshold 0.38 --hysteresis 0.10 --smooth ema --window 7 --ema-alpha 0.35 --min-run 4 --xaxis iter --export bo_11_24_events.csv --save bo_11_24_overlay.png')
    _wire_bond_events(p_bev)
