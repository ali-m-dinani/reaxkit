"""fort.7 based file for analyzing atomic connectivities."""
from __future__ import annotations
from typing import Iterable, List, Optional, Sequence, Union, Literal, Tuple
import pandas as pd
import numpy as np

from reaxkit.utils.frame_utils import resolve_indices
from reaxkit.utils.moving_average import moving_average, exponential_moving_average
from reaxkit.utils.signal_ops import schmitt_hysteresis, clean_flicker
from reaxkit.analysis.plotter import single_plot

Indexish = Union[int, Iterable[int], None]

# ---- moved from fort7_analyzer: connectivity ----
def connection_list(
    handler,
    frames: Indexish = None,
    iterations: Indexish = None,
    min_bo: float = 0.0,
    undirected: bool = True,
    aggregate: Literal["max", "mean"] = "max",
    include_self: bool = False,
) -> pd.DataFrame:
    """makes a list showing which two atoms (source and dest) and connected by which bond order.
    Build a tidy connection (edge) list with bond orders as weights.

    Each selected frame contributes rows with:
      frame_idx, iter, src, dst, bo, j
    where:
      - src = atom_num of the row atom
      - dst = atom_cnn{j} (j = 1..num_bonds for that frame)
      - bo  = BO{j} (bond order paired to atom_cnn{j})

    Args
    ----
    min_bo       : keep only edges with bo >= min_bo
    undirected   : if True, canonicalize edges (src<=dst) and aggregate duplicates
        **Use undirected=True (default) for typical chemistry/bond analysis — clean, one row per bond.
    aggregate    : "max" or "mean" bond order when merging duplicates
    include_self : keep self-edges if True (usually False)

    Returns
    -------
    pd.DataFrame with columns: ["frame_idx", "iter", "src", "dst", "bo", "j"]
    (If undirected=True duplicates are collapsed; otherwise both directions remain.)
    """
    sim_df = handler.dataframe()
    idx_list = resolve_indices(handler, frames=frames, iterations=iterations)

    all_edges: List[pd.DataFrame] = []

    for fi in idx_list:
        atoms = handler._frames[fi]
        iter = int(sim_df.iloc[fi]["iter"])
        # Find how many cnn/BO columns exist for this frame
        nb = int(sim_df.iloc[fi]["num_of_bonds"])
        cnn_cols = [f"atom_cnn{j}" for j in range(1, nb + 1)]
        bo_cols  = [f"BO{j}"       for j in range(1, nb + 1)]

        # Sanity: skip if columns are missing
        missing = [c for c in cnn_cols + bo_cols if c not in atoms.columns]
        if missing:
            # Skip silently (or raise) — here we skip this frame
            continue

        # Build edge blocks for each neighbor slot j
        blocks: List[pd.DataFrame] = []
        src_series = atoms["atom_num"].astype(int)

        for j, (cnn_c, bo_c) in enumerate(zip(cnn_cols, bo_cols), start=1):
            dst_series = atoms[cnn_c].astype(int)
            bo_series  = atoms[bo_c].astype(float)

            dfj = pd.DataFrame(
                {
                    "src": src_series.values,
                    "dst": dst_series.values,
                    "bo":  bo_series.values,
                    "j":   j,
                }
            )
            blocks.append(dfj)

        edges = pd.concat(blocks, ignore_index=True)

        # Filter invalid/empty connections:
        # - Some datasets mark no-connection with 0 or negative dst or bo<=0
        mask = edges["dst"] > 0
        if not include_self:
            mask &= edges["dst"] != edges["src"]
        if min_bo is not None:
            mask &= edges["bo"] >= float(min_bo)
        edges = edges.loc[mask].copy()

        # Attach frame metadata
        edges.insert(0, "iter", iter)
        edges.insert(0, "frame_idx", fi)

        # Canonicalize for undirected graphs and collapse duplicates
        if undirected:
            # Ensure src <= dst
            src_min = edges[["src", "dst"]].min(axis=1)
            dst_max = edges[["src", "dst"]].max(axis=1)
            edges["src"], edges["dst"] = src_min, dst_max

            # Combine duplicates (the same bond appears from both atoms)
            by = ["frame_idx", "iter", "src", "dst"]
            if aggregate == "mean":
                agg = edges.groupby(by, as_index=False)["bo"].mean()
            else:
                agg = edges.groupby(by, as_index=False)["bo"].max()

            # Keep one representative j (optional; not meaningful when aggregated)
            agg["j"] = -1  # indicates aggregated
            edges = agg

        all_edges.append(edges)

    if not all_edges:
        return pd.DataFrame(columns=["frame_idx", "iter", "src", "dst", "bo", "j"])

    # Concatenate all frames
    out = pd.concat(all_edges, ignore_index=True)
    # Sort for stability
    out = out.sort_values(["frame_idx", "src", "dst", "j"], kind="stable").reset_index(drop=True)
    return out

def connection_table(
    handler,
    frame: int,
    min_bo: float = 0.0,
    undirected: bool = True,
    fill_value: float = 0.0,
) -> pd.DataFrame:
    """for a single frame, converts connectivity list to an adjacency (or connectivity) table.
    Build a dense adjacency-like table (wide format) for a single frame.
    WARNING: memory-heavy for large systems. Prefer connection_list for most tasks.

    Returns
    -------
    DataFrame with index=src_atom_num, columns=dst_atom_num, values=bond order (>=min_bo).
    """
    edges = connection_list(
        handler,
        frames=[frame],
        iterations=None,
        min_bo=min_bo,
        undirected=undirected,
    )
    if edges.empty:
        return pd.DataFrame()

    # Use pivot to form a (possibly sparse) adjacency matrix
    tbl = edges.pivot_table(
        index="src",
        columns="dst",
        values="bo",
        aggfunc="max",
        fill_value=fill_value,
    )
    # Make it a regular DataFrame with sorted axes
    tbl = tbl.sort_index(axis=0).sort_index(axis=1)
    return tbl

def connection_stats_over_frames(
    handler,
    frames: Indexish = None,
    iterations: Indexish = None,
    min_bo: float = 0.0,
    undirected: bool = True,
    how: Literal["mean", "max", "count"] = "mean",
) -> pd.DataFrame:
    """for each pair of source-dest, it calculates a metrics across selected frames.
    Aggregate edges across selected frames, returning a per-pair statistic.

    Returns
    -------
    DataFrame with columns: ["src","dst","value"]
      - value = mean/max/edge-count across frames (after min_bo filtering)
    """
    edges = connection_list(
        handler,
        frames=frames,
        iterations=iterations,
        min_bo=min_bo,
        undirected=undirected,
    )
    if edges.empty:
        return pd.DataFrame(columns=["src", "dst", "value"])

    by = ["src", "dst"]
    if how == "count":
        out = edges.groupby(by, as_index=False).size().rename(columns={"size": "value"})
    elif how == "max":
        out = edges.groupby(by, as_index=False)["bo"].max().rename(columns={"bo": "value"})
    else:  # mean
        out = edges.groupby(by, as_index=False)["bo"].mean().rename(columns={"bo": "value"})
    return out.sort_values(["src", "dst"], kind="stable").reset_index(drop=True)

# ---- moved helper (keep here since it’s bond-table specific) ----
def _pivot_to_tidy_bo(pivot: pd.DataFrame) -> pd.DataFrame:
    """
    Convert a (frame_idx, iter) x (src,dst) pivot to tidy:
    columns: frame_idx, iter, src, dst, bo
    Works across pandas versions (Series/DataFrame stack behavior).
    """
    # Try the modern stack; fall back to legacy
    try:
        stacked = pivot.stack(future_stack=True)
    except TypeError:
        stacked = pivot.stack()

    # Case 1: Series → easy
    if isinstance(stacked, pd.Series):
        tidy = stacked.rename("bo").reset_index()
    else:
        # Case 2: DataFrame (some pandas builds) → if 1 col, rename; else manual
        if stacked.shape[1] == 1:
            value_col = stacked.columns[0]
            tidy = stacked.rename(columns={value_col: "bo"}).reset_index()
        else:
            # Manual, bulletproof & fast enough for moderate sizes
            rows = []
            for (fi, it), row in pivot.iterrows():
                for (src, dst), bo in row.items():
                    rows.append([fi, it, src, dst, bo])
            tidy = pd.DataFrame(rows, columns=["frame_idx", "iter", "src", "dst", "bo"])
            return tidy

    # Normalize column names for stacked levels
    # After reset_index we expect: frame_idx, iter, <src_level>, <dst_level>, bo
    want = {"src", "dst"}
    got = set(tidy.columns)
    if not want.issubset(got):
        mi_names = pivot.columns.names or ["src", "dst"]
        rename_map = {}
        if mi_names[0] in tidy.columns:
            rename_map[mi_names[0]] = "src"
        if mi_names[1] in tidy.columns:
            rename_map[mi_names[1]] = "dst"
        tidy = tidy.rename(columns=rename_map)

        # Final fallback for unnamed levels (e.g., level_2/level_3)
        if "src" not in tidy.columns or "dst" not in tidy.columns:
            for c in list(tidy.columns):
                if c.startswith("level_") and "src" not in tidy.columns:
                    tidy = tidy.rename(columns={c: "src"})
                elif c.startswith("level_") and "dst" not in tidy.columns:
                    tidy = tidy.rename(columns={c: "dst"})

    return tidy.sort_values(["frame_idx", "src", "dst"], kind="stable").reset_index(drop=True)


def bond_timeseries(
    handler,
    frames: Indexish = None,
    iterations: Indexish = None,
    undirected: bool = True,
    bo_threshold: float = 0.0,
    as_wide: bool = False,
) -> pd.DataFrame:
    """tracks a specific bond across selected frames in terms of its bond order.
    Track each unique bond's bond order (BO) across selected frames.
    If a bond doesn't appear in a frame (broken/not reported), its BO is set to 0.

    Args
    ----
    handler     : Fort7Handler with parsed data
    frames      : frame indices to include (int, iterable, slice-string already resolved upstream)
    iterations  : iter numbers to include (optional)
    undirected  : treat bonds as undirected (merge A–B and B–A)
    bo_threshold: values below this are set to 0 (e.g., noise floor like 0.1)
    as_wide     : if True, return wide matrix (index=[frame_idx, iter], columns="src-dst");
                  if False (default), return tidy long-form with columns:
                  [frame_idx, iter, src, dst, bo]

    Returns
    -------
    pd.DataFrame
      - tidy (default): one row per (frame, bond); missing bonds filled with bo=0
      - wide (as_wide=True): matrix with frames × bonds, bo filled with 0
    """
    # 1) Resolve frame indices in order
    idx_list = resolve_indices(handler, frames=frames, iterations=iterations)
    if not idx_list:
        return pd.DataFrame(columns=["frame_idx", "iter", "src", "dst", "bo"] if not as_wide else [])

    # 2) Build edge list for selected frames (keep all reported edges; we'll threshold later)
    edges = connection_list(
        handler,
        frames=idx_list,
        iterations=None,          # idx_list already chosen
        min_bo=0.0,               # keep every reported bond; we'll zero later via bo_threshold
        undirected=undirected,
        include_self=False,
    )
    # Ensure unique rows (max BO if duplicated)
    if not edges.empty:
        edges = (
            edges.groupby(["frame_idx", "iter", "src", "dst"], as_index=False)["bo"]
                 .max()
        )

    # 3) Build the full frame index (even if a frame reported no edges)
    sim_df = handler.dataframe()
    frame_meta = pd.DataFrame({
        "frame_idx": idx_list,
        "iter": [int(sim_df.iloc[i]["iter"]) for i in idx_list],
    })
    frame_meta = frame_meta.drop_duplicates().sort_values(["frame_idx"]).reset_index(drop=True)

    # 4) If no edges at all in these frames, return zeros-only structure
    if edges.empty:
        if as_wide:
            # no bonds → empty columns; just return index of frames
            wide = frame_meta.set_index(["frame_idx", "iter"])
            return wide
        else:
            # no bonds → empty tidy table
            return pd.DataFrame(columns=["frame_idx", "iter", "src", "dst", "bo"])

    # 5) Create a pivot (frames × bonds), then reindex to include *all* bonds and *all* frames
    pivot = edges.pivot_table(
        index=["frame_idx", "iter"],
        columns=["src", "dst"],
        values="bo",
        aggfunc="max",
    )

    # All bonds observed in any selected frame
    all_bonds: List[Tuple[int, int]] = sorted(set(zip(edges["src"], edges["dst"])))
    # Reindex rows to ensure all frames present in the matrix (even those w/o edges)
    pivot = pivot.reindex(
        index=pd.MultiIndex.from_frame(frame_meta[["frame_idx", "iter"]]),
        fill_value=0.0
    )
    # Reindex columns to ensure all bonds present
    pivot = pivot.reindex(
        columns=pd.MultiIndex.from_tuples(all_bonds, names=["src", "dst"]),
        fill_value=0.0
    )

    # Force any remaining gaps to zero (handles rare pandas edge cases)
    pivot = pivot.fillna(0.0)

    # 6) Threshold-to-zero for small BO (noise floor)
    if bo_threshold > 0.0:
        pivot = pivot.mask(pivot < float(bo_threshold), 0.0)

    # 7) Return in desired shape
    if as_wide:
        pivot.columns = [f"{s}-{d}" for (s, d) in pivot.columns.to_list()]
        pivot = pivot.sort_index(level=[0, 1])
        return pivot

    # Robust tidy conversion
    tidy = _pivot_to_tidy_bo(pivot)
    return tidy

# ---- events: inline smoothing, use utils for hysteresis/flicker ----
def bond_events(
    handler,
    frames: Indexish = None,
    iterations: Indexish = None,
    *,
    src: Optional[int] = None,
    dst: Optional[int] = None,
    threshold: float = 0.35,
    hysteresis: float = 0.05,
    smooth: Optional[Literal["ma","ema"]] = "ma",
    window: int = 7,
    ema_alpha: Optional[float] = None,
    min_run: int = 3,
    xaxis: Literal["iter","frame"] = "iter",
    undirected: bool = True,
) -> pd.DataFrame:
    """a function which determines when a bond is broken or formed.
    Detect bond **formation** and **breakage** events from a bond-order time series,
    using optional smoothing and Schmitt-trigger hysteresis to avoid flicker.

    The function first builds a long-form bond-order time series via
    `bond_timeseries(..., bo_threshold=0.0, as_wide=False)`, then for each (src,dst)
    pair it:
      1) sorts by frame index,
      2) optionally smooths the bond order (moving average or EMA),
      3) applies a Schmitt hysteresis (`threshold` ± `hysteresis`) to create a stable
         boolean "bonded" state,
      4) de-bounces short runs (`min_run`) to clean one-frame flicker,
      5) records the frames/iterations where the state rises (formation) or falls
         (breakage).

    Notes
    -----
    - Smoothing:
        * "ma" uses a centered moving average with `min_periods=1`, so edges are kept.
        * "ema" uses `adjust=False` (recursive form) for streaming-friendly behavior;
          set `ema_alpha` for explicit smoothing strength.
    - Hysteresis: `schmitt_hysteresis` converts the smoothed bond order into a stable
      boolean state by requiring separate enter/exit thresholds, reducing spurious
      toggles near `threshold`.
    - Debounce: `clean_flicker(..., min_run)` removes state segments shorter than
      `min_run`, further suppressing noise-induced events.
    - Pair filtering: if `src` and `dst` are provided, only that pair is analyzed.
      With `undirected=True`, the pair is internally ordered (min, max).
    """
    ts = bond_timeseries(
        handler,
        frames=frames,
        iterations=iterations,
        undirected=undirected,
        bo_threshold=0.0,
        as_wide=False,
    )
    if ts.empty:
        return pd.DataFrame(columns=["src","dst","event","frame_idx","iter","x_axis","bo_at_event","threshold","hysteresis"])

    if src is not None and dst is not None:
        a, b = (src, dst) if (not undirected or src <= dst) else (dst, src)
        ts = ts[(ts["src"] == a) & (ts["dst"] == b)].copy()
        if ts.empty:
            return pd.DataFrame(columns=["src","dst","event","frame_idx","iter","x_axis","bo_at_event","threshold","hysteresis"])

    groups = ts.groupby(["src","dst"], sort=False)
    xcol = "iter" if xaxis == "iter" else "frame_idx"
    out_rows: List[pd.DataFrame] = []

    for (a, b), g in groups:
        g = g.sort_values(["frame_idx"]).reset_index(drop=True)
        x  = g[xcol].to_numpy()
        bo = g["bo"].to_numpy(dtype=float)

        # --- inline smoothing (replaces _smooth_series) ---
        if smooth is None:
            bo_s = bo
        elif smooth == "ema":
            bo_s = exponential_moving_average(pd.Series(bo), window=window, alpha=ema_alpha, adjust=False).to_numpy()
        else:  # "ma"
            bo_s = moving_average(pd.Series(bo), window=window, center=True, min_periods=1).to_numpy()

        # --- hysteresis & flicker clean via utils ---
        st = schmitt_hysteresis(bo_s, th=threshold, hys=hysteresis)
        st = clean_flicker(st, min_run=min_run)

        prev = np.r_[st[0], st[:-1]]
        rising  = (~prev) & st
        falling = prev & (~st)
        mask = rising | falling
        if not mask.any():
            continue

        ev = pd.DataFrame({
            xcol: x[mask],
            "event": np.where(rising[mask], "formation", "breakage"),
            "bo_at_event": bo_s[mask],
        })
        ev.insert(0, "dst", b)
        ev.insert(0, "src", a)

        merge_cols = ["frame_idx", "iter"]
        if xcol not in merge_cols:
            merge_cols.append(xcol)
        meta = g[merge_cols].drop_duplicates(subset=[xcol])
        ev = ev.merge(meta, on=xcol, how="left")
        ev["x_axis"] = ev[xcol]
        ev["threshold"] = float(threshold)
        ev["hysteresis"] = float(hysteresis)

        out_rows.append(ev[["src","dst","event","frame_idx","iter","x_axis","bo_at_event","threshold","hysteresis"]])

    if not out_rows:
        return pd.DataFrame(columns=["src","dst","event","frame_idx","iter","x_axis","bo_at_event","threshold","hysteresis"])

    out = pd.concat(out_rows, ignore_index=True)
    return out.sort_values(["src","dst","x_axis","event"], kind="stable").reset_index(drop=True)

def bond_events_single(handler, src: int, dst: int, **kwargs) -> pd.DataFrame:
    """simply doing bond event analysis for a single frame.
    """
    return bond_events(handler, src=src, dst=dst, **kwargs)

def debug_bond_trace_overlay(
    handler,
    src: int,
    dst: int,
    *,
    smooth: str = "ema",         # "ema" or "ma"
    window: int = 8,
    hysteresis: float = 0.05,
    threshold: float = 0.10,
    min_run: int = 0,            # >=2 to match bond_events behavior
    xaxis: str = "iter",    # "iter" or "frame"
    save: str | None = None,     # file path OR directory; None -> show
):
    """One plot with: raw BO, smoothed BO, ON/OFF bands, and event markers. Uses analysis.plotter.single_plot (multi-series).
    Plot a single diagnostic overlay for one bond (src–dst): raw bond order (BO),
    smoothed BO, Schmitt-trigger ON/OFF bands, and event markers (formation/breakage).

    This is a visual companion to `bond_events`, useful for tuning `threshold`,
    `hysteresis`, smoothing, and `min_run`. It:
      1) extracts the (src,dst) BO time series via `bond_timeseries(..., as_wide=False)`,
      2) normalizes the pair order to (min(src,dst), max(src,dst)),
      3) smooths BO with moving average ("ma") or exponential moving average ("ema"),
      4) computes a hysteretic bonded state on the *smoothed* signal,
      5) optionally de-bounces short state runs with `clean_flicker(min_run)`,
      6) marks rising edges as “formation” (▲) and falling edges as “breakage” (▼),
      7) draws horizontal ON/OFF lines at `threshold ± hysteresis/2`,
      8) renders via `analysis.plotter.single_plot`.
    """

    # --- data ---
    a, b = (src, dst) if src <= dst else (dst, src)
    ts = bond_timeseries(handler, as_wide=False)
    g = ts[(ts["src"] == a) & (ts["dst"] == b)].sort_values("iter")
    if g.empty:
        print(f"No data for bond {a}-{b}.")
        return

    x = g["iter"].to_numpy() if xaxis == "iter" else g["frame_idx"].to_numpy()
    y = g["bo"].to_numpy(dtype=float)

    # --- smoothing ---
    y_s = (exponential_moving_average(y, window=window)
           if smooth == "ema" else moving_average(y, window=window)).to_numpy()

    th, hys = float(threshold), float(hysteresis)
    th_on, th_off = th + hys/2.0, th - hys/2.0

    # --- hysteresis state & events on *smoothed* series ---
    def _schmitt(sig: np.ndarray, base_th: float, band: float) -> np.ndarray:
        on, off = base_th + band/2.0, base_th - band/2.0
        st = np.zeros_like(sig, dtype=bool)
        cur = sig[0] >= on
        for i, v in enumerate(sig):
            if not cur and v >= on:
                cur = True
            elif cur and v <= off:
                cur = False
            st[i] = cur
        return st

    st = _schmitt(y_s, th, hys)
    if min_run and min_run > 1:
        st = clean_flicker(st, min_run=min_run)

    prev = np.r_[st[0], st[:-1]]
    rising  = (~prev) & st   # formation
    falling = prev & (~st)   # breakage
    n_form, n_break = int(rising.sum()), int(falling.sum())

    # --- build series for single_plot (multi-series) ---
    series = [
        {'x': x, 'y': y,   'label': 'raw',            'marker': '.', 'linewidth': 0,   'markersize': 3, 'alpha': 0.75},
        {'x': x, 'y': y_s, 'label': f'{smooth} (w={window})', 'marker': None, 'linewidth': 1.6, 'alpha': 1.0},
    ]
    # event markers (as point-only series)
    if n_form:
        series.append({'x': x[rising],  'y': y_s[rising],  'label': f'formation ×{n_form}', 'marker': '^', 'linewidth': 0, 'markersize': 7, 'alpha': 1.0})
    if n_break:
        series.append({'x': x[falling], 'y': y_s[falling], 'label': f'breakage ×{n_break}',  'marker': 'v', 'linewidth': 0, 'markersize': 7, 'alpha': 1.0})

    # horizontal ON/OFF bands
    hlines = [
        {'y': th_on,  'label': f'ON ≥ {th_on:.3f}',  'linestyle': '--', 'linewidth': 1},
        {'y': th_off, 'label': f'OFF ≤ {th_off:.3f}', 'linestyle': '--', 'linewidth': 1},
    ]

    title = (f"Bond {a}-{b} | th={th:.3f}, hyst={hys:.3f} | "
             f"min_run={min_run} | events: +{n_form}/-{n_break}")

    single_plot(
        series=series,
        hlines=hlines,
        title=title,
        xlabel=("iter" if xaxis == "iter" else "frame"),
        ylabel="BO",
        save=save,
        legend=True,
        figsize=(9.0, 3.8),
    )
