"""
Example: xmolout + fort.7 (multi-file analysis)

This script demonstrates how to:
1) Load xmolout + fort.7 using handlers
2) Extract fort.7 per-atom features (partial charges, sum_BOs)
3) Build connectivity edge lists (bond network) from fort.7
4) Track bond-order time series and detect bond events
5) Classify coordination status over frames (needs xmolout atom types + fort.7 sum_BOs)
6) Export results to CSV (docs/examples-friendly)

"""

from __future__ import annotations

from pathlib import Path
import pandas as pd


# -----------------------------
# Robust imports (path-safe)
# -----------------------------
from reaxkit.io.handlers.fort7_handler import Fort7Handler
from reaxkit.io.handlers.xmolout_handler import XmoloutHandler

# Fort7 analyzers
try:
    # Your newer layout (common)
    from reaxkit.analysis.per_file.fort7_analyzer import (
        get_fort7_data_per_atom,
        get_fort7_data_summaries,
        get_partial_charges_conv_fnc,
        get_all_atoms_cnn_conv_fnc,
        per_atom_coordination_status_over_frames,
    )
except Exception:
    # Older / alternative layout (fallback)
    from reaxkit.analysis.per_file.fort7_analyzer import (  # type: ignore
        get_features_atom,
        get_fort7_data_summaries,
        get_partial_charges_conv_fnc,
        get_all_atoms_cnn_conv_fnc,
        per_atom_coordination_status_over_frames,
    )

# Connectivity analyzers
try:
    from reaxkit.analysis.composed.connectivity_analyzer import (
        connection_list,
        connection_table,
        connection_stats_over_frames,
        bond_timeseries,
        bond_events,
    )
except Exception:
    from reaxkit.analysis.composed.connectivity_analyzer import (  # type: ignore
        connection_list,
        connection_table,
        connection_stats_over_frames,
        bond_timeseries,
        bond_events,
    )


# -----------------------------
# Config
# -----------------------------
HERE = Path(__file__).resolve().parent
DATA = HERE / "data"

XMOL_PATH = DATA / "small_xmolout"
FORT7_PATH = DATA / "small_fort.7"

# Keep examples fast + small
FRAMES_SAMPLE = range(0, 50, 5)  # frames: 0,5,10,...45
MIN_BO = 0.30

OUTDIR = Path("reaxkit_outputs/examples/xmolout_fort7")
OUTDIR.mkdir(parents=True, exist_ok=True)


def _print_head(df: pd.DataFrame, name: str, n: int = 5) -> None:
    print(f"\n=== {name} (head {n}) ===")
    if df.empty:
        print("(empty)")
    else:
        print(df.head(n).to_string(index=False))


def main() -> None:
    # -----------------------------
    # 1) Load handlers
    # -----------------------------
    if not XMOL_PATH.exists():
        raise FileNotFoundError(f"Missing xmolout sample file: {XMOL_PATH}")
    if not FORT7_PATH.exists():
        raise FileNotFoundError(f"Missing fort.7 sample file: {FORT7_PATH}")

    xh = XmoloutHandler(str(XMOL_PATH))
    f7 = Fort7Handler(str(FORT7_PATH))

    xdf = xh.dataframe()
    fdf = f7.dataframe()

    print(f"Loaded xmolout frames: {len(xdf)}")
    print(f"Loaded fort.7 frames : {len(fdf)}")

    _print_head(xdf, "xmolout summary dataframe")
    _print_head(fdf, "fort.7 summary dataframe")

    # -----------------------------
    # 2) Fort.7 features (atom + summary)
    # -----------------------------
    # Partial charges over a frame sample
    charges = get_partial_charges_conv_fnc(f7, frames=FRAMES_SAMPLE)
    _print_head(charges, "fort.7 partial charges (sampled frames)")
    charges.to_csv(OUTDIR / "partial_charges_sample.csv", index=False)

    # sum_BOs is a very useful per-atom scalar for coordination-type analysis
    sum_bos = get_fort7_data_per_atom(f7, columns="sum_BOs", frames=FRAMES_SAMPLE)
    _print_head(sum_bos, "fort.7 sum_BOs (sampled frames)")
    sum_bos.to_csv(OUTDIR / "sum_BOs_sample.csv", index=False)

    # Summary-level features (example)
    # (Only keep those that exist in your file; adjust names if needed.)
    wanted_summary_cols = [c for c in ["iter", "num_bonds", "total_BO", "total_charge"] if c in fdf.columns]
    if wanted_summary_cols:
        summary_feats = get_fort7_data_summaries(f7, wanted_summary_cols, frames=FRAMES_SAMPLE)
        _print_head(summary_feats, "fort.7 summary features (sampled frames)")
        summary_feats.to_csv(OUTDIR / "fort7_summary_features_sample.csv", index=False)

    # -----------------------------
    # 3) Connectivity edge list (bond network)
    # -----------------------------
    edges = connection_list(f7, frames=FRAMES_SAMPLE, min_bo=MIN_BO, undirected=True)
    _print_head(edges, f"connectivity edge list (min_bo={MIN_BO})")
    edges.to_csv(OUTDIR / f"edges_minbo_{MIN_BO:.2f}.csv", index=False)

    # Optional: adjacency matrix for one frame (can be large)
    # Only do for frame 0 and keep min_bo high.
    adj0 = connection_table(f7, frame=0, min_bo=MIN_BO, undirected=True)
    if not adj0.empty:
        print(f"\nAdjacency table frame 0 shape: {adj0.shape}")
        adj0.to_csv(OUTDIR / f"adjacency_frame0_minbo_{MIN_BO:.2f}.csv")

    # Aggregated bond stats over frames
    stats = connection_stats_over_frames(f7, frames=FRAMES_SAMPLE, min_bo=MIN_BO, how="count")
    _print_head(stats, "bond stats over frames (count)")
    stats.to_csv(OUTDIR / f"bond_stats_count_minbo_{MIN_BO:.2f}.csv", index=False)

    # -----------------------------
    # 4) Bond-order time series + events
    # -----------------------------
    # Tidy bond-order time series (long form)
    ts = bond_timeseries(f7, frames=FRAMES_SAMPLE, undirected=True, bo_threshold=0.0, as_wide=False)
    _print_head(ts, "bond timeseries (tidy)")
    ts.to_csv(OUTDIR / "bond_timeseries_tidy.csv", index=False)

    # Pick a bond to demonstrate event detection:
    # Choose the most frequent bond from stats if available
    if not stats.empty:
        src = int(stats.iloc[0]["src"])
        dst = int(stats.iloc[0]["dst"])
    else:
        # fallback
        src, dst = 1, 2

    ev = bond_events(
        f7,
        frames=FRAMES_SAMPLE,
        src=src,
        dst=dst,
        threshold=0.10,
        hysteresis=0.05,
        smooth="ma",
        window=7,
        min_run=3,
        xaxis="iter",
        undirected=True,
    )
    _print_head(ev, f"bond events for {src}-{dst}")
    ev.to_csv(OUTDIR / f"bond_events_{src}_{dst}.csv", index=False)

    # -----------------------------
    # 5) Coordination status (fort.7 + xmolout)
    # -----------------------------
    # Requires:
    # - fort.7 sum_BOs (per atom)
    # - xmolout atom types (per atom)
    #
    # Provide valences per atom type symbol used in your xmolout (adjust as needed).
    # Tip: inspect xmolout types via xh.frame(0)["atom_types"] if unsure.
    valences = {
        "H": 1,
        "O": 2,
        "N": 3,
        "C": 4,
        "Al": 3,
        "Mg": 2,
        "Zn": 2,
    }

    coord = per_atom_coordination_status_over_frames(
        f7, xh,
        valences=valences,
        threshold=0.25,
        frames=FRAMES_SAMPLE,
        require_all_valences=False,  # set True once your valence map is complete
    )
    _print_head(coord, "coordination status (sampled frames)")
    coord.to_csv(OUTDIR / "coordination_status_sample.csv", index=False)

    print(f"\n✅ Done. Results written to:\n  {OUTDIR.resolve()}")


if __name__ == "__main__":
    main()
