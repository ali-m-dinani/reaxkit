"""
Example: xmolout + fort.7 (multi-file analysis)

This script demonstrates how to:
1) Load xmolout + fort.7 using handlers and ConnectivityData
2) Extract fort.7 per-atom features (partial charges, sum_BOs)
3) Build connectivity edge lists (bond network) from ConnectivityData
4) Track bond-order time series and detect bond events
5) Classify coordination status over frames
6) Export results to CSV (docs/examples-friendly)
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from reaxkit.analysis.connectivity import (
    BondEventsRequest,
    BondEventsTask,
    BondTimeseriesRequest,
    BondTimeseriesTask,
    ConnectionListRequest,
    ConnectionListTask,
    ConnectionStatsRequest,
    ConnectionStatsTask,
    ConnectionTableRequest,
    ConnectionTableTask,
    CoordinationStatusRequest,
    CoordinationStatusTask,
)
from reaxkit.analysis.electrostatics import ChargeTableRequest, ChargeTableTask
from reaxkit.analysis.per_file.fort7_analyzer import (
    get_fort7_data_per_atom,
    get_fort7_data_summaries,
)
from reaxkit.domain.data_models import ChargeData, ConnectivityData
from reaxkit.engine.reaxff.adapter import ReaxFFAdapter
from reaxkit.engine.reaxff.io.fort7_handler import Fort7Handler
from reaxkit.engine.reaxff.io.xmolout_handler import XmoloutHandler


HERE = Path(__file__).resolve().parent
DATA = HERE / "data"

XMOL_PATH = DATA / "small_xmolout"
FORT7_PATH = DATA / "small_fort.7"

FRAMES_SAMPLE = range(0, 50, 5)
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
    if not XMOL_PATH.exists():
        raise FileNotFoundError(f"Missing xmolout sample file: {XMOL_PATH}")
    if not FORT7_PATH.exists():
        raise FileNotFoundError(f"Missing fort.7 sample file: {FORT7_PATH}")

    xh = XmoloutHandler(str(XMOL_PATH))
    f7 = Fort7Handler(str(FORT7_PATH))
    adapter = ReaxFFAdapter()
    conn_data = adapter.load(
        ConnectivityData,
        {"xmolout": str(XMOL_PATH), "fort7": str(FORT7_PATH)},
    )
    charge_data = adapter.load(
        ChargeData,
        {"xmolout": str(XMOL_PATH), "fort7": str(FORT7_PATH)},
    )

    xdf = xh.dataframe()
    fdf = f7.dataframe()

    print(f"Loaded xmolout frames      : {len(xdf)}")
    print(f"Loaded fort.7 frames       : {len(fdf)}")
    print(f"Loaded ConnectivityData BO : {len(conn_data.bond_orders or [])}")

    _print_head(xdf, "xmolout summary dataframe")
    _print_head(fdf, "fort.7 summary dataframe")

    charges = ChargeTableTask().run(
        charge_data,
        ChargeTableRequest(frames=list(FRAMES_SAMPLE)),
    ).table
    _print_head(charges, "fort.7 partial charges (sampled frames)")
    charges.to_csv(OUTDIR / "partial_charges_sample.csv", index=False)

    sum_bos = get_fort7_data_per_atom(f7, columns="sum_BOs", frames=FRAMES_SAMPLE)
    _print_head(sum_bos, "fort.7 sum_BOs (sampled frames)")
    sum_bos.to_csv(OUTDIR / "sum_BOs_sample.csv", index=False)

    wanted_summary_cols = [c for c in ["iter", "num_of_bonds", "total_BO", "total_charge"] if c in fdf.columns]
    if wanted_summary_cols:
        summary_feats = get_fort7_data_summaries(f7, wanted_summary_cols, frames=FRAMES_SAMPLE)
        _print_head(summary_feats, "fort.7 summary features (sampled frames)")
        summary_feats.to_csv(OUTDIR / "fort7_summary_features_sample.csv", index=False)

    edges = ConnectionListTask().run(
        conn_data,
        ConnectionListRequest(frames=list(FRAMES_SAMPLE), min_bo=MIN_BO, undirected=True),
    ).table
    _print_head(edges, f"connectivity edge list (min_bo={MIN_BO})")
    edges.to_csv(OUTDIR / f"edges_minbo_{MIN_BO:.2f}.csv", index=False)

    adj0 = ConnectionTableTask().run(
        conn_data,
        ConnectionTableRequest(frame=0, min_bo=MIN_BO, undirected=True),
    ).table
    if not adj0.empty:
        print(f"\nAdjacency table frame 0 shape: {adj0.shape}")
        adj0.to_csv(OUTDIR / f"adjacency_frame0_minbo_{MIN_BO:.2f}.csv")

    stats = ConnectionStatsTask().run(
        conn_data,
        ConnectionStatsRequest(frames=list(FRAMES_SAMPLE), min_bo=MIN_BO, how="count"),
    ).table
    _print_head(stats, "bond stats over frames (count)")
    stats.to_csv(OUTDIR / f"bond_stats_count_minbo_{MIN_BO:.2f}.csv", index=False)

    ts = BondTimeseriesTask().run(
        conn_data,
        BondTimeseriesRequest(
            frames=list(FRAMES_SAMPLE),
            undirected=True,
            bo_threshold=0.0,
            as_wide=False,
        ),
    ).table
    _print_head(ts, "bond timeseries (tidy)")
    ts.to_csv(OUTDIR / "bond_timeseries_tidy.csv", index=False)

    if not stats.empty:
        src = int(stats.iloc[0]["src"])
        dst = int(stats.iloc[0]["dst"])
    else:
        src, dst = 1, 2

    ev = BondEventsTask().run(
        conn_data,
        BondEventsRequest(
            frames=list(FRAMES_SAMPLE),
            src=src,
            dst=dst,
            threshold=0.10,
            hysteresis=0.05,
            smooth="ma",
            window=7,
            min_run=3,
            xaxis="iter",
            undirected=True,
        ),
    ).table
    _print_head(ev, f"bond events for {src}-{dst}")
    ev.to_csv(OUTDIR / f"bond_events_{src}_{dst}.csv", index=False)

    valences = {
        "H": 1,
        "O": 2,
        "N": 3,
        "C": 4,
        "Al": 3,
        "Mg": 2,
        "Zn": 2,
    }

    coord = CoordinationStatusTask().run(
        conn_data,
        CoordinationStatusRequest(
            valences=valences,
            threshold=0.25,
            frames=list(FRAMES_SAMPLE),
            require_all_valences=False,
        ),
    ).table
    _print_head(coord, "coordination status (sampled frames)")
    coord.to_csv(OUTDIR / "coordination_status_sample.csv", index=False)

    print(f"\nDone. Results written to:\n  {OUTDIR.resolve()}")


if __name__ == "__main__":
    main()
