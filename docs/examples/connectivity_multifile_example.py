"""
Example: connectivity + charge multi-file analysis

This script demonstrates how to:
1) Load xmolout + fort.7 using handlers and ConnectivityData
2) Extract per-atom charge and sum_bond_orders features
3) Build connectivity edge lists (bond network) from ConnectivityData
4) Detect bond events from bond-order trajectories
5) Classify coordination status over frames
6) Export results to CSV (docs/examples-friendly)
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from reaxkit.analysis.connectivity import (
    BondEventsRequest,
    BondEventsTask,
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

OUTDIR = Path("reaxkit_outputs/examples/connectivity_multifile")
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
    bo_frames = conn_data.bond_orders
    if bo_frames is None:
        bo_count = 0
    elif hasattr(bo_frames, "shape"):
        bo_count = int(bo_frames.shape[0])
    else:
        bo_count = len(bo_frames)
    print(f"Loaded ConnectivityData BO : {bo_count}")

    _print_head(xdf, "xmolout summary dataframe")
    _print_head(fdf, "fort.7 summary dataframe")

    charges = ChargeTableTask().run(
        charge_data,
        ChargeTableRequest(frames=list(FRAMES_SAMPLE)),
    ).table
    _print_head(charges, "fort.7 partial charges (sampled frames)")
    charges.to_csv(OUTDIR / "partial_charges_sample.csv", index=False)

    if conn_data.sum_bond_orders is not None:
        sum_bo_arr = conn_data.sum_bond_orders
        atom_ids = list(conn_data.atom_ids or [])
        iterations = list(conn_data.iterations) if conn_data.iterations is not None else list(range(sum_bo_arr.shape[0]))
        rows = []
        frame_indices = [i for i in FRAMES_SAMPLE if 0 <= i < int(sum_bo_arr.shape[0])]
        for fi in frame_indices:
            for ai, atom_id in enumerate(atom_ids):
                rows.append(
                    {
                        "frame_idx": int(fi),
                        "iter": int(iterations[fi]),
                        "atom_id": int(atom_id),
                        "sum_bond_order": float(sum_bo_arr[fi, ai]),
                    }
                )
        sum_bos = pd.DataFrame(rows)
        _print_head(sum_bos, "sum_bond_orders (sampled frames)")
        sum_bos.to_csv(OUTDIR / "sum_bond_orders_sample.csv", index=False)

    wanted_summary_cols = [c for c in ["iter", "num_of_bonds", "total_BO", "total_charge"] if c in fdf.columns]
    if wanted_summary_cols:
        valid_rows = [i for i in FRAMES_SAMPLE if 0 <= i < len(fdf)]
        summary_feats = fdf.iloc[valid_rows][wanted_summary_cols].copy()
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

    if not stats.empty:
        src = int(stats.iloc[0]["source"])
        dst = int(stats.iloc[0]["destination"])
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
