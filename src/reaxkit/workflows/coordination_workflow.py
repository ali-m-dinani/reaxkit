"""reads fort7 coordination data (i.e., total BO) and then re-writes a xmolout based on coordination of atoms."""
from __future__ import annotations
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, Sequence, Union, Literal, List
import pandas as pd

from reaxkit.io.xmolout_handler import XmoloutHandler
from reaxkit.io.xmolout_generator import write_xmolout_from_frames
from reaxkit.analysis.fort7_analyzer import coordination_status_over_frames

FrameSel = Optional[Union[Sequence[int], range, slice]]

# -----------------------------
# Helpers
# -----------------------------
def _normalize_frames(xh: XmoloutHandler, frames: FrameSel, start: Optional[int], end: Optional[int], every: int) -> list[int]:
    n = xh.n_frames()
    if frames is not None:
        if isinstance(frames, slice):
            idx = list(range(*frames.indices(n)))
        else:
            idx = [int(i) for i in frames if 0 <= int(i) < n]
    else:
        lo = 0 if start is None else max(0, int(start))
        hi = n if end   is None else min(n, int(end))
        idx = list(range(lo, hi))
    every = max(1, int(every))
    return idx[::every]

def _parse_kv_map(s: Optional[str], value_cast=float) -> Dict[str, float]:
    if not s:
        return {}
    out: Dict[str, float] = {}
    for item in s.split(","):
        if not item.strip():
            continue
        if "=" not in item:
            raise ValueError(f"Invalid mapping entry {item!r}; use key=value,comma-separated.")
        k, v = item.split("=", 1)
        out[k.strip()] = value_cast(v.strip())
    return out

def _parse_status_labels(s: Optional[str]) -> Dict[int, str]:
    if not s:
        return {-1: "U", 0: "C", 1: "O"}
    out: Dict[int, str] = {}
    for item in s.split(","):
        if not item.strip():
            continue
        if "=" not in item:
            raise ValueError(f"Invalid label entry {item!r}; use -1=U,0=C,1=O")
        k, v = item.split("=", 1)
        ki = int(k.strip())
        if ki not in (-1, 0, 1):
            raise ValueError("Status keys must be -1, 0, or 1")
        out[ki] = v.strip()
    for k in (-1, 0, 1):
        out.setdefault(k, { -1:"U", 0:"C", 1:"O" }[k])
    return out

def _frame_record_from_handler(xh: XmoloutHandler, i: int) -> Dict[str, Any]:
    fr = xh.frame(i)
    df = xh.dataframe()
    row = df.iloc[i] if i < len(df) else pd.Series()
    return {
        "iter": int(fr.get("iter", int(row["iter"]) if "iter" in row else i)),
        "coords": fr["coords"],
        "atom_types": fr["atom_types"],
        "energy": float(row["energy"]) if "energy" in row else 0.0,
        "a": float(row["a"]) if "a" in row else 1.0,
        "b": float(row["b"]) if "b" in row else 1.0,
        "c": float(row["c"]) if "c" in row else 1.0,
        "alpha": float(row["alpha"]) if "alpha" in row else 90.0,
        "beta": float(row["beta"]) if "beta" in row else 90.0,
        "gamma": float(row["gamma"]) if "gamma" in row else 90.0,
    }

# -----------------------------
# Action implementations
# -----------------------------
def _run_analyze(args: argparse.Namespace) -> int:
    xh = XmoloutHandler(args.xmolout)
    from reaxkit.io.fort7_handler import Fort7Handler
    f7 = Fort7Handler(args.fort7)

    valences = _parse_kv_map(args.valences, float)
    if not valences:
        raise SystemExit("❌ Provide --valences like 'Mg=2,O=2'.")

    frames = _normalize_frames(xh, None, args.start, args.end, args.every)

    df = coordination_status_over_frames(
        f7, xh,
        valences=valences,
        threshold=args.threshold,
        frames=frames,
        iterations=None,
        require_all_valences=not args.allow_missing_valences,
    )

    if args.save:
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.save, index=False)
        print(f"[Done] Saved coordination table to {args.save}")
    else:
        summary = df.groupby(["frame_index", "status"], dropna=False).size().unstack(fill_value=0)
        print(summary)
    return 0

def _run_relabel(args: argparse.Namespace) -> int:
    xh = XmoloutHandler(args.xmolout)
    from reaxkit.io.fort7_handler import Fort7Handler
    f7 = Fort7Handler(args.fort7)

    valences = _parse_kv_map(args.valences, float)
    if not valences:
        raise SystemExit("❌ Provide --valences like 'Mg=2,O=2'.")

    labels = _parse_status_labels(args.labels)
    mode: Literal["global", "by_type"] = args.mode
    frames = _normalize_frames(xh, None, args.start, args.end, args.every)

    status_df = coordination_status_over_frames(
        f7, xh,
        valences=valences,
        threshold=args.threshold,
        frames=frames,
        iterations=None,
        require_all_valences=not args.allow_missing_valences,
    )

    if status_df.empty:
        blocks = (_frame_record_from_handler(xh, i) for i in frames)
        write_xmolout_from_frames(
            blocks, args.out,
            simulation_name=args.simulation or getattr(xh, "simulation_name", None) or "MD",
            precision=args.precision,
        )
        print(f"⚠️ No frames selected or empty status; wrote pass-through xmolout to {args.out}")
        return 0

    sim_df = xh.dataframe()
    frame_blocks: List[Dict[str, Any]] = []
    for fi, g in status_df.groupby("frame_index", sort=True):
        fi = int(fi)
        fr = xh.frame(fi)
        coords = fr["coords"]
        orig_types = fr["atom_types"]
        n = len(orig_types)

        g_sorted = g.sort_values("atom_id")  # 1-based ids
        if len(g_sorted) != n:
            raise SystemExit(f"❌ Frame {fi}: mismatch atoms between xmolout ({n}) and status rows ({len(g_sorted)}).")

        new_types: List[str] = []
        for k in range(n):
            t0 = str(orig_types[k])
            st_val = g_sorted.iloc[k]["status"]
            if pd.isna(st_val):
                new_types.append(t0)
                continue
            st = int(st_val)
            tag = labels.get(st, { -1:"U", 0:"C", 1:"O" }[st])
            if mode == "global":
                new_types.append(str(tag))
            else:
                if st == 0 and args.keep_coord_original:
                    new_types.append(t0)
                else:
                    new_types.append(f"{t0}{tag}")

        row = sim_df.iloc[fi] if fi < len(sim_df) else pd.Series()
        frame_blocks.append({
            "iter": int(g_sorted.iloc[0]["iter"]) if "iter" in g_sorted.columns else int(fr["iter"]),
            "coords": coords,
            "atom_types": new_types,
            "E_pot": float(row["E_pot"]) if "E_pot" in row else 0.0,
            "a": float(row["a"]) if "a" in row else 1.0,
            "b": float(row["b"]) if "b" in row else 1.0,
            "c": float(row["c"]) if "c" in row else 1.0,
            "alpha": float(row["alpha"]) if "alpha" in row else 90.0,
            "beta":  float(row["beta"])  if "beta"  in row else 90.0,
            "gamma": float(row["gamma"]) if "gamma" in row else 90.0,
        })

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    write_xmolout_from_frames(
        frame_blocks, args.out,
        simulation_name=args.simulation or getattr(xh, "simulation_name", None) or "MD",
        precision=args.precision,
    )
    print(f"[Done] Wrote relabeled xmolout to {args.out}")
    return 0

# -----------------------------
# Template-style registration
# -----------------------------
def register_tasks(subparsers: argparse._SubParsersAction) -> None:
    """
    Your CLI already adds a top-level 'coord' group.
    Here we register two child subcommands under that group:
      - analyze
      - relabel
    So you will run:
      reaxkit coord analyze ...
      reaxkit coord relabel ...
    """
    # coord analyze
    p1 = subparsers.add_parser("analyze", help="Analyze coordination per atom per frame"
                                               "reaxkit coord analyze --valences 'Mg=2,O=2,Zn=2' --save coord.csv")
    p1.add_argument("--xmolout", default='xmolout', help="Path to xmolout")
    p1.add_argument("--fort7", default='fort.7', help="Path to fort.7 (or Fort7-compatible) file")
    p1.add_argument("--valences", required=True, help="Type valences, e.g. 'Mg=2,O=2,H=1'")
    p1.add_argument("--threshold", type=float, default=0.3, help="Tolerance around valence")
    p1.add_argument("--allow-missing-valences", action="store_true", help="Keep atoms with unknown valence (status=NaN)")
    p1.add_argument("--start", type=int, default=None, help="Start frame (0-based)")
    p1.add_argument("--end", type=int, default=None, help="End frame (exclusive)")
    p1.add_argument("--every", type=int, default=1, help="Stride over frames")
    p1.add_argument("--save", default=None, help="CSV path to save full coordination table")
    p1.set_defaults(_run=_run_analyze)

    # coord relabel
    p2 = subparsers.add_parser("relabel", help="Relabel atom types by coordination and write a new xmolout"
                                               "reaxkit coord relabel --valences 'Mg=2,O=2,Zn=2' --out xmolout_relabeled --mode global --labels=-1=U,0=C,1=O"
                                               "reaxkit coord relabel --valences 'Mg=2,O=2,Zn=2' --out xmolout_type --mode by_type --keep-coord-original"
                                               "reaxkit coord relabel --valences 'Mg=2,O=2,Zn=2' --out xmolout_relabeled --mode global")
    p2.add_argument("--xmolout", default='xmolout', help="Path to xmolout")
    p2.add_argument("--fort7", default='fort.7', help="Path to fort.7 (or Fort7-compatible) file")
    p2.add_argument("--out", required=True, help="Output xmolout path")
    p2.add_argument("--valences", required=True, help="Type valences, e.g. 'Mg=2,O=2,H=1'")
    p2.add_argument("--threshold", type=float, default=0.3, help="Tolerance around valence")
    p2.add_argument("--mode", choices=["global", "by_type"], default="global", help="Relabeling mode")
    p2.add_argument("--labels", default=None, help="Status→tag map, e.g. '-1=U,0=C,1=O'")
    p2.add_argument("--keep-coord-original", action="store_true", help="In by_type mode, keep original label when status==0")
    p2.add_argument("--simulation", default=None, help="Override header simulation name")
    p2.add_argument("--precision", type=int, default=6, help="Float precision")
    p2.add_argument("--start", type=int, default=None, help="Start frame (0-based)")
    p2.add_argument("--end", type=int, default=None, help="End frame (exclusive)")
    p2.add_argument("--every", type=int, default=1, help="Stride over frames")
    p2.add_argument("--allow-missing-valences", action="store_true", help="Keep atoms with unknown valence (status=NaN)")
    p2.set_defaults(_run=_run_relabel)
