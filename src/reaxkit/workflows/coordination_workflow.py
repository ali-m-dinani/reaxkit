"""reads fort7 coordination data (i.e., total BO) and then re-writes a xmolout file based on coordination of atoms."""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, Sequence, Union, Literal, List
import pandas as pd

from reaxkit.io.xmolout_handler import XmoloutHandler
from reaxkit.io.xmolout_generator import write_xmolout_from_frames
from reaxkit.analysis.fort7_analyzer import coordination_status_over_frames
from reaxkit.utils.path import resolve_output_path
from reaxkit.io.ffield_handler import FFieldHandler

# -----------------------------
# Helpers
# -----------------------------
def _normalize_frames(xh: XmoloutHandler, frames: Optional[str]) -> list[int]:
    """
    Normalize a frame-selection string into a sorted list of frame indices.

    Supported formats:
      - 'start:end:step'  (any of the 3 can be omitted, e.g. '::5', '10:', ':100')
      - 'i,j,k' or 'i j k' (explicit indices)
    If frames is None, all frames [0, n_frames) are used.
    """
    n = xh.n_frames()

    if not frames:
        return list(range(n))

    s = frames.strip()

    # Slice-like 'start:end:step'
    if ":" in s:
        parts = s.split(":")
        if len(parts) > 3:
            raise ValueError(f"Invalid --frames slice specification: {frames!r}")
        start_s = parts[0].strip() if len(parts) >= 1 else ""
        end_s   = parts[1].strip() if len(parts) >= 2 else ""
        step_s  = parts[2].strip() if len(parts) == 3 else ""

        start = int(start_s) if start_s else 0
        end   = int(end_s)   if end_s   else n
        step  = int(step_s)  if step_s  else 1
        if step == 0:
            raise ValueError("Step in --frames slice cannot be 0.")

        start = max(0, start)
        end   = min(n, end)
        return list(range(start, end, step))

    # Comma/space-separated explicit indices
    idx: List[int] = []
    for tok in s.replace(",", " ").split():
        i = int(tok)
        if 0 <= i < n:
            idx.append(i)
    return sorted(set(idx))


def _valences_from_ffield(ffield_path: str) -> Dict[str, float]:
    """
    Read atom valencies from ffield atom section.

    Returns a dict that includes BOTH:
      - symbol -> valency  (e.g., "O": 2)
      - atom_index -> valency as string key (e.g., "2": 2)
    This makes it robust whether xmolout uses symbols or numeric atom types.
    """
    fh = FFieldHandler(ffield_path)
    atom_df = fh.section_df(FFieldHandler.SECTION_ATOM)

    if "valency" not in atom_df.columns:
        raise SystemExit("❌ ffield atom section missing 'valency' column.")

    out: Dict[str, float] = {}
    used_pairs = []  # for printing

    for atom_idx, row in atom_df.iterrows():
        # atom_idx is the ffield atom index (usually 1-based)
        sym = row.get("symbol")
        val = row.get("valency")

        if val is None or (isinstance(val, float) and pd.isna(val)):
            continue

        try:
            v = float(val)
        except Exception:
            continue

        # key by atom index (string) too, for numeric xmolout types
        out[str(int(atom_idx))] = v

        # key by symbol if present
        if sym is not None and not (isinstance(sym, float) and pd.isna(sym)):
            s = str(sym).strip()
            if s:
                out[s] = v
                used_pairs.append(f"{s}={v:g}")
            else:
                used_pairs.append(f"{int(atom_idx)}={v:g}")
        else:
            used_pairs.append(f"{int(atom_idx)}={v:g}")

    if not out:
        raise SystemExit("❌ No usable valencies found in ffield atom section.")

    # Print one clean line showing what you used (symbol-first when available)
    # (You can keep it in this helper so both tasks benefit.)
    print("[Valences] " + ", ".join(used_pairs))
    return out


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
        out.setdefault(k, {-1: "U", 0: "C", 1: "O"}[k])
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
def _task_analyze(args: argparse.Namespace) -> int:
    xh = XmoloutHandler(args.xmolout)
    from reaxkit.io.fort7_handler import Fort7Handler
    f7 = Fort7Handler(args.fort7)

    valences = _parse_kv_map(args.valences, float)
    if valences:
        # User explicitly provided valences; print what will be used
        print("[Valences] " + ", ".join(f"{k}={v:g}" for k, v in valences.items()))
    else:
        valences = _valences_from_ffield(args.ffield)
    if not valences:
        raise SystemExit("❌ Provide --valences like 'Mg=2,O=2'.")

    frames = _normalize_frames(xh, args.frames)

    df = coordination_status_over_frames(
        f7, xh,
        valences=valences,
        threshold=args.threshold,
        frames=frames,
        iterations=None,
        require_all_valences=not args.allow_missing_valences,
    )

    workflow_name = args.kind
    if args.export:
        export_path = resolve_output_path(args.export, workflow_name)
        export_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(export_path, index=False)
        print(f"[Done] Exported coordination table to {export_path}")
    else:
        summary = df.groupby(["frame_index", "status"], dropna=False) \
            .size().unstack(fill_value=0)
        print(summary)

    return 0


def _task_relabel(args: argparse.Namespace) -> int:
    xh = XmoloutHandler(args.xmolout)
    from reaxkit.io.fort7_handler import Fort7Handler
    f7 = Fort7Handler(args.fort7)

    valences = _parse_kv_map(args.valences, float)
    if valences:
        # User explicitly provided valences; print what will be used
        print("[Valences] " + ", ".join(f"{k}={v:g}" for k, v in valences.items()))
    else:
        valences = _valences_from_ffield(args.ffield)
    if not valences:
        raise SystemExit("❌ Provide --valences like 'Mg=2,O=2'.")

    labels = _parse_status_labels(args.labels)
    mode: Literal["global", "by_type"] = args.mode
    frames = _normalize_frames(xh, args.frames)

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
            blocks, args.output,
            simulation_name=args.simulation or getattr(xh, "simulation_name", None) or "MD",
            precision=args.precision,
        )
        print(f"⚠️ No frames selected or empty status; wrote pass-through xmolout to {args.output}")
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
            tag = labels.get(st, {-1: "U", 0: "C", 1: "O"}[st])
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

    workflow_name = args.kind
    output_path = resolve_output_path(args.output, workflow_name)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    write_xmolout_from_frames(
        frame_blocks,
        output_path,
        simulation_name=args.simulation or getattr(xh, "simulation_name", None) or "MD",
        precision=args.precision,
    )

    print(f"[Done] Wrote relabeled xmolout to {output_path}")
    return 0


# -----------------------------
# Template-style registration
# -----------------------------
def _add_common_coord_args(p: argparse.ArgumentParser) -> None:
    """Common args shared by coord analyze / relabel."""
    p.add_argument("--xmolout", default="xmolout", help="Path to xmolout.")
    p.add_argument("--fort7", default="fort.7", help="Path to fort.7 file.")
    p.add_argument("--valences", default=None,
                   help="Optional: override type valences, e.g. 'Mg=2,O=2,H=1'. "
                        "If omitted, reads from ffield atom section ('valency').")
    p.add_argument("--ffield", default="ffield", help="Path to ffield (used if --valences not given).")
    p.add_argument("--threshold", type=float, default=0.3, help="Tolerance around valence.")
    p.add_argument("--frames", default=None,
                   help="Frame selection, e.g. '0:100:5' or '0,5,10'.")
    p.add_argument("--allow-missing-valences", action="store_true",
                   help="Keep atoms with unknown valence (status=NaN).")


def register_tasks(subparsers: argparse._SubParsersAction) -> None:
    """
    Register subcommands under:
        reaxkit coord analyze
        reaxkit coord relabel
    """

    # -------------------- analyze --------------------
    p1 = subparsers.add_parser(
        "analyze",
        help="Analyze coordination per atom per frame.",
        description=(
            "Examples:\n"
            "  reaxkit coord analyze --export coord_analysis.csv\n"
            "  reaxkit coord analyze --valences 'Mg=2,O=2' --frames 0:200:2 --export coord_0_200_2.csv\n"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )

    _add_common_coord_args(p1)
    p1.add_argument("--export", default=None, help="Path to export coordination CSV.")
    p1.set_defaults(_run=_task_analyze)


    # -------------------- relabel --------------------
    p2 = subparsers.add_parser(
        "relabel",
        help="Relabel atom types by coordination and write a new xmolout.",
        description=(
            "Examples:\n"
            "  reaxkit coord relabel --output xmolout_relabeled --mode global --labels=-1=U,0=C,1=O\n"
            "  reaxkit coord relabel --output xmolout_type --mode by_type --keep-coord-original\n"
            "  reaxkit coord relabel --valences 'Mg=2,O=2,Zn=2' --frames 0:400:5 "
            "--output xmolout_relabeled --mode global\n"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )

    _add_common_coord_args(p2)
    p2.add_argument("--output", required=True, help="Output xmolout path.")
    p2.add_argument("--mode", choices=["global", "by_type"], default="global",
                    help="Relabeling mode.")
    p2.add_argument("--labels", default=None,
                    help="Status→tag map, e.g. '-1=U,0=C,1=O'.")
    p2.add_argument("--keep-coord-original", action="store_true",
                    help="In by_type mode, keep original label when status==0.")
    p2.add_argument("--simulation", default=None, help="Override header simulation name.")
    p2.add_argument("--precision", type=int, default=6, help="Float precision.")
    p2.set_defaults(_run=_task_relabel)
