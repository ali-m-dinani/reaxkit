"""analyzer for fort.7 file"""
from __future__ import annotations
import re
import pandas as pd

from reaxkit.utils.frame_utils import resolve_indices
from typing import Iterable, List, Optional, Sequence, Union, Literal, Tuple
from reaxkit.io.xmolout_handler import XmoloutHandler
from reaxkit.analysis.coordination_analyzer import (
    classify_coordination_for_frame,
    status_label,
)

Indexish = Union[int, Iterable[int], None]

# --------------------------- frame/column helpers ---------------------------
def _resolve_columns(
    df_example: pd.DataFrame,
    columns: Union[str, Sequence[str]],
    regex: bool = False,
    must_exist: bool = True,
) -> List[str]:
    """
    Map requested column name(s) or regex pattern(s) to real DataFrame columns.
    """
    if isinstance(columns, str):
        columns = [columns]

    resolved: List[str] = []
    cols = list(df_example.columns)

    for spec in columns:
        if regex:
            pat = re.compile(spec)
            matches = [c for c in cols if pat.search(c)]
            resolved.extend(matches)
        else:
            if spec in cols:
                resolved.append(spec)
            elif not must_exist:
                # skip silently
                pass
            else:
                raise KeyError(f"Column '{spec}' not found. Example columns: {cols[:10]} ...")

    # de-dup preserving order
    out, seen = [], set()
    for c in resolved:
        if c not in seen:
            out.append(c)
            seen.add(c)

    if must_exist and not out:
        raise KeyError("No columns matched the request.")
    return out


# --------------------------- atom-level features ---------------------------

def features_atom(
    handler,
    columns: Union[str, Sequence[str]],
    frames: Indexish = None,
    iterations: Indexish = None,
    regex: bool = False,
    add_index_cols: bool = True,
) -> pd.DataFrame:
    """Extracts per-ATOM feature columns (such as partial charges) across selected frames.

    Examples:
      features_atom(h, "partial_charge")
      features_atom(h, r"^atom_cnn\\d+$", regex=True)   # all atom_cnn*

    Returns:
      Tidy DataFrame with columns:
        - frame_idx, iter, atom_idx  (if add_index_cols=True)
        - requested feature columns
    """
    sim_df = handler.dataframe()
    idx_list = resolve_indices(handler, frames, iterations)

    out_frames = []
    for fi in idx_list:
        atoms = handler._frames[fi]
        cols = _resolve_columns(atoms, columns, regex=regex, must_exist=True)
        part = atoms[cols].copy()
        if add_index_cols:
            part.insert(0, "atom_idx", range(len(part)))
            part.insert(0, "iter", int(sim_df.iloc[fi]["iter"]))
            part.insert(0, "frame_idx", fi)
        out_frames.append(part)

    if not out_frames:
        return pd.DataFrame(columns=(["frame_idx", "iter", "atom_idx"] if add_index_cols else []) + (columns if isinstance(columns, list) else [columns]))
    return pd.concat(out_frames, ignore_index=True)


# ------------------------- iter-level features -------------------------

def features_summary(
    handler,
    columns: Union[str, Sequence[str]],
    frames: Indexish = None,
    iterations: Indexish = None,
    regex: bool = False,
    add_index_cols: bool = True,
) -> pd.DataFrame:
    """Extract per-iter (summary) feature columns (such as total lone pairs) from sim_df.

    Examples:
      features_summary(h, "total_charge")      # from sim_df
      features_summary(h, ["iter","num_bonds","total_BO"])
      features_summary(h, r"^total_.*$", regex=True)
    """
    sim_df = handler.dataframe()
    idx_list = resolve_indices(handler, frames, iterations)

    if len(sim_df) == 0:
        return pd.DataFrame()

    cols = _resolve_columns(sim_df, columns, regex=regex, must_exist=True)
    part = sim_df.iloc[idx_list][cols].reset_index(drop=True)

    if add_index_cols and "iter" not in part.columns:
        # Provide iter + frame_idx for context if the user didn't request iter
        meta_df = sim_df.loc[idx_list, ["iter"]].reset_index(drop=True)
        meta_df.insert(0, "frame_idx", idx_list)
        part = pd.concat([meta_df, part], axis=1)
    elif add_index_cols and "iter" in part.columns:
        part.insert(0, "frame_idx", idx_list)

    return part


# ---------------------------- convenience slices ----------------------------

def partial_charges(
    handler,
    frames: Indexish = None,
    iterations: Indexish = None,
) -> pd.DataFrame:
    """convenience function for getting all partial charges across selected frames."""
    return features_atom(handler, "partial_charge", frames=frames, iterations=iterations)


def all_atom_cnn(
    handler,
    frames: Indexish = None,
    iterations: Indexish = None,
) -> pd.DataFrame:
    """convenience function for getting all atom_cnn* columns (connectivity to other atoms) across selected frames."""
    return features_atom(handler, r"^atom_cnn\d+$", frames=frames, iterations=iterations, regex=True)

def sum_bos(
    handler,
    frames: Indexish = None,
    iterations: Indexish = None,
) -> pd.DataFrame:
    """convenience function for getting per-atom sum_BOs (total bond order) across selected frames.

    Returns tidy DataFrame with columns:
      - frame_idx (0-based)
      - iter
      - atom_idx (0-based)
      - sum_BOs
    """
    return features_atom(
        handler,
        columns="sum_BOs",
        frames=frames,
        iterations=iterations,
        regex=False,
        add_index_cols=True,
    )


# --- Coordination classification using features_atom("sum_BOs") --------------
def coordination_status_over_frames(
    f7_handler,
    xh: XmoloutHandler,
    *,
    valences: Mapping[str, float],
    threshold: float = 0.9,
    frames: Indexish = None,
    iterations: Indexish = None,
    require_all_valences: bool = True,
) -> pd.DataFrame:
    """finds the coordination status (over- or under-coordination) for each atom across all frames.
    For every selected frame and atom:
      - read sum_BOs from fort7_analyzer.features_atom
      - read atom types from xmolout_handler
      - compare to per-type valence with a ±threshold window
      - output {-1,0,+1} as under/coord/over, plus label

    Returns columns:
      frame_index, iter, atom_id (1-based), atom_type, sum_BOs,
      valence, delta, status (-1/0/+1 or NaN), status_label
    """
    # Pull sum_BOs in tidy form (frame_idx, iter, atom_idx, sum_BOs)
    df_sum = sum_bos(f7_handler, frames=frames, iterations=iterations)
    if df_sum.empty:
        return pd.DataFrame(columns=[
            "frame_index","iter","atom_id","atom_type","sum_BOs",
            "valence","delta","status","status_label"
        ])

    # Group by frame and classify
    rows = []
    for fi, g in df_sum.groupby("frame_idx", sort=True):
        g = g.sort_values("atom_idx")
        sum_vals = g["sum_BOs"].to_numpy(dtype=float)

        # Atom types for this frame come from xmolout
        fr = xh.frame(int(fi))
        atom_types = fr["atom_types"]
        if len(sum_vals) != len(atom_types):
            raise ValueError(
                f"Length mismatch at frame {fi}: sum_BOs({len(sum_vals)}) vs atom_types({len(atom_types)})"
            )

        per_atom = classify_coordination_for_frame(
            sum_bos=sum_vals,
            atom_types=atom_types,
            valences=valences,
            threshold=threshold,
            require_all_valences=require_all_valences,
        )

        # Attach frame metadata
        iter = int(g["iter"].iloc[0]) if "iter" in g.columns else int(fi)
        per_atom.insert(0, "iter", iter)
        per_atom.insert(0, "frame_index", int(fi))

        # Friendly label
        per_atom["status_label"] = status_label(per_atom["status"])

        rows.append(per_atom)

    out = pd.concat(rows, ignore_index=True)
    out = out.sort_values(["frame_index", "atom_id"], kind="mergesort").reset_index(drop=True)
    return out


def summary_metric_vs_iter(
    handler,
    field: str = "total_charge",
    frames: Indexish = None,
    iterations: Indexish = None,
) -> pd.DataFrame:
    """extracts a single summary field (e.g., total_charge) vs iter.
    """
    df = features_summary(handler, ["iter", field], frames=frames, iterations=iterations, regex=False, add_index_cols=False)
    return df.sort_values("iter").reset_index(drop=True)








