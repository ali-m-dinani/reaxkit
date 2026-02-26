"""
xmolout trajectory analysis utilities.

This module provides atomistic and trajectory-level analysis tools for
ReaxFF ``xmolout`` files via ``XmoloutHandler``.

It supports extraction of atom properties, trajectories, simulation box
information, displacement metrics, and atom-type mappings.

Typical use cases include:

- exporting per-atom coordinates or properties across frames
- building atom trajectories in long or wide format
- tracking box dimensions and thermodynamic scalars over time
- computing mean-squared displacement (MSD)
"""


from __future__ import annotations
from typing import Optional, Sequence, Union, Dict, Any, List
import numpy as np
import pandas as pd

from reaxkit.engine.reaxff.io.xmolout_handler import XmoloutHandler
from reaxkit.core.frame_utils import select_frames as _df_select

# ==========================================================
# === non-RDF helpers (atom tables/trajectories/box) ===
# ==========================================================

FrameSel = Optional[Union[Sequence[int], range, slice]]
AtomSel  = Optional[Union[Sequence[int], slice]]

BASE_ATOM_COLS = ("atom_type", "x", "y", "z")

def _frame_table(xh: XmoloutHandler, i: int) -> pd.DataFrame:
    """Internal helper: return a DataFrame for a specific frame index from an XmoloutHandler.

    If pre-parsed frames are cached in `xh._frames`, retrieves directly from cache.
    Otherwise, loads the frame via `xh.frame(i)` and constructs a DataFrame with:
        - atom_type
        - x, y, z coordinates

    Used internally by higher-level utilities (e.g., atom table extraction,
    per-frame analysis, or visualization routines).
    """
    if hasattr(xh, "_frames") and i < len(xh._frames):
        return xh._frames[i]
    fr = xh.frame(i)
    return pd.DataFrame({
        "atom_type": fr["atom_types"],
        "x": fr["coords"][:, 0], "y": fr["coords"][:, 1], "z": fr["coords"][:, 2],
    })

def _get_xmolout_data_per_atom(
    xh: XmoloutHandler,
    *,
    frames: FrameSel = None,
    every: int = 1,
    atoms: AtomSel = None,
    atom_types: Optional[Sequence[str]] = None,
    extra_cols: Optional[Sequence[str]] = None,
    include_xyz: bool = True,
    format: str = "long",
) -> pd.DataFrame:
    """Extract per-atom properties across selected frames.

    Works on
    --------
    XmoloutHandler — ``xmolout``

    Parameters
    ----------
    xh : XmoloutHandler
        Parsed xmolout handler.
    frames : int, slice, or sequence of int, optional
        Frame indices to include.
    every : int, default=1
        Subsample frames by taking every Nth frame.
    atoms : sequence of int or slice, optional
        Atom indices to include (0-based internally, exported as 1-based).
    atom_types : sequence of str, optional
        Atom types to include (e.g. ``["Al", "N"]``).
    extra_cols : sequence of str, optional
        Additional per-atom columns to include if present.
    include_xyz : bool, default=True
        Whether to include ``x``, ``y``, ``z`` coordinates.
    format : {"long", "wide"}, default="long"
        Output format.

    Returns
    -------
    pandas.DataFrame
        Atom-level table in long or wide format, including frame and iteration
        metadata.

    Examples
    --------
    >>> df = _get_xmolout_data_per_atom(xh, frames=slice(0, 10), atom_types=["O"])
      """
    df_sim = xh.dataframe()
    sub_df = _df_select(df_sim, frames)
    fidx_all = list(sub_df.index)[::max(1, int(every))]

    rows: list[dict[str, Any]] = []
    for i in fidx_all:
        ft = _frame_table(xh, i)
        if atoms is not None:
            if isinstance(atoms, slice):
                atom_sel = list(range(*atoms.indices(len(ft))))
            else:
                atom_sel = [int(a) for a in atoms if 0 <= int(a) < len(ft)]
        elif atom_types:
            tset = {str(t) for t in atom_types}
            atom_sel = [j for j, t in enumerate(ft["atom_type"].astype(str)) if t in tset]
        else:
            atom_sel = list(range(len(ft)))

        extras_here = [c for c in ft.columns if c not in BASE_ATOM_COLS]
        wanted = extras_here if extra_cols is None else [c for c in extra_cols if c in ft.columns]
        vals_cols = (["x", "y", "z"] if include_xyz else []) + wanted

        for j in atom_sel:
            rec = {
                "frame_index": int(i),
                "iter": int(df_sim.iloc[i]["iter"]) if "iter" in df_sim.columns else int(i),
                "atom_id": int(j) + 1, #1-based atom format
                "atom_type": str(ft.at[j, "atom_type"]),
            }
            for c in vals_cols:
                rec[c] = ft.at[j, c] if c in ft.columns else np.nan
            rows.append(rec)

    out = pd.DataFrame(rows).sort_values(["frame_index", "atom_id"]).reset_index(drop=True)

    if format == "long":
        return out

    if format == "wide":
        id_cols = ["frame_index", "iter"]
        value_cols = [c for c in out.columns if c not in (id_cols + ["atom_id", "atom_type"])]
        wide = out[id_cols + ["atom_id"] + value_cols].pivot(index=id_cols, columns="atom_id", values=value_cols)
        wide.columns = [f"{col}[{aid}]" for (col, aid) in wide.columns.to_flat_index()]
        return wide.reset_index().sort_values("frame_index").reset_index(drop=True)

    raise ValueError("format must be 'long' or 'wide'")

def get_unit_cell_dimensions_across_frames(
    xh: XmoloutHandler,
    *,
    frames: FrameSel = None,
    every: int = 1,
) -> pd.DataFrame:
    """
    Extract simulation box dimensions and scalar quantities per frame.

    Works on
    --------
    XmoloutHandler — ``xmolout``

    Parameters
    ----------
    xh : XmoloutHandler
        Parsed xmolout handler.
    frames : int, slice, or sequence of int, optional
        Frame indices to include.
    every : int, default=1
        Subsample frames by taking every Nth frame.

    Returns
    -------
    pandas.DataFrame
        Table with columns such as ``a``, ``b``, ``c``, ``alpha``, ``beta``,
        ``gamma``, ``E_pot``, and ``num_of_atoms``.

    Examples
    --------
    >>> df = get_unit_cell_dimensions_across_frames(xh)
    """
    df_sim = xh.dataframe()
    sub_df = _df_select(df_sim, frames)
    fidx = list(sub_df.index)[::max(1, int(every))]

    if df_sim.empty:
        return pd.DataFrame(columns=["frame_index", "iter", "a", "b", "c",
                                     "alpha", "beta", "gamma", "E_pot", "num_of_atoms"])

    out = df_sim.iloc[fidx].copy().reset_index(drop=True)
    out.insert(0, "frame_index", fidx)
    wanted = ["frame_index", "iter", "a", "b", "c", "alpha", "beta", "gamma", "E_pot", "num_of_atoms"]
    return out[[c for c in wanted if c in out.columns]]


def get_atom_trajectories(
    xh: XmoloutHandler,
    *,
    frames: FrameSel = None,
    every: int = 1,
    atoms: AtomSel = None,                    # indices or slice
    atom_types: Optional[Sequence[str]] = None,
    dims: Sequence[str] = ("x", "y", "z"),
    format: str = "long",                     # 'long' or 'wide'
) -> pd.DataFrame:
    """Extract atomic trajectories across frames.

    Works on
    --------
    XmoloutHandler — ``xmolout``

    Parameters
    ----------
    xh : XmoloutHandler
        Parsed xmolout handler.
    frames : int, slice, or sequence of int, optional
        Frame indices to include.
    every : int, default=1
        Subsample frames by taking every Nth frame.
    atoms : sequence of int or slice, optional
        Atom indices to include.
    atom_types : sequence of str, optional
        Atom types to include.
    dims : sequence of {"x", "y", "z"}, default=("x","y","z")
        Coordinate components to extract.
    format : {"long", "wide"}, default="long"
        Output format.

    Returns
    -------
    pandas.DataFrame
        Atom trajectories in long or wide format.

    Examples
    --------
    >>> df = get_atom_trajectories(xh, atoms=[1, 2], dims=("z",))
    """
    dims = tuple(d for d in dims if d in ("x", "y", "z"))
    if not dims:
        raise ValueError("dims must include at least one of 'x','y','z'")

    df_sim = xh.dataframe()
    sub_df = _df_select(df_sim, frames)
    fidx = list(sub_df.index)[::max(1, int(every))]

    rows: List[Dict[str, Any]] = []
    for i in fidx:
        fr = xh.frame(i)
        coords = fr["coords"]

        # selection
        if atoms is not None:
            if isinstance(atoms, slice):
                atom_sel = list(range(*atoms.indices(coords.shape[0])))
            else:
                atom_sel = [int(a) for a in atoms if 0 <= int(a) < coords.shape[0]]
        elif atom_types:
            tset = {str(t) for t in atom_types}
            atom_sel = [j for j, t in enumerate(fr["atom_types"]) if str(t) in tset]
        else:
            atom_sel = list(range(coords.shape[0]))

        for j in atom_sel:
            rec = {
                "frame_index": int(i),
                "iter": int(fr.get("iter", i)),
                "atom_id": int(j) + 1,
                "atom_type": str(fr["atom_types"][j]),
            }
            if "x" in dims: rec["x"] = float(coords[j, 0])
            if "y" in dims: rec["y"] = float(coords[j, 1])
            if "z" in dims: rec["z"] = float(coords[j, 2])
            rows.append(rec)

    out = pd.DataFrame(rows).sort_values(["frame_index", "atom_id"]).reset_index(drop=True)
    if format == "long":
        return out

    if format == "wide":
        val_cols = [d for d in ("x", "y", "z") if d in dims]
        to_pivot = out[["frame_index", "iter", "atom_id"] + val_cols]
        wide = to_pivot.pivot(index=["frame_index", "iter"], columns="atom_id", values=val_cols)
        wide.columns = [f"{d}[{aid}]" for (d, aid) in wide.columns.to_flat_index()]
        return wide.reset_index().sort_values("frame_index").reset_index(drop=True)

    raise ValueError("format must be 'long' or 'wide'")




def get_atom_type_mapping(
    xh: XmoloutHandler,
    frame: int = 0,
) -> Dict[str, Any]:
    """Return atom-type mappings for a given frame.

    Works on
    --------
    XmoloutHandler — ``xmolout``

    Parameters
    ----------
    xh : XmoloutHandler
        Parsed xmolout handler.
    frame : int, default=0
        Frame index to inspect.

    Returns
    -------
    dict
        Mapping containing:
        - ``types``: sorted unique atom types
        - ``type_to_indices``: atom indices per type (1-based)
        - ``index_to_type``: per-atom type list

    Examples
    --------
    >>> m = get_atom_type_mapping(xh)
    >>> m["types"]
    """
    fr = xh.frame(int(frame))
    types = [str(t) for t in fr["atom_types"]]
    uniq = sorted(set(types))
    type_to_indices: Dict[str, List[int]] = {t: [] for t in uniq}
    for idx, t in enumerate(types, start=1):  # start=1 for 1-based numbering
        type_to_indices[t].append(idx)
    return {
        "types": uniq,
        "type_to_indices": type_to_indices,
        "index_to_type": types,
    }


# ==========================================================
# ========== Single public entry for RDF & properties ======
# ==========================================================

