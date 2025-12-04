"""analyzer for xmolout file"""
from __future__ import annotations
from typing import Iterable, Optional, Sequence, Union, Dict, Any, List, Tuple
import numpy as np
import pandas as pd

from reaxkit.io.xmolout_handler import XmoloutHandler
from reaxkit.utils.frame_utils import select_frames as _df_select

from reaxkit.analysis.RDF_analyzer import (
        rdf_using_freud as _rdf_freud_many,
        rdf_using_ovito as _rdf_ovito_many,
        rdf_property_over_frames as _rdf_props,
    )

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

def get_atom_properties(
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
    """Extract per-atom properties across selected frames from an XmoloutHandler.

    Iterates over frames in the simulation and collects atomic properties such as
    type, coordinates, and any additional per-atom quantities (charges, forces, etc.).
    Supports selection by frame, atom index, or atom type, and returns data in either
    long (tidy) or wide format for downstream analysis or visualization.

    Behavior
    --------
    - Selects frames using `frames` (single, slice, or iterable) and optionally subsamples
      with `every` (e.g., every 5th frame).
    - Atom selection can be:
        * explicit indices (`atoms`),
        * filtered by atom type (`atom_types`),
        * or all atoms (default).
    - Includes coordinates (`x`, `y`, `z`) by default, plus extra per-atom columns
      from the frame table (e.g., `"partial_charge"`, `"bo_sum"`, etc.).
    - If `extra_cols` is given, only those columns are added if they exist.
    - Returns:
        * **long format** — one row per atom per frame.
        * **wide format** — each atom’s properties in separate columns.

    Notes
    -----
    - Coordinates are always included unless `include_xyz=False`.
    - Column names in wide format follow the pattern `"property[atom_id]"`.
    - Frame and iteration numbers are included for each record.
    - Uses internal `_frame_table()` and `_df_select()` for consistency with
      other ReaxKit data handlers.
    - Particularly useful for exporting structured atom data to pandas or
      CSV for further analysis.
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

def get_box_dimensions(
    xh: XmoloutHandler,
    *,
    frames: FrameSel = None,
    every: int = 1,
) -> pd.DataFrame:
    """Extract simulation box dimensions and related scalar quantities per selected frame.

    Returns a compact DataFrame containing box vectors, angles, energy, and atom count
    for each chosen frame in the Xmolout trajectory. Useful for tracking volume changes,
    strain evolution, or thermodynamic trends over time.
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
    """getting trajectories of selected atoms over frames.
    - format='long': one row per (frame, atom)
    - format='wide': columns like x[0], y[0], ... for each atom id
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


def mean_squared_displacement(
    xh: XmoloutHandler,
    *,
    frames: FrameSel = None,
    every: int = 1,
    atoms: AtomSel = None,
    atom_types: Optional[Sequence[str]] = None,
    dims: Sequence[str] = ("x", "y", "z"),
    origin: str = "first",     # 'first' or an int frame index inside selection
) -> pd.DataFrame:
    """Naive MSD (no PBC unwrapping). Computes |r_i(t) - r_i(0)|^2 per atom.

    Returns a long-format DataFrame with columns:
        frame_index, iter, atom_id, msd
    where `msd` is the per-atom squared displacement (no averaging).
    """
    dims = tuple(d for d in dims if d in ("x", "y", "z"))
    if not dims:
        raise ValueError("dims must include at least one of 'x','y','z'")

    df_sim = xh.dataframe()
    sub_df = _df_select(df_sim, frames)
    fidx = list(sub_df.index)[::max(1, int(every))]
    if not fidx:
        return pd.DataFrame(columns=["frame_index", "iter", "atom_id", "msd"])

    # Choose reference frame in the selection
    ref_frame = fidx[0] if origin == "first" else int(origin)
    if ref_frame not in fidx:
        raise ValueError("origin must be 'first' or a frame index inside the selected frames")

    # Build selection of atoms using the reference frame
    fr0 = xh.frame(ref_frame)
    coords0 = fr0["coords"]

    if atoms is not None:
        # interpret `atoms` as 1-based indices
        if isinstance(atoms, slice):
            # if user passes a slice, assume 0-based like Python
            sel = list(range(*atoms.indices(coords0.shape[0])))
        else:
            sel = []
            for a in atoms:
                ai = int(a)
                if 1 <= ai <= coords0.shape[0]:
                    sel.append(ai - 1)  # convert 1-based -> 0-based
    elif atom_types:
        tset = {str(t) for t in atom_types}
        sel = [j for j, t in enumerate(fr0["atom_types"]) if str(t) in tset]
    else:
        sel = list(range(coords0.shape[0]))

    if not sel:
        return pd.DataFrame(columns=["frame_index", "iter", "atom_id", "msd"])

    axes = {"x": 0, "y": 1, "z": 2}
    use_cols = [axes[d] for d in dims]

    sel_idx = np.asarray(sel, dtype=int)                # 0-based indices
    atom_ids = (sel_idx + 1).tolist()                   # 1-based atom ids
    r0 = coords0[sel_idx[:, None], use_cols].astype(float)  # (n_sel, len(dims))

    rows: List[Dict[str, Any]] = []
    for i in fidx:
        fr = xh.frame(i)
        coords = fr["coords"][sel_idx[:, None], use_cols].astype(float)
        dr = coords - r0                                 # (n_sel, len(dims))
        sq = np.sum(dr * dr, axis=1)                     # per-atom MSD, shape (n_sel,)

        iter_val = int(fr.get("iter", i))
        for atom_id, msd_val in zip(atom_ids, sq):
            rows.append(
                {
                    "frame_index": int(i),
                    "iter": iter_val,
                    "atom_id": int(atom_id),
                    "msd": float(msd_val),
                }
            )

    return (
        pd.DataFrame(rows)
        .sort_values(["frame_index", "atom_id"])
        .reset_index(drop=True)
    )



def get_atom_type_mapping(
    xh: XmoloutHandler,
    frame: int = 0,
) -> Dict[str, Any]:
    """Return a mapping of atom types for a given frame. it shows which atom (atom 1) is whic kind (i.e., Al).
      - types: sorted unique type names (list[str])
      - type_to_indices: dict[str, list[int]]
      - index_to_type:  list[str] (per-atom type for that frame)
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

def rdf(
    xh: XmoloutHandler,
    *,
    backend: str = "freud",                 # 'freud' or 'ovito'
    frames: Optional[Iterable[int]] = None, # frame indices; None => all
    types_a: Optional[Iterable[str]] = None,
    types_b: Optional[Iterable[str]] = None,
    r_max: Optional[float] = None,          # OVITO accepts float; FREUD accepts Optional[float]
    bins: int = 200,
    property: Optional[str] = None,         # None => return RDF curve; else 'first_peak'|'dominant_peak'|'area'|'excess_area'
    average: bool = True,                   # for curves: average across frames
    return_stack: bool = False              # for curves: if average=False, optionally return list of per-frame g(r)
):
    """Compute the radial distribution function (RDF) or RDF-derived properties over one or more frames.

    This is a unified entry point for both the `freud` and `ovito` RDF backends, providing
    total or partial RDFs and common derived metrics (e.g., first peak position, area).

    Behavior
    --------
    - If `property` is **None**:
        * Returns `(r, g)` RDF curve(s).
        * With `average=True` → returns mean RDF over all frames.
        * With `average=False` and `return_stack=True` → returns list of per-frame `g(r)` arrays.
        * With `average=False` and `return_stack=False` → returns last frame’s RDF only.
    - If `property` is set (`'first_peak'`, `'dominant_peak'`, `'area'`, `'excess_area'`):
        * Returns a tidy `DataFrame` with one value per frame (see `rdf_property_over_frames`).

    Notes
    -----
    - `backend` can be `"freud"` (faster, in-memory) or `"ovito"` (supports total/partial RDFs).
    - Frame selection via `frames`; if None, computes over all available frames.
    - If `types_a` and `types_b` are provided, computes a **partial RDF** for that atom pair.
    - `r_max` defines the cutoff radius; defaults vary by backend (≈4.0 Å for OVITO).
    - Common derived properties:
        * `"first_peak"` — first local maximum of g(r).
        * `"dominant_peak"` — highest global maximum.
        * `"area"` — total ∫g(r)dr.
        * `"excess_area"` — ∫(g(r) − 1)dr.
    """


    if property is None:
        if backend.lower() == "freud":
            return _rdf_freud_many(
                xh, frames=frames, types_a=types_a, types_b=types_b,
                r_max=r_max, bins=bins, average=average, return_stack=return_stack
            )
        elif backend.lower() == "ovito":
            return _rdf_ovito_many(
                xh, frames=frames, r_max=float(r_max or 4.0), bins=bins,
                types_a=types_a, types_b=types_b, average=average, return_stack=return_stack
            )
        else:
            raise ValueError("backend must be 'freud' or 'ovito'")

    # property mode
    return _rdf_props(
        xh, backend=backend, frames=frames, property=property,
        r_max=r_max, bins=bins, types_a=types_a, types_b=types_b
    )
