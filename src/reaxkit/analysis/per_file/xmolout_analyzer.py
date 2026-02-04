"""
xmolout trajectory analysis utilities.

This module provides atomistic and trajectory-level analysis tools for
ReaxFF ``xmolout`` files via ``XmoloutHandler``.

It supports extraction of atom properties, trajectories, simulation box
information, displacement metrics, atom-type mappings, and radial
distribution functions (RDFs) using multiple backends.

Typical use cases include:

- exporting per-atom coordinates or properties across frames
- building atom trajectories in long or wide format
- tracking box dimensions and thermodynamic scalars over time
- computing mean-squared displacement (MSD)
- computing total or partial RDFs and RDF-derived properties
"""


from __future__ import annotations
from typing import Iterable, Optional, Sequence, Union, Dict, Any, List
import numpy as np
import pandas as pd

from reaxkit.io.handlers.xmolout_handler import XmoloutHandler
from reaxkit.utils.frame_utils import select_frames as _df_select

from reaxkit.analysis.composed.RDF_analyzer import (
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


def get_mean_squared_displacement(
    xh: XmoloutHandler,
    *,
    frames: FrameSel = None,
    every: int = 1,
    atoms: AtomSel = None,
    atom_types: Optional[Sequence[str]] = None,
    dims: Sequence[str] = ("x", "y", "z"),
    origin: str = "first",     # 'first' or an int frame index inside selection
) -> pd.DataFrame:
    """Compute per-atom mean-squared displacement (MSD) without PBC unwrapping.

    Works on
    --------
    XmoloutHandler — ``xmolout``

    Parameters
    ----------
    xh : XmoloutHandler
        Parsed xmolout handler.
    frames : int, slice, or sequence of int, optional
        Frames over which MSD is computed.
    every : int, default=1
        Subsample frames by taking every Nth frame.
    atoms : sequence of int, optional
        Atom indices to include.
    atom_types : sequence of str, optional
        Atom types to include.
    dims : sequence of {"x","y","z"}, default=("x","y","z")
        Coordinate components used for displacement.
    origin : {"first"} or int, default="first"
        Reference frame for displacement.

    Returns
    -------
    pandas.DataFrame
        Long-format table with columns ``frame_index``, ``iter``,
        ``atom_id``, and ``msd``.

    Examples
    --------
    >>> df = get_mean_squared_displacement(xh, atom_types=["O"])
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

def get_radial_dist_fnc(
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
    """Compute radial distribution functions (RDFs) or RDF-derived properties.

    Works on
    --------
    XmoloutHandler — ``xmolout``

    Parameters
    ----------
    xh : XmoloutHandler
        Parsed xmolout handler.
    backend : {"freud", "ovito"}, default="freud"
        RDF backend to use.
    frames : iterable of int, optional
        Frame indices to include.
    types_a, types_b : iterable of str, optional
        Atom types defining a partial RDF.
    r_max : float, optional
        Maximum radius cutoff.
    bins : int, default=200
        Number of RDF bins.
    property : str, optional
        RDF-derived quantity (e.g. ``first_peak``, ``dominant_peak``,
        ``area``, ``excess_area``).
    average : bool, default=True
        Average RDF curves across frames.
    return_stack : bool, default=False
        Return per-frame RDF curves when not averaging.

    Returns
    -------
    numpy.ndarray, tuple, or pandas.DataFrame
        RDF curves or per-frame RDF-derived properties.

    Examples
    --------
    >>> r, g = get_radial_dist_fnc(xh, types_a=["Al"], types_b=["N"])
    >>> df = get_radial_dist_fnc(xh, property="first_peak")
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
