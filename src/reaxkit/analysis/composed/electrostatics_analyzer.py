"""
Electrostatics analysis utilities (dipole and polarization).

This module provides fast, vectorized electrostatics calculations for ReaxFF
simulations, including total and local dipoles, polarization, and
polarization–electric-field hysteresis analysis.

Design notes
------------
- Heavy computations are vectorized with NumPy for performance.
- Coordinates, charges, and connectivity are preloaded once per frame batch.
- Local electrostatics operate on atom-centered clusters using connectivity.
- Polarization volumes can be estimated via convex hull or bounding box.

Typical use cases include:
------------
- computing total dipole or polarization vs time/frame
- computing local dipoles around selected core atom types
- generating polarization–field hysteresis loops
"""


from __future__ import annotations

from typing import Literal, Sequence, Optional, Tuple, Dict, Any, List

import re

import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull

from reaxkit.io.handlers.xmolout_handler import XmoloutHandler
from reaxkit.io.handlers.fort7_handler import Fort7Handler
from reaxkit.io.handlers.fort78_handler import Fort78Handler
from reaxkit.io.handlers.control_handler import ControlHandler

from reaxkit.analysis.per_file.fort7_analyzer import (
    get_partial_charges_conv_fnc,
    get_all_atoms_cnn_conv_fnc,
)
from reaxkit.analysis.per_file.fort78_analyzer import match_electric_field_to_iout2

from reaxkit.utils.constants import const
from reaxkit.utils.numerical.numerical_calcs import find_zero_crossings


Scope = Literal["total", "local"]
Mode = Literal["dipole", "polarization"]
VolumeMethod = Literal["hull", "bbox"]
AggregateKind = Optional[Literal["mean", "max", "min", "last"]]


# -------------------------------------------------------------------------------------
# Volume helpers
# -------------------------------------------------------------------------------------

def _convex_hull_volume(coords: np.ndarray) -> float:
    """Convex-hull volume; NaN if not computable."""
    coords = np.asarray(coords, float)
    if coords.ndim != 2 or coords.shape[1] != 3 or coords.shape[0] < 4:
        return np.nan
    try:
        return float(ConvexHull(coords).volume)
    except Exception:
        return np.nan


def _bbox_volume(coords: np.ndarray) -> float:
    """Axis-aligned bounding box volume; NaN if empty/invalid."""
    coords = np.asarray(coords, float)
    if coords.ndim != 2 or coords.shape[1] != 3 or coords.shape[0] == 0:
        return np.nan
    mn = np.nanmin(coords, axis=0)
    mx = np.nanmax(coords, axis=0)
    if np.any(~np.isfinite(mn)) or np.any(~np.isfinite(mx)):
        return np.nan
    d = mx - mn
    if np.any(d < 0):
        return np.nan
    return float(d[0] * d[1] * d[2])


# -------------------------------------------------------------------------------------
# Core dipole/polarization primitive (single set of coords/charges)
# -------------------------------------------------------------------------------------

def _dipole_and_polarization(
    coords: np.ndarray,
    charges: np.ndarray,
    *,
    mode: Mode = "dipole",
    volume_method: VolumeMethod = "hull",
) -> Tuple[pd.DataFrame, float]:
    """Compute dipole moment and optional polarization for a single atomic cluster.

    Works on
    --------
    Raw arrays — coordinates + charges (no handler required)

    Parameters
    ----------
    coords : (N, 3) ndarray
        Atomic coordinates in Å.
    charges : (N,) ndarray
        Atomic partial charges in units of e.
    mode : {"dipole", "polarization"}, default="dipole"
        Whether to compute dipole only or dipole + polarization.
    volume_method : {"hull", "bbox"}, default="hull"
        Method used to estimate volume for polarization.

    Returns
    -------
    pandas.DataFrame
        One-row table with dipole components (and polarization if requested).
    float
        Estimated volume in Å³ (NaN if not computable).

    Examples
    --------
    >>> df, vol = _dipole_and_polarization(coords, charges, mode="polarization")"""
    coords = np.asarray(coords, float)
    charges = np.asarray(charges, float)

    if coords.shape[0] != charges.shape[0]:
        raise ValueError("coords and charges must have same length")

    mu_ea = (coords * charges[:, None]).sum(axis=0)  # (3,) in e·Å
    mu_debye = mu_ea * const["ea_to_debye"]

    data: Dict[str, List[float]] = {
        "mu_x (debye)": [float(mu_debye[0])],
        "mu_y (debye)": [float(mu_debye[1])],
        "mu_z (debye)": [float(mu_debye[2])],
    }

    volume = np.nan
    if mode == "polarization":
        if volume_method == "bbox":
            volume = _bbox_volume(coords)
        else:
            volume = _convex_hull_volume(coords)

        if np.isfinite(volume) and volume > 0:
            P = mu_ea / volume * const["ea3_to_uC_cm2"]
            data["P_x (uC/cm^2)"] = [float(P[0])]
            data["P_y (uC/cm^2)"] = [float(P[1])]
            data["P_z (uC/cm^2)"] = [float(P[2])]
        else:
            data["P_x (uC/cm^2)"] = [np.nan]
            data["P_y (uC/cm^2)"] = [np.nan]
            data["P_z (uC/cm^2)"] = [np.nan]

    return pd.DataFrame(data), float(volume)


# -------------------------------------------------------------------------------------
# Preload (bulk) electrostatics
# -------------------------------------------------------------------------------------

def _preload_electrostatics(
    xh: XmoloutHandler,
    f7: Fort7Handler,
    *,
    frames: Optional[Sequence[int]] = None,
    every: int = 1,
) -> Dict[str, Any]:
    """Preload coordinates, charges, atom types, and connectivity for multiple frames.

    Works on
    --------
    XmoloutHandler + Fort7Handler — ``xmolout`` + ``fort.7``

    Parameters
    ----------
    xh : XmoloutHandler
        Parsed trajectory handler.
    f7 : Fort7Handler
        Parsed connectivity/charge handler.
    frames : sequence of int, optional
        Frame indices to include. If None, all frames are used.
    every : int, default=1
        Subsampling stride (e.g. every=10 → every 10th frame).

    Returns
    -------
    dict
        Dictionary containing NumPy arrays for:
        ``coords``, ``charges``, ``atom_types``, ``cnn``, ``frame_index``, ``iter``.

    Notes
    -----
    Intended as an internal fast path for electrostatics calculations.
    """
    df_sim = xh.dataframe()
    if df_sim.empty:
        return {
            "frame_index": np.asarray([], dtype=int),
            "iter": np.asarray([], dtype=int),
            "coords": np.zeros((0, 0, 3), dtype=float),
            "atom_types": [],
            "charges": np.zeros((0, 0), dtype=float),
            "cnn": np.zeros((0, 0, 0), dtype=np.int32),
            "cnn_cols": [],
        }

    if frames is None:
        frame_list = list(range(len(df_sim)))
    else:
        frame_list = [int(f) for f in frames]

    frame_list = frame_list[::max(1, int(every))]
    frame_list = [f for f in frame_list if 0 <= f < len(df_sim)]
    if not frame_list:
        return {
            "frame_index": np.asarray([], dtype=int),
            "iter": np.asarray([], dtype=int),
            "coords": np.zeros((0, 0, 3), dtype=float),
            "atom_types": [],
            "charges": np.zeros((0, 0), dtype=float),
            "cnn": np.zeros((0, 0, 0), dtype=np.int32),
            "cnn_cols": [],
        }

    iters = np.asarray(
        [int(df_sim.iloc[fi]["iter"]) if "iter" in df_sim.columns else int(fi) for fi in frame_list],
        dtype=int,
    )

    # --- coords + types from xmolout (per frame) ---
    coords_list: List[np.ndarray] = []
    types_list: List[np.ndarray] = []

    nA: Optional[int] = None
    for fi in frame_list:
        fr = xh.frame(int(fi))
        coords = np.asarray(fr["coords"], dtype=float)
        types = np.asarray([str(t) for t in fr["atom_types"]], dtype=object)
        if nA is None:
            nA = int(coords.shape[0])
        if coords.shape[0] != nA:
            raise ValueError(f"Atom count changes across frames (frame {fi} has {coords.shape[0]} vs {nA}).")
        coords_list.append(coords)
        types_list.append(types)

    coords_arr = np.stack(coords_list, axis=0)  # (nF, nA, 3)

    # --- charges from fort.7 (bulk by iterations) ---
    q_df = get_partial_charges_conv_fnc(f7, iterations=iters.tolist())
    if q_df.empty:
        raise ValueError("No partial charges found in fort.7 for requested iterations.")

    q_df = q_df[q_df["iter"].isin(iters)].copy()
    q_df = q_df.sort_values(["iter", "atom_idx"]).reset_index(drop=True)

    charges_by_iter: Dict[int, np.ndarray] = {}
    for it in iters.tolist():
        sub = q_df[q_df["iter"] == it]
        if sub.empty:
            charges_by_iter[it] = np.full((nA,), np.nan, dtype=float)
            continue
        arr = sub["partial_charge"].to_numpy(dtype=float)
        if arr.shape[0] != nA:
            raise ValueError(f"Charges atom count mismatch at iter={it}: {arr.shape[0]} vs {nA}")
        charges_by_iter[it] = arr

    charges_arr = np.stack([charges_by_iter[int(it)] for it in iters], axis=0)  # (nF, nA)

    # --- connectivity (cnn) from fort.7 (bulk) ---
    cnn_df = get_all_atoms_cnn_conv_fnc(f7, iterations=iters.tolist())
    if cnn_df.empty:
        cnn_arr = np.zeros((len(frame_list), nA, 0), dtype=np.int32)
        cnn_cols: List[str] = []
    else:
        cnn_df = cnn_df[cnn_df["iter"].isin(iters)].copy()
        cnn_cols = [c for c in cnn_df.columns if c.startswith("atom_cnn")]
        cnn_cols = sorted(cnn_cols, key=lambda s: int(re.sub(r"\D+", "", s) or 0))

        cnn_by_iter: Dict[int, np.ndarray] = {}
        for it in iters.tolist():
            sub = cnn_df[cnn_df["iter"] == it].sort_values("atom_idx").reset_index(drop=True)
            if sub.empty:
                cnn_by_iter[it] = np.zeros((nA, len(cnn_cols)), dtype=np.int32)
                continue
            mat = sub[cnn_cols].to_numpy(dtype=np.int32, copy=True)
            if mat.shape[0] != nA:
                raise ValueError(f"Connectivity atom count mismatch at iter={it}: {mat.shape[0]} vs {nA}")
            cnn_by_iter[it] = mat

        cnn_arr = np.stack([cnn_by_iter[int(it)] for it in iters], axis=0)  # (nF, nA, max_cnn)

    return {
        "frame_index": np.asarray(frame_list, dtype=int),
        "iter": iters,
        "coords": coords_arr,
        "atom_types": types_list,
        "charges": charges_arr,
        "cnn": cnn_arr,
        "cnn_cols": cnn_cols,
    }


# -------------------------------------------------------------------------------------
# Fast total/local calculators
# -------------------------------------------------------------------------------------

def _total_dipole_calc(
    coords: np.ndarray,   # (nF, nA, 3)
    charges: np.ndarray,  # (nF, nA)
    *,
    mode: Mode = "dipole",
    volume_method: VolumeMethod = "hull",
) -> pd.DataFrame:
    """Compute total dipole (and optional polarization) over many frames.

    Works on
    --------
    Preloaded NumPy arrays — multiple frames

    Parameters
    ----------
    coords : (nF, nA, 3) ndarray
        Atomic coordinates for each frame.
    charges : (nF, nA) ndarray
        Atomic partial charges for each frame.
    mode : {"dipole", "polarization"}, default="dipole"
        Quantity to compute.
    volume_method : {"hull", "bbox"}, default="hull"
        Volume estimator for polarization.

    Returns
    -------
    pandas.DataFrame
        One row per frame with dipole (and polarization if requested).

    Examples
    --------
    >>> df = _total_dipole_calc(coords, charges, mode="dipole")"""
    coords = np.asarray(coords, float)
    charges = np.asarray(charges, float)
    if coords.ndim != 3 or coords.shape[2] != 3:
        raise ValueError("coords must be (nF, nA, 3)")
    if charges.shape != coords.shape[:2]:
        raise ValueError("charges must be (nF, nA) matching coords")

    mu_ea = (coords * charges[..., None]).sum(axis=1)  # (nF, 3) in e·Å
    mu_debye = mu_ea * const["ea_to_debye"]

    out: Dict[str, Any] = {
        "mu_x (debye)": mu_debye[:, 0],
        "mu_y (debye)": mu_debye[:, 1],
        "mu_z (debye)": mu_debye[:, 2],
    }

    volumes = np.full((coords.shape[0],), np.nan, dtype=float)
    if mode == "polarization":
        if volume_method == "bbox":
            mn = np.min(coords, axis=1)
            mx = np.max(coords, axis=1)
            d = mx - mn
            volumes = d[:, 0] * d[:, 1] * d[:, 2]
        else:
            for i in range(coords.shape[0]):
                volumes[i] = _convex_hull_volume(coords[i])

        P = np.full_like(mu_ea, np.nan, dtype=float)
        good = np.isfinite(volumes) & (volumes > 0)
        if np.any(good):
            P[good] = (mu_ea[good] / volumes[good, None]) * const["ea3_to_uC_cm2"]

        out["P_x (uC/cm^2)"] = P[:, 0]
        out["P_y (uC/cm^2)"] = P[:, 1]
        out["P_z (uC/cm^2)"] = P[:, 2]

    out["volume (angstrom^3)"] = volumes
    return pd.DataFrame(out)


def _local_dipole_calc(
    coords: np.ndarray,      # (nA, 3)
    charges: np.ndarray,     # (nA,)
    atom_types: np.ndarray,  # (nA,)
    cnn_mat: np.ndarray,     # (nA, max_cnn) 1-based neighbors, 0 padded
    *,
    core_types: Sequence[str],
    mode: Mode = "dipole",
    volume_method: VolumeMethod = "bbox",
) -> pd.DataFrame:
    """Compute local dipoles (or polarization) around selected core atoms for one frame.

    Works on
    --------
    Raw arrays — single frame + connectivity

    Parameters
    ----------
    coords : (nA, 3) ndarray
        Atomic coordinates.
    charges : (nA,) ndarray
        Partial charges.
    atom_types : (nA,) ndarray
        Atom type labels.
    cnn_mat : (nA, k) ndarray
        Connectivity matrix (1-based indices, zero-padded).
    core_types : sequence of str
        Atom types treated as cluster centers.
    mode : {"dipole", "polarization"}, default="dipole"
        Quantity to compute.
    volume_method : {"bbox", "hull"}, default="bbox"
        Volume estimator (bbox is faster for local clusters).

    Returns
    -------
    pandas.DataFrame
        One row per core atom with local electrostatic quantities.
    """
    coords = np.asarray(coords, float)
    charges = np.asarray(charges, float)
    atom_types = np.asarray(atom_types)
    cnn_mat = np.asarray(cnn_mat, dtype=np.int32)

    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("coords must be (nA, 3)")
    nA = coords.shape[0]
    if charges.shape != (nA,):
        raise ValueError("charges must be (nA,)")
    if atom_types.shape[0] != nA:
        raise ValueError("atom_types must be (nA,)")
    if cnn_mat.ndim != 2 or cnn_mat.shape[0] != nA:
        raise ValueError("cnn_mat must be (nA, max_cnn)")

    core_set = {str(t) for t in core_types}
    core_mask = np.array([str(t) in core_set for t in atom_types], dtype=bool)
    core_idx0 = np.nonzero(core_mask)[0]  # 0-based
    if core_idx0.size == 0:
        cols = ["core_atom_type", "core_atom_id", "mu_x (debye)", "mu_y (debye)", "mu_z (debye)"]
        if mode == "polarization":
            cols += ["P_x (uC/cm^2)", "P_y (uC/cm^2)", "P_z (uC/cm^2)", "volume (angstrom^3)"]
        return pd.DataFrame(columns=cols)

    neigh_1b = cnn_mat[core_idx0]                     # (n_core, max_cnn)
    neigh0 = neigh_1b.astype(np.int64) - 1           # 0-based, -1 for padded zeros
    neigh0[neigh_1b == 0] = -1

    k = 1 + neigh0.shape[1]
    cluster_idx0 = np.empty((core_idx0.size, k), dtype=np.int64)
    cluster_idx0[:, 0] = core_idx0
    cluster_idx0[:, 1:] = neigh0

    idx_clip = cluster_idx0.copy()
    idx_clip[idx_clip < 0] = 0

    coords_g = coords[idx_clip]            # (n_core, k, 3)
    q_g = charges[idx_clip]               # (n_core, k)
    mask = (cluster_idx0 >= 0)            # (n_core, k)

    q_g = q_g * mask

    n_neigh = mask[:, 1:].sum(axis=1).astype(float)  # (n_core,)
    scale = np.where(n_neigh > 0, 1.0 / n_neigh, 0.0)
    q_g[:, 1:] = q_g[:, 1:] * scale[:, None]

    mu_ea = (coords_g * q_g[..., None]).sum(axis=1)   # (n_core, 3)
    mu_debye = mu_ea * CONSTANTS["ea_to_debye"]

    out: Dict[str, Any] = {
        "core_atom_type": [str(atom_types[i]) for i in core_idx0],
        "core_atom_id": (core_idx0 + 1).astype(int),
        "mu_x (debye)": mu_debye[:, 0],
        "mu_y (debye)": mu_debye[:, 1],
        "mu_z (debye)": mu_debye[:, 2],
    }

    volumes = np.full((core_idx0.size,), np.nan, dtype=float)
    if mode == "polarization":
        if volume_method == "bbox":
            # broadcast-safe masking: (n_core, k, 3)
            cc = np.where(mask[..., None], coords_g, np.nan)
            mn = np.nanmin(cc, axis=1)
            mx = np.nanmax(cc, axis=1)
            d = mx - mn
            volumes = d[:, 0] * d[:, 1] * d[:, 2]
        else:
            for i in range(core_idx0.size):
                pts = coords_g[i][mask[i]]
                volumes[i] = _convex_hull_volume(pts)

        P = np.full_like(mu_ea, np.nan, dtype=float)
        good = np.isfinite(volumes) & (volumes > 0)
        if np.any(good):
            P[good] = (mu_ea[good] / volumes[good, None]) * CONSTANTS["ea3_to_uC_cm2"]

        out["P_x (uC/cm^2)"] = P[:, 0]
        out["P_y (uC/cm^2)"] = P[:, 1]
        out["P_z (uC/cm^2)"] = P[:, 2]
        out["volume (angstrom^3)"] = volumes

    return pd.DataFrame(out)


# -------------------------------------------------------------------------------------
# Public API: over multiple frames (fast)
# -------------------------------------------------------------------------------------

def dipoles_polarizations_over_multiple_frames(
    xh: XmoloutHandler,
    f7: Fort7Handler,
    *,
    scope: Scope = "total",
    core_types: Optional[Sequence[str]] = None,
    mode: Mode = "dipole",
    volume_method: Optional[VolumeMethod] = None,
    frames: Optional[Sequence[int]] = None,
    every: int = 1,
) -> pd.DataFrame:
    """Compute dipoles or polarization over multiple frames (fast path).

    Works on
    --------
    XmoloutHandler + Fort7Handler — ``xmolout`` + ``fort.7``

    Parameters
    ----------
    scope : {"total", "local"}, default="total"
        Whether to compute total system electrostatics or local clusters.
    core_types : sequence of str, optional
        Required when ``scope="local"``.
    mode : {"dipole", "polarization"}, default="dipole"
        Quantity to compute.
    frames : sequence of int, optional
        Frame indices to include.
    every : int, default=1
        Subsampling stride.

    Returns
    -------
    pandas.DataFrame
        Electrostatics results with frame and iteration metadata.

    Examples
    --------
    >>> df = dipoles_polarizations_over_multiple_frames(xh, f7, mode="dipole")
    """
    if scope == "local" and (core_types is None or len(core_types) == 0):
        raise ValueError("core_types must be provided when scope='local'.")

    if volume_method is None:
        volume_method = "bbox" if scope == "local" else "hull"

    pre = _preload_electrostatics(xh, f7, frames=frames, every=every)
    if pre["frame_index"].size == 0:
        return pd.DataFrame()

    fidx = pre["frame_index"]
    iters = pre["iter"]
    coords = pre["coords"]
    charges = pre["charges"]

    if scope == "total":
        df = _total_dipole_calc(coords, charges, mode=mode, volume_method=volume_method)
        df.insert(0, "iter", iters)
        df.insert(0, "frame_index", fidx)
        return df.reset_index(drop=True)

    cnn = pre["cnn"]
    types_list = pre["atom_types"]

    rows: List[pd.DataFrame] = []
    for i in range(len(fidx)):
        df_one = _local_dipole_calc(
            coords[i],
            charges[i],
            types_list[i],
            cnn[i] if cnn.shape[2] > 0 else np.zeros((coords.shape[1], 0), dtype=np.int32),
            core_types=core_types or [],
            mode=mode,
            volume_method=volume_method,
        )
        if df_one.empty:
            continue
        df_one.insert(0, "iter", int(iters[i]))
        df_one.insert(0, "frame_index", int(fidx[i]))
        rows.append(df_one)

    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


# -------------------------------------------------------------------------------------
# Hysteresis: polarization vs electric field
# -------------------------------------------------------------------------------------

def polarization_field_analysis(
    xh: XmoloutHandler,
    f7: Fort7Handler,
    f78: Fort78Handler,
    ctrl: ControlHandler,
    *,
    field_var: str = "field_z",
    aggregate: AggregateKind = None,
    x_variable: str = "field_z",
    y_variable: str = "P_z (uC/cm^2)",
) -> Tuple[pd.DataFrame, pd.DataFrame, list[float], list[float]]:
    """
    Perform polarization–electric-field hysteresis analysis.

    Works on
    --------
    XmoloutHandler + Fort7Handler + Fort78Handler + ControlHandler

    Parameters
    ----------
    field_var : str, default="field_z"
        Electric-field component to use.
    aggregate : {"mean","max","min","last"}, optional
        Aggregation mode over identical field values.
    x_variable, y_variable : str
        Columns used for zero-crossing detection.

    Returns
    -------
    joint_df : pandas.DataFrame
        Polarization and electric field per frame.
    aggregated_df : pandas.DataFrame
        Aggregated hysteresis curve.
    y_zeros : list[float]
        Zero crossings of polarization.
    x_zeros : list[float]
        Zero crossings of electric field.

    Examples
    --------
    >>> joint, agg, p0, e0 = polarization_field_analysis(xh, f7, f78, ctrl)
    """
    pol_df = dipoles_polarizations_over_multiple_frames(
        xh,
        f7,
        scope="total",
        core_types=None,
        mode="polarization",
        volume_method="hull",
    )
    if pol_df.empty:
        raise ValueError("No polarization data produced by electrostatics_over_frames.")
    if "iter" not in pol_df.columns:
        raise KeyError("electrostatics_over_frames output has no 'iter' column.")

    pol_df = pol_df.sort_values("iter").reset_index(drop=True)

    target_iters = pol_df["iter"].to_list()
    series_E = match_electric_field_to_iout2(
        f78,
        ctrl,
        target_iters=target_iters,
        field_var=field_var,
    )

    # align by iter to be robust + convert units
    series_E = series_E.reindex(pol_df["iter"].values) * const["electric_field_VA_to_MVcm"]
    joint = pol_df.copy()
    joint[field_var] = series_E.to_numpy(dtype=float)

    if aggregate is None:
        agg_df = joint.copy()
    else:
        if aggregate not in {"mean", "max", "min", "last"}:
            raise ValueError("aggregate must be one of: mean|max|min|last (or None).")

        g = joint.groupby(field_var, as_index=False)
        if aggregate == "mean":
            agg_df = g.mean(numeric_only=True)
        elif aggregate == "max":
            agg_df = g.max(numeric_only=True)
        elif aggregate == "min":
            agg_df = g.min(numeric_only=True)
        else:
            joint2 = joint.sort_values("iter").reset_index(drop=True)
            agg_df = joint2.groupby(field_var, as_index=False).tail(1).reset_index(drop=True)

    if x_variable not in agg_df.columns or y_variable not in agg_df.columns:
        raise KeyError(f"Missing required columns '{x_variable}' or '{y_variable}' in aggregated data.")

    y_zeros = find_zero_crossings(agg_df[x_variable].to_numpy(float), agg_df[y_variable].to_numpy(float))
    x_zeros = find_zero_crossings(agg_df[y_variable].to_numpy(float), agg_df[x_variable].to_numpy(float))

    return joint, agg_df, y_zeros, x_zeros
