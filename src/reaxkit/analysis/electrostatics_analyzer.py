"""analyzer for electrostatics calculations such as dipole moment calculation for a molecule, slab, etc."""

from typing import Literal, Sequence, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull

from reaxkit.io.xmolout_handler import XmoloutHandler
from reaxkit.io.fort7_handler import Fort7Handler
from reaxkit.io.fort78_handler import Fort78Handler
from reaxkit.io.control_handler import ControlHandler

from reaxkit.analysis.xmolout_analyzer import (
    get_atom_trajectories,
    get_atom_type_mapping,
)
from reaxkit.analysis.fort7_analyzer import (
    get_partial_charges,
    get_all_atom_cnn,
)
from reaxkit.analysis.fort78_analyzer import match_electric_field_to_iout2

from reaxkit.utils.constants import CONSTANTS
from reaxkit.utils.numerical_analysis_utils import _find_zero_crossings


Scope = Literal["total", "local"]
Mode = Literal["dipole", "polarization"]
AggregateKind = Optional[Literal["mean", "max", "min", "last"]]

#-------------------------------------------------------------------------------------
#calculating the dipole moment or polarization
#-------------------------------------------------------------------------------------

def _convex_hull_volume(coords: np.ndarray) -> float:
    """Return convex-hull volume or NaN if not computable."""
    coords = np.asarray(coords, float)
    if coords.shape[0] < 4:
        return np.nan
    try:
        return float(ConvexHull(coords).volume)
    except Exception:
        return np.nan

def dipole_and_polarization(
    coords: np.ndarray,
    charges: np.ndarray,
    *,
    mode: Mode = "dipole",
) -> Tuple[pd.DataFrame, float]:
    """
    Compute dipole (always) and polarization (optionally) using convex-hull volume.

    Parameters
    ----------
    coords : (N, 3)
        Atom coordinates.
    charges : (N,)
        Partial charges.
    mode : {"dipole", "polarization"}
        - "dipole": return dipole only.
        - "polarization": return dipole + polarization.

    Returns
    -------
    df : pandas.DataFrame
        Columns always include:
            "mu_x", "mu_y", "mu_z"
        If polarization is requested:
            "P_x", "P_y", "P_z"
    volume : float
        Convex hull volume used.
    """
    coords = np.asarray(coords, float)
    charges = np.asarray(charges, float)

    if coords.shape[0] != charges.shape[0]:
        raise ValueError("coords and charges must have same length")

    # --- dipole moment μ = Σ qᵢ rᵢ ---
    mu = (coords * charges[:, None]).sum(axis=0)  # shape (3,)
    mu_x, mu_y, mu_z = mu*CONSTANTS['ea_to_debye']

    # DataFrame always includes dipole
    data = {
        "mu_x (debye)": [mu_x],
        "mu_y (debye)": [mu_y],
        "mu_z (debye)": [mu_z],
    }

    # --- volume and polarization ---
    volume = _convex_hull_volume(coords)

    if mode == "polarization":
        if np.isfinite(volume) and volume > 0:
            P = mu / volume * CONSTANTS['ea3_to_uC_cm2']
            data["P_x (uC/cm^2)"] = [P[0]]
            data["P_y (uC/cm^2)"] = [P[1]]
            data["P_z (uC/cm^2)"] = [P[2]]
        else:
            data["P_x (uC/cm^2)"] = [np.nan]
            data["P_y (uC/cm^2)"] = [np.nan]
            data["P_z (uC/cm^2)"] = [np.nan]

    df = pd.DataFrame(data)
    return df, volume

def single_frame_dipoles_polarizations(
    xh: XmoloutHandler,
    f7: Fort7Handler,
    *,
    frame: int = 0,                 # xmolout frame index (0-based)
    scope: Scope = "total",
    core_types: Optional[Sequence[str]] = None,
    mode: Mode = "polarization",
) -> pd.DataFrame:
    """
    Compute dipole/polarization for one frame, using analyzers:

      - coords from xmolout_analyzer.get_atom_trajectories
      - charges (and cnn) from fort7_analyzer.partial_charges / all_atom_cnn

    scope = "total"  -> whole system
    scope = "local"  -> per core atom (e.g. Zn, Mg) + its neighbors

    mode  = "dipole" or "polarization" (passed to dipole_and_polarization)
    """

    frame = int(frame)

    # --- 1) get iteration for this xmolout frame ---
    df_sim = xh.dataframe()
    if df_sim.empty:
        raise ValueError("XmoloutHandler dataframe is empty.")
    if frame < 0 or frame >= len(df_sim):
        raise IndexError(f"Requested frame {frame} is out of range (0..{len(df_sim)-1}).")

    iter_val = int(df_sim.iloc[frame]["iter"]) if "iter" in df_sim.columns else frame

    # --- 2) coordinates & atom types from get_atom_trajectories ---
    traj = get_atom_trajectories(
        xh,
        frames=[frame],
        every=1,
        atoms=None,
        atom_types=None,
        dims=("x", "y", "z"),
        format="long",
    )
    if traj.empty:
        raise ValueError(f"No trajectory data returned for frame {frame}.")

    traj = traj.sort_values(["frame_index", "atom_id"]).reset_index(drop=True)
    coords_all = traj[["x", "y", "z"]].to_numpy(dtype=float)  # (N, 3)
    atom_types = traj["atom_type"].astype(str).tolist()

    # --- 3) charges from fort7_analyzer.partial_charges (matched by iteration) ---
    q_df = get_partial_charges(f7, iterations=[iter_val])
    if q_df.empty:
        raise ValueError(f"No partial charges found in fort.7 for iter={iter_val}.")

    # One frame for this iter → sort by atom_idx (0-based)
    q_df = q_df[q_df["iter"] == iter_val].sort_values("atom_idx").reset_index(drop=True)
    charges_all = q_df["partial_charge"].to_numpy(dtype=float)

    if len(charges_all) != len(coords_all):
        raise ValueError(
            f"Atom count mismatch at iter {iter_val}: "
            f"coords({len(coords_all)}) vs charges({len(charges_all)})"
        )

    # ================== TOTAL MODE ==================
    if scope == "total":
        df_muP, volume = dipole_and_polarization(
            coords_all,
            charges_all,
            mode=mode,
        )
        df_muP["volume (angstrom^3)"] = volume
        return df_muP

    # ================== LOCAL MODE ==================
    if core_types is None or len(core_types) == 0:
        raise ValueError("core_types must be provided when scope='local'.")

    # mapping: which atoms are Zn, Mg, etc (1-based ids)
    mapping = get_atom_type_mapping(xh, frame=frame)
    type_to_indices = mapping["type_to_indices"]  # dict[type -> list of atom_ids (1-based)]

    core_atom_records: list[tuple[int, str]] = []
    for t in core_types:
        t_str = str(t)
        if t_str in type_to_indices:
            for atom_id in type_to_indices[t_str]:
                core_atom_records.append((atom_id, t_str))

    if not core_atom_records:
        cols = ["core_atom_type", "core_atom_id", "mu_x", "mu_y", "mu_z"]
        if mode == "polarization":
            cols += ["P_x", "P_y", "P_z"]
        cols.append("volume")
        return pd.DataFrame(columns=cols)

    # connectivity from fort7_analyzer.all_atom_cnn (also matched by iteration)
    cnn_df = get_all_atom_cnn(f7, iterations=[iter_val])
    if cnn_df.empty:
        raise ValueError(f"No connectivity (atom_cnn*) data found in fort.7 for iter={iter_val}.")

    cnn_df = cnn_df[cnn_df["iter"] == iter_val].copy()
    cnn_cols = [c for c in cnn_df.columns if c.startswith("atom_cnn")]

    rows = []
    for atom_id, atype in core_atom_records:
        atom_idx0 = atom_id - 1  # 0-based index
        row_cnn = cnn_df[cnn_df["atom_idx"] == atom_idx0]
        if row_cnn.empty:
            # no cnn info; treat core atom alone
            cluster_ids_1based = [atom_id]
        else:
            row_cnn = row_cnn.iloc[0]
            neighbors: list[int] = []
            for c in cnn_cols:
                val = row_cnn[c]
                if pd.isna(val):
                    continue
                v_int = int(val)
                if v_int != 0:
                    neighbors.append(v_int)
            cluster_ids_1based = [atom_id] + neighbors

        cluster_idx0 = [i - 1 for i in cluster_ids_1based]
        coords = coords_all[cluster_idx0, :]
        # charges for neighbors should be averaged
        charges = charges_all[cluster_idx0].astype(float)
        N = len(charges) - 1
        if N > 0:
            charges[1:] /= N  # average the neighbors

        df_muP, volume = dipole_and_polarization(coords, charges, mode=mode)
        df_muP["volume (angstrom^3)"] = volume
        df_muP.insert(0, "core_atom_id", atom_id)
        df_muP.insert(0, "core_atom_type", atype)
        rows.append(df_muP)

    out = pd.concat(rows, ignore_index=True)
    return out


def dipoles_polarizations_over_multiple_frames(
    xh: XmoloutHandler,
    f7: Fort7Handler,
    *,
    scope: Scope = "total",
    core_types: Optional[Sequence[str]] = None,
    mode: Mode = "polarization",
) -> pd.DataFrame:
    """
    Run frame_electrostatics over all frames in the simulation and
    return a single stacked DataFrame.

    - scope = "total" → one row per frame (global dipole/polarization)
    - scope = "local" → one or more rows per frame (one per core atom)

    Columns added:
      - frame_index
      - iter
    """
    df_sim = xh.dataframe()
    if df_sim.empty:
        return pd.DataFrame()

    rows = []
    for fi in range(len(df_sim)):
        iter_val = int(df_sim.iloc[fi]["iter"]) if "iter" in df_sim.columns else fi

        df_one = single_frame_dipoles_polarizations(
            xh,
            f7,
            frame=fi,
            scope=scope,
            core_types=core_types,
            mode=mode,
        )
        if df_one.empty:
            continue

        df_one.insert(0, "iter", iter_val)
        df_one.insert(0, "frame_index", fi)
        rows.append(df_one)

    if not rows:
        return pd.DataFrame()

    return pd.concat(rows, ignore_index=True)

#-------------------------------------------------------------------------------------
#getting the hysteresis data, which is polarization vs electric field
#-------------------------------------------------------------------------------------

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
    1) Compute total polarization over frames (electrostatics_over_frames).
    2) Match electric field from fort.78 to those iters (match_electric_field_to_iout2).
    3) Build joint DataFrame of polarization + E-field.
    4) Optionally aggregate rows with identical E-field values using mean/max/min/last.
    5) Find zero crossings for:
         - y(x) == 0   (y vs x)
         - x(y) == 0   (x vs y)
       using _find_zero_crossings.

    Returns
    -------
    full_df : pd.DataFrame
        Per-frame joint data (polarization + field).
    agg_df : pd.DataFrame
        Aggregated data (or a copy of full_df if aggregate is None).
    y_zeros : list[float]
        x-values where y(x) crosses zero.
    x_zeros : list[float]
        y-values where x(y) crosses zero.
    """
    # ------------------------------------------------------------------
    # 1) total polarization over frames
    # ------------------------------------------------------------------
    pol_df = dipoles_polarizations_over_multiple_frames(
        xh,
        f7,
        scope="total",
        core_types=None,
        mode="polarization",
    )
    if pol_df.empty:
        raise ValueError("No polarization data produced by electrostatics_over_frames.")

    if "iter" not in pol_df.columns:
        raise KeyError("electrostatics_over_frames output has no 'iter' column.")

    pol_df = pol_df.sort_values("iter").reset_index(drop=True)

    # ------------------------------------------------------------------
    # 2) match electric field to these iters
    # ------------------------------------------------------------------
    target_iters = pol_df["iter"].to_list()
    series_E = match_electric_field_to_iout2(
        f78,
        ctrl,
        target_iters=target_iters,
        field_var=field_var,
    )

    # align by iter to be robust
    series_E = series_E.reindex(pol_df["iter"].values)*CONSTANTS['electric_field_VA_to_MVcm']
    full_df = pol_df.copy()
    full_df[field_var] = series_E.to_numpy(dtype=float)

    # ------------------------------------------------------------------
    # 3–4) aggregation over identical *consecutive* E-field (if requested)
    # ------------------------------------------------------------------
    if aggregate is None:
        agg_df = full_df.copy()
    else:
        df = full_df.copy()

        # ensure sorted by iter so "consecutive" is well-defined
        if "iter" in df.columns:
            df = df.sort_values("iter").reset_index(drop=True)

        # define groups of consecutive identical field values
        # every time field_var changes, we start a new group
        group_id = (df[field_var].diff().fillna(0) != 0).cumsum()

        if aggregate in ("mean", "max", "min"):
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            agg_func = aggregate  # 'mean' | 'max' | 'min'

            # aggregate all numeric columns over each consecutive block
            agg_map = {col: agg_func for col in numeric_cols}
            agg_df = df.groupby(group_id, as_index=False).agg(agg_map)

            # field_var is constant within each group; keep it explicitly
            agg_df[field_var] = (
                df.groupby(group_id)[field_var].first().values
            )

        elif aggregate == "last":
            # take the last row of each consecutive block of equal field
            agg_df = (
                df.groupby(group_id, as_index=False)
                .tail(1)
                .reset_index(drop=True)
            )
        else:
            raise ValueError(f"Unknown aggregate kind: {aggregate!r}")

    # ------------------------------------------------------------------
    # 5) zero crossings for chosen x_variable, y_variable
    # ------------------------------------------------------------------
    if x_variable not in agg_df.columns:
        raise KeyError(f"x_variable '{x_variable}' not found in DataFrame columns.")
    if y_variable not in agg_df.columns:
        raise KeyError(f"y_variable '{y_variable}' not found in DataFrame columns.")

    # drop NaNs for root-finding
    mask = ~(agg_df[x_variable].isna() | agg_df[y_variable].isna())
    x_vals = agg_df.loc[mask, x_variable].to_numpy(dtype=float)
    y_vals = agg_df.loc[mask, y_variable].to_numpy(dtype=float)

    # y(x) = 0 which is the coercive field
    coercive_fields = _find_zero_crossings(x_vals, y_vals)

    # x(y) = 0 which is the remnant polarizations
    remnant_polarizations = _find_zero_crossings(y_vals, x_vals)

    return full_df, agg_df, coercive_fields, remnant_polarizations




