"""
XMOL trajectory file generators.

This module provides utilities for generating new ReaxFF ``xmolout`` files
from in-memory trajectory data or from an existing ``XmoloutHandler``.
Generated files are fully compatible with downstream ReaxKit analyses
(e.g., coordination, connectivity, and visualization workflows).

Typical use cases include:

- writing a filtered or reduced ``xmolout`` from an existing trajectory
- exporting selected frames or atoms for focused analysis
- generating synthetic or post-processed trajectories with extra per-atom fields
"""


from __future__ import annotations
from pathlib import Path
from typing import Iterable, Optional, Sequence, Union, Dict, Any, List
import numpy as np
import pandas as pd
from reaxkit.io.handlers.xmolout_handler import XmoloutHandler

FrameSel = Optional[Union[Sequence[int], range, slice]]
AtomSel  = Optional[Union[Sequence[int], slice]]

def _normalize_frames_for_write(xh: XmoloutHandler, frames: FrameSel) -> list[int]:
    n = xh.n_frames()
    if frames is None:
        return list(range(n))
    if isinstance(frames, slice):
        return list(range(*frames.indices(n)))
    return [int(i) for i in frames if 0 <= int(i) < n]

def _normalize_atoms_for_write(frame_dict: Dict[str, Any],
                               atoms: AtomSel,
                               atom_types: Optional[Sequence[str]]) -> list[int]:
    n_atoms = frame_dict["coords"].shape[0]
    if atoms is not None:
        if isinstance(atoms, slice):
            return list(range(*atoms.indices(n_atoms)))
        return [int(a) for a in atoms if 0 <= int(a) < n_atoms]
    if atom_types:
        tset = {str(t) for t in atom_types}
        return [j for j, t in enumerate(frame_dict["atom_types"]) if str(t) in tset]
    return list(range(n_atoms))

def _format_header_line(sim_name: str,
                        iter: int,
                        E_pot: float,
                        a: float, b: float, c: float,
                        alpha: float, beta: float, gamma: float,
                        prec: int) -> str:
    f = f"{{:.{prec}f}}"
    return (
        f"{sim_name} {iter} "
        f"{f.format(E_pot)} {f.format(a)} {f.format(b)} {f.format(c)} "
        f"{f.format(alpha)} {f.format(beta)} {f.format(gamma)}\n"
    )

def _safe_get(row: pd.Series, key: str, default: float = 0.0) -> float:
    return float(row[key]) if (isinstance(row, pd.Series) and key in row and pd.notna(row[key])) else float(default)

def _get_frame_table(xh: XmoloutHandler, i: int) -> pd.DataFrame:
    """
    Access the per-frame atom table (including any extra columns).
    Falls back to a minimal table if not available.
    """
    if hasattr(xh, "_frames") and i < len(xh._frames):
        return xh._frames[i]
    # minimal fallback (no extras)
    fr = xh.frame(i)
    df = pd.DataFrame({
        "atom_type": fr["atom_types"],
        "x": fr["coords"][:, 0], "y": fr["coords"][:, 1], "z": fr["coords"][:, 2],
    })
    return df

def _format_atom_line_extended(row: pd.Series, prec: int, extra_order: list[str] | None) -> str:
    """
    Format one atom line: type, x, y, z, then (optional) extras in the given order.
    """
    f = f"{{:.{prec}f}}"
    t = str(row["atom_type"])
    x, y, z = float(row["x"]), float(row["y"]), float(row["z"])
    parts = [f"{t:<3}", f.format(x), f.format(y), f.format(z)]
    if extra_order:
        for col in extra_order:
            val = row.get(col, np.nan)
            # Allow strings for label-like extras, but prefer floats when possible
            if isinstance(val, (int, float, np.floating, np.integer)) or pd.isna(val):
                parts.append(f.format(float(val)) if pd.notna(val) else "nan")
            else:
                parts.append(str(val))
    return " ".join(parts) + "\n"

def write_xmolout_from_handler(
    xh: XmoloutHandler,
    out_path: str | Path,
    *,
    frames: FrameSel = None,
    atoms: AtomSel = None,
    atom_types: Optional[Sequence[str]] = None,
    simulation_name: Optional[str] = None,
    precision: int = 6,
    include_extras: Union[bool, Sequence[str], str] = False,
) -> Path:
    """
    Write a filtered xmolout file from an existing XmoloutHandler.

    Works on
    --------
    XmoloutHandler — ``xmolout``

    Parameters
    ----------
    xh : XmoloutHandler
        Parsed trajectory handler providing frame and metadata access.
    out_path : str or pathlib.Path
        Output path for the generated xmolout file.
    frames : sequence of int, range, slice, or None, optional
        Frame indices to include. If None, all frames are written.
    atoms : sequence of int, slice, or None, optional
        Atom indices to include per frame.
    atom_types : sequence of str or None, optional
        Atom types to include per frame (alternative to ``atoms``).
    simulation_name : str or None, optional
        Simulation name written to the xmolout header line.
    precision : int, optional
        Decimal precision for floating-point values.
    include_extras : bool, str, or sequence of str, optional
        Control writing of extra per-atom columns:
        - False: write only atom type and coordinates
        - True or "all": write all extra columns
        - sequence: write only the specified columns

    Returns
    -------
    pathlib.Path
        Path to the written xmolout file.

    Examples
    --------
    >>> xh = XmoloutHandler("xmolout")
    >>> write_xmolout_from_handler(
    ...     xh,
    ...     "xmolout_filtered",
    ...     frames=range(0, 100),
    ...     atom_types=["Al", "N"],
    ... )
    """
    out_path = Path(out_path)
    df = xh.dataframe()
    sim_name = simulation_name or getattr(xh, "simulation_name", None) or "MD"
    fidx = _normalize_frames_for_write(xh, frames)

    with out_path.open("w", encoding="utf-8") as fh:
        for i in fidx:
            fr = xh.frame(i)  # {"iter", "coords"(n,3), "atom_types"[n]}
            sel = _normalize_atoms_for_write(fr, atoms, atom_types)

            # Resolve header values
            row   = df.iloc[i] if (i < len(df)) else pd.Series()
            it    = int(fr.get("iter", int(row["iter"]) if "iter" in row else i))
            E     = _safe_get(row, "E_pot", 0.0)
            a     = _safe_get(row, "a", 1.0)
            b     = _safe_get(row, "b", 1.0)
            c     = _safe_get(row, "c", 1.0)
            alpha = _safe_get(row, "alpha", 90.0)
            beta  = _safe_get(row, "beta", 90.0)
            gamma = _safe_get(row, "gamma", 90.0)

            # 1) number-of-atoms line
            fh.write(f"{len(sel)}\n")
            # 2) header line
            fh.write(_format_header_line(sim_name, it, E, a, b, c, alpha, beta, gamma, precision))

            # 3) atom lines (with optional extras)
            frame_tbl = _get_frame_table(xh, i).reset_index(drop=True)

            base_cols = ["atom_type", "x", "y", "z"]
            present_cols = frame_tbl.columns.tolist()
            extra_cols_default = [c for c in present_cols if c not in base_cols]

            if include_extras is True or (isinstance(include_extras, str) and include_extras.lower() == "all"):
                extra_order = extra_cols_default
            elif isinstance(include_extras, (list, tuple)):
                extra_order = [c for c in include_extras if c in present_cols and c not in base_cols]
            else:
                extra_order = []

            # select atoms and write
            # Ensure required base columns exist
            need = base_cols + ([c for c in extra_order if c not in base_cols])
            sub = frame_tbl.loc[sel, [c for c in need if c in frame_tbl.columns]].copy()

            # If any requested extra is missing in a particular frame, add as NaN
            for c in need:
                if c not in sub.columns:
                    sub[c] = np.nan

            for _, r in sub[base_cols + extra_order].iterrows():
                fh.write(_format_atom_line_extended(r, precision, extra_order))

    return out_path

def write_xmolout_from_frames(
    frames: Iterable[Dict[str, Any]],
    out_path: str | Path,
    *,
    simulation_name: str = "MD",
    precision: int = 6,
    defaults: Dict[str, float] | None = None,
) -> Path:
    """
    Write an xmolout file from explicit per-frame dictionaries.

    Works on
    --------
    In-memory frame dictionaries → ``xmolout``

    Parameters
    ----------
    frames : iterable of dict
        Frame dictionaries with required keys:
        ``iter``, ``coords``, ``atom_types``.
        Optional keys include:
        ``E_pot``, ``a``, ``b``, ``c``, ``alpha``, ``beta``, ``gamma``,
        and ``extras`` for per-atom additional columns.
    out_path : str or pathlib.Path
        Output path for the generated xmolout file.
    simulation_name : str, optional
        Simulation name written to the xmolout header line.
    precision : int, optional
        Decimal precision for floating-point values.
    defaults : dict or None, optional
        Default header values used when missing from a frame.

    Returns
    -------
    pathlib.Path
        Path to the written xmolout file.

    Examples
    --------
    >>> frames = [{
    ...     "iter": 0,
    ...     "coords": [[0,0,0], [1,0,0]],
    ...     "atom_types": ["H", "H"],
    ... }]
    >>> write_xmolout_from_frames(frames, "xmolout_test")
    """
    out_path = Path(out_path)
    defaults = defaults or {
        "E_pot": 0.0,
        "a": 1.0, "b": 1.0, "c": 1.0,
        "alpha": 90.0, "beta": 90.0, "gamma": 90.0,
    }

    with out_path.open("w", encoding="utf-8") as fh:
        for fr in frames:
            coords = np.asarray(fr["coords"], dtype=float)
            if coords.ndim != 2 or coords.shape[1] != 3:
                raise ValueError("Each frame['coords'] must be an (n,3) array-like.")
            types  = [str(t) for t in fr["atom_types"]]
            if len(types) != coords.shape[0]:
                raise ValueError("Length of frame['atom_types'] must match number of coordinate rows.")

            it    = int(fr["iter"])
            E     = float(fr.get("E_pot", defaults["E_pot"]))
            a     = float(fr.get("a", defaults["a"]))
            b     = float(fr.get("b", defaults["b"]))
            c     = float(fr.get("c", defaults["c"]))
            alpha = float(fr.get("alpha", defaults["alpha"]))
            beta  = float(fr.get("beta", defaults["beta"]))
            gamma = float(fr.get("gamma", defaults["gamma"]))

            # Prepare extras (if any)
            extras_dict: Dict[str, Any] = dict(fr.get("extras", {}))
            extra_keys: List[str] = list(extras_dict.keys())
            extras_cols: List[np.ndarray] = []
            for k in extra_keys:
                arr = np.asarray(extras_dict[k])
                if arr.shape[0] != coords.shape[0]:
                    raise ValueError(f"extras['{k}'] length must match number of atoms.")
                extras_cols.append(arr)

            # 1) number-of-atoms line
            fh.write(f"{coords.shape[0]}\n")
            # 2) header line
            fh.write(_format_header_line(simulation_name, it, E, a, b, c, alpha, beta, gamma, precision))
            # 3) atom lines
            f = f"{{:.{precision}f}}"
            for idx, (t, (x, y, z)) in enumerate(zip(types, coords)):
                parts = [f"{t:<3}", f.format(x), f.format(y), f.format(z)]
                for col in extras_cols:
                    v = col[idx]
                    if isinstance(v, (int, float, np.floating, np.integer)) or (isinstance(v, str) and v.replace('.','',1).isdigit()):
                        parts.append(f.format(float(v)))
                    else:
                        parts.append(str(v))
                fh.write(" ".join(parts) + "\n")

    return out_path
