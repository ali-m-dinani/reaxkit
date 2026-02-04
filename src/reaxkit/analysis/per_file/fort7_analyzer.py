"""
fort.7 analysis utilities.

This module provides functions for extracting atom-level and iteration-level
features from a parsed ReaxFF ``fort.7`` file, as well as higher-level
coordination analysis when combined with ``xmolout``.

Typical use cases include:

- extracting per-atom quantities such as partial charges or bond-order sums
- extracting per-iteration (summary) quantities from the simulation table
- classifying atomic coordination (under / proper / over) across frames
"""


from __future__ import annotations
import re
import pandas as pd
from collections.abc import Mapping

from reaxkit.utils.frame_utils import resolve_indices
from typing import Iterable, List, Sequence, Union
from reaxkit.io.handlers.xmolout_handler import XmoloutHandler
from reaxkit.analysis.composed.coordination_analyzer import (
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

def get_fort7_data_per_atom(
    handler,
    columns: Union[str, Sequence[str]],
    frames: Indexish = None,
    iterations: Indexish = None,
    regex: bool = False,
    add_index_cols: bool = True,
) -> pd.DataFrame:
    """Extract per-atom feature columns across selected frames.

    Works on
    --------
    Fort7Handler — ``fort.7``

    Parameters
    ----------
    handler : Fort7Handler
        Parsed ``fort.7`` handler.
    columns : str or sequence of str
        Atom-level column name(s) to extract (e.g. ``partial_charge``,
        ``sum_BOs``). Regex patterns are allowed if ``regex=True``.
    frames : int or iterable of int, optional
        Frame indices to include.
    iterations : int or iterable of int, optional
        Iteration numbers to include.
    regex : bool, default=False
        Whether ``columns`` should be interpreted as regular expressions.
    add_index_cols : bool, default=True
        If True, include ``frame_idx``, ``iter``, and ``atom_idx`` columns.

    Returns
    -------
    pandas.DataFrame
        Tidy table with one row per atom per frame, including requested
        feature columns and index metadata.

    Examples
    --------
    >>> df = get_fort7_data_per_atom(f7, "partial_charge", frames=0)
    >>> df = get_fort7_data_per_atom(f7, r"^atom_cnn\\d+$", regex=True)
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

def get_fort7_data_summaries(
    handler,
    columns: Union[str, Sequence[str]],
    frames: Indexish = None,
    iterations: Indexish = None,
    regex: bool = False,
    add_index_cols: bool = True,
) -> pd.DataFrame:
    """Extract per-iteration (summary) feature columns from a ``fort.7`` file.

    Works on
    --------
    Fort7Handler — ``fort.7``

    Parameters
    ----------
    handler : Fort7Handler
        Parsed ``fort.7`` handler.
    columns : str or sequence of str
        Summary-level column name(s) to extract.
    frames : int or iterable of int, optional
        Frame indices to include.
    iterations : int or iterable of int, optional
        Iteration numbers to include.
    regex : bool, default=False
        Whether ``columns`` should be interpreted as regular expressions.
    add_index_cols : bool, default=True
        If True, include ``frame_idx`` and ``iter`` columns.

    Returns
    -------
    pandas.DataFrame
        Table with one row per selected frame containing summary quantities.

    Examples
    --------
    >>> df = get_fort7_data_summaries(f7, ["num_bonds", "total_BO"])
    >>> df = get_fort7_data_summaries(f7, r"^total_.*$", regex=True)
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

def get_partial_charges_conv_fnc(
    handler,
    frames: Indexish = None,
    iterations: Indexish = None,
) -> pd.DataFrame:
    """
    Convenience function to extract per-atom partial charges across selected frames.

    Works on
    --------
    Fort7Handler — ``fort.7``

    Parameters
    ----------
    handler : Fort7Handler
        Parsed ``fort.7`` handler.
    frames, iterations
        Frame indices or iteration numbers to include.

    Returns
    -------
    pandas.DataFrame
        Per-atom partial charges with frame and atom indices.

    Examples
    --------
    >>> df = get_partial_charges_conv_fnc(f7, frames=0)
    """
    return get_fort7_data_per_atom(handler, "partial_charge", frames=frames, iterations=iterations)


def get_all_atoms_cnn_conv_fnc(
    handler,
    frames: Indexish = None,
    iterations: Indexish = None,
) -> pd.DataFrame:
    """
    Convenience function to extract all ``atom_cnn*`` connectivity columns across selected frames.

    Works on
    --------
    Fort7Handler — ``fort.7``

    Parameters
    ----------
    handler : Fort7Handler
        Parsed ``fort.7`` handler.
    frames, iterations
        Frame indices or iteration numbers to include.

    Returns
    -------
    pandas.DataFrame
        Per-atom connectivity values with frame and atom indices.

    Examples
    --------
    >>> df = get_all_atoms_cnn_conv_fnc(f7)
    """
    return get_fort7_data_per_atom(handler, r"^atom_cnn\d+$", frames=frames, iterations=iterations, regex=True)

def get_sum_bos_conv_fnc(
    handler,
    frames: Indexish = None,
    iterations: Indexish = None,
) -> pd.DataFrame:
    """
    Convenience function to extract per-atom total bond order (``sum_BOs``) across selected frames.

    Works on
    --------
    Fort7Handler — ``fort.7``

    Parameters
    ----------
    handler : Fort7Handler
        Parsed ``fort.7`` handler.
    frames, iterations
        Frame indices or iteration numbers to include.

    Returns
    -------
    pandas.DataFrame
        Tidy table with columns: ``frame_idx``, ``iter``, ``atom_idx``,
        and ``sum_BOs``.

    Examples
    --------
    >>> df = get_sum_bos_conv_fnc(f7)
    """
    return get_fort7_data_per_atom(
        handler,
        columns="sum_BOs",
        frames=frames,
        iterations=iterations,
        regex=False,
        add_index_cols=True,
    )


# --- Coordination classification using features_atom("sum_BOs") --------------
def per_atom_coordination_status_over_frames(
    f7_handler,
    xh: XmoloutHandler,
    *,
    valences: Mapping[str, float],
    threshold: float = 0.9,
    frames: Indexish = None,
    iterations: Indexish = None,
    require_all_valences: bool = True,
) -> pd.DataFrame:
    """Classify atomic coordination status across frames using bond-order sums.

    Works on
    --------
    Fort7Handler + XmoloutHandler — ``fort.7`` + ``xmolout``

    Parameters
    ----------
    f7_handler : Fort7Handler
        Parsed ``fort.7`` handler providing bond-order information.
    xh : XmoloutHandler
        Parsed ``xmolout`` handler providing atom types per frame.
    valences : mapping of str to float
        Reference valence values for each atom type.
    threshold : float, default=0.9
        Tolerance window for under/over-coordination classification.
    frames, iterations
        Frame indices or iteration numbers to include.
    require_all_valences : bool, default=True
        If True, raise an error when a valence is missing for any atom type.

    Returns
    -------
    pandas.DataFrame
        One row per atom per frame with columns including:
        ``frame_index``, ``iter``, ``atom_id``, ``atom_type``,
        ``sum_BOs``, ``valence``, ``delta``, ``status``, ``status_label``.

    Examples
    --------
    >>> df = per_atom_coordination_status_over_frames(
    ...     f7, xh, valences={"C": 4.0, "H": 1.0}
    ... )
    """
    # Pull sum_BOs in tidy form (frame_idx, iter, atom_idx, sum_BOs)
    df_sum = get_sum_bos_conv_fnc(f7_handler, frames=frames, iterations=iterations)
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








