"""
Coordination (valence satisfaction) analysis utilities.

This module provides functions for classifying atomic coordination in a frame
by comparing each atom's total bond order (typically from ``fort.7``) to its
expected valence. The resulting labels help identify under- and over-coordinated
sites (e.g., dangling bonds, over-saturated atoms, and coordination defects).

Typical use cases include:

- classifying atoms as under / well / over coordinated for a single frame
- attaching a numeric status code and a human-readable label
- using coordination labels as defect indicators in post-processing
"""


from __future__ import annotations
from typing import Mapping, Sequence, Optional, Literal
import numpy as np
import pandas as pd

Status = Literal[-1, 0, +1]  # under, coordinated, over

def classify_coordination_for_frame(
    *,
    sum_bos: Sequence[float],
    atom_types: Sequence[str],
    valences: Mapping[str, float],
    threshold: float = 0.3,
    require_all_valences: bool = True,
) -> pd.DataFrame:
    """ Classify atoms in a single frame as under-, well-, or over-coordinated.

    Classification is based on ``delta = sum_bos - valence`` and a tolerance
    threshold:

    - ``status = -1``: under-coordinated (``delta < -threshold``)
    - ``status =  0``: well-coordinated (``|delta| <= threshold``)
    - ``status = +1``: over-coordinated (``delta > threshold``)

    Works on
    --------
    Fort.7-derived arrays — ``sum_BOs`` + atom types (typically from ``fort.7`` and ``xmolout``)

    Parameters
    ----------
    sum_bos : sequence of float
        Per-atom total bond order values (e.g., the ``sum_BOs`` column from fort.7).
    atom_types : sequence of str
        Per-atom type symbols corresponding to ``sum_bos`` (e.g., ``["Al", "N", ...]``).
    valences : mapping of str to float
        Expected valence for each atom type (e.g., ``{"Al": 3.0, "N": 3.0}``).
    threshold : float, default=0.3
        Tolerance used for deciding under/over coordination.
    require_all_valences : bool, default=True
        If True, raise an error if any atom type is missing from ``valences``.

    Returns
    -------
    pandas.DataFrame
        Per-atom classification table with columns:
        ``atom_id``, ``atom_type``, ``sum_BOs``, ``valence``, ``delta``, ``status``.

    Examples
    --------
    >>> sum_bos = [2.8, 3.1, 0.9]
    >>> atom_types = ["Al", "Al", "H"]
    >>> valences = {"Al": 3.0, "H": 1.0}
    >>> df = classify_coordination_for_frame(
    ...     sum_bos=sum_bos, atom_types=atom_types, valences=valences, threshold=0.3
    ... )
    """
    sum_bos = np.asarray(sum_bos, dtype=float)
    types = np.asarray(atom_types, dtype=object)
    if sum_bos.shape[0] != types.shape[0]:
        raise ValueError(f"Length mismatch: sum_bos({sum_bos.shape[0]}) vs atom_types({types.shape[0]})")

    val_arr = np.empty_like(sum_bos, dtype=float)
    missing: list[str] = []
    for i, t in enumerate(types):
        if t in valences:
            val_arr[i] = float(valences[t])
        else:
            missing.append(str(t))
            val_arr[i] = np.nan

    if missing and require_all_valences:
        uniq = ", ".join(sorted(set(missing)))
        raise KeyError(f"Missing valence(s) for atom types: {uniq}")

    delta = sum_bos - val_arr
    status = np.full(sum_bos.shape, np.nan)
    if np.isfinite(threshold) and threshold >= 0:
        status = np.where(delta < -threshold, -1, status)
        status = np.where(np.abs(delta) <= threshold, 0, status)
        status = np.where(delta > threshold, +1, status)

    out = pd.DataFrame({
        "atom_id": np.arange(1, len(sum_bos) + 1, dtype=int),  # 1-based
        "atom_type": types.astype(str),
        "sum_BOs": sum_bos.astype(float),
        "valence": val_arr.astype(float),
        "delta": delta.astype(float),
        "status": status.astype(float),
    })
    if out["status"].notna().all():
        out["status"] = out["status"].astype(int)
    return out

def status_label(series: Sequence[int | float]) -> list[Optional[str]]:
    """Convert numeric coordination status codes into human-readable labels.

    Mapping:
    - ``-1`` → ``"under"``
    - `` 0`` → ``"coord"``
    - ``+1`` → ``"over"``
    - ``NaN`` → ``None``

    Works on
    --------
    Coordination status arrays — output from :func:`classify_coordination_for_frame`

    Parameters
    ----------
    series : sequence of int or float
        Numeric status values (``-1``, ``0``, ``+1``), optionally containing NaNs.

    Returns
    -------
    list[str | None]
        Coordination labels in the same order as input.

    Examples
    --------
    >>> labels = status_label([-1, 0, 1, float("nan")])
    """
    labels: list[Optional[str]] = []
    for v in series:
        if pd.isna(v):
            labels.append(None)
        else:
            vi = int(v)
            labels.append("under" if vi == -1 else ("coord" if vi == 0 else "over"))
    return labels
