"""coordination analyzer based on fort.7 data

Coordination analysis evaluates how well each atom in a frame satisfies its
expected valence by comparing its total bond order (from fort.7 data) to its
nominal valence. This module computes per-atom coordination states—under-,
well-, or over-coordinated—based on the difference between summed bond orders
and ideal valence values. The resulting labels help identify defects,
dangling bonds, over-saturated sites, and other chemically relevant
coordination anomalies in ReaxFF simulations.

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
    """label (i.e., classify) atoms based on their coordination to under- or over-coordinated.
    Classify atoms in a single simulation frame as **under-coordinated**,
    **well-coordinated**, or **over-coordinated** relative to their expected valence.

    This function compares the per-atom total bond order (`sum_bos`) to the
    nominal valence assigned to each atom type. The difference (`delta = sum_bos - valence`)
    is used to assign a coordination **status**:

    - `status = -1`: under-coordinated (`delta < -threshold`)
    - `status =  0`: well-coordinated (`|delta| ≤ threshold`)
    - `status = +1`: over-coordinated (`delta > threshold`)

    The classification helps identify atoms deviating from their ideal bonding
    environment in a given frame, such as dangling bonds or over-saturated sites.

    Notes
    -----
    - Operates on a **single frame** (no time dimension).
    - Setting `require_all_valences=False` allows partial evaluation if some
      atom types have no known valence.
    - Useful for detecting coordination defects or reaction intermediates
      in ReaxFF simulations.
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
    """Convert numeric coordination status codes (-1, 0, and +1) into descriptive string labels (under- or over-coordinated).

    Maps each value in `series` as follows:
      - `-1` → "under"  (under-coordinated)
      - ` 0` → "coord"  (well-coordinated)
      - `+1` → "over"   (over-coordinated)
      - `NaN` → None

    Useful for translating numeric coordination states (e.g., from
    `classify_coordination_for_frame`) into human-readable labels.
    """
    labels: list[Optional[str]] = []
    for v in series:
        if pd.isna(v):
            labels.append(None)
        else:
            vi = int(v)
            labels.append("under" if vi == -1 else ("coord" if vi == 0 else "over"))
    return labels
