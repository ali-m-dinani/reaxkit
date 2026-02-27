"""Structured xmolout extraction helpers."""

from __future__ import annotations

from typing import Any, Optional, Sequence, Union

import numpy as np
import pandas as pd

from reaxkit.core.frame_utils import select_frames as _df_select
from reaxkit.engine.reaxff.io.xmolout_handler import XmoloutHandler

FrameSel = Optional[Union[Sequence[int], range, slice]]
AtomSel = Optional[Union[Sequence[int], slice]]

BASE_ATOM_COLS = ("atom_type", "x", "y", "z")


def _frame_table(xh: XmoloutHandler, i: int) -> pd.DataFrame:
    """Return a DataFrame for a specific frame index from an XmoloutHandler."""
    if hasattr(xh, "_frames") and i < len(xh._frames):
        return xh._frames[i]
    fr = xh.frame(i)
    return pd.DataFrame(
        {
            "atom_type": fr["atom_types"],
            "x": fr["coords"][:, 0],
            "y": fr["coords"][:, 1],
            "z": fr["coords"][:, 2],
        }
    )


def extract_xmolout_data_per_atom(
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
    """Extract per-atom xmolout properties across selected frames."""
    df_sim = xh.dataframe()
    sub_df = _df_select(df_sim, frames)
    fidx_all = list(sub_df.index)[:: max(1, int(every))]

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
                "atom_id": int(j) + 1,
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


__all__ = ["extract_xmolout_data_per_atom"]
