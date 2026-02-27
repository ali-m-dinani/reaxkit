"""Structured fort.7 extraction helpers."""

from __future__ import annotations

import re
from typing import Iterable, List, Sequence, Union

import pandas as pd

from reaxkit.core.frame_utils import resolve_indices

Indexish = Union[int, Iterable[int], None]


def _resolve_columns(
    df_example: pd.DataFrame,
    columns: Union[str, Sequence[str]],
    regex: bool = False,
    must_exist: bool = True,
) -> List[str]:
    """Resolve requested column specs to concrete DataFrame column names."""
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
                pass
            else:
                raise KeyError(f"Column '{spec}' not found. Example columns: {cols[:10]} ...")

    out, seen = [], set()
    for col in resolved:
        if col not in seen:
            out.append(col)
            seen.add(col)

    if must_exist and not out:
        raise KeyError("No columns matched the request.")
    return out


def extract_fort7_data_per_atom(
    handler,
    columns: Union[str, Sequence[str]],
    frames: Indexish = None,
    iterations: Indexish = None,
    regex: bool = False,
    add_index_cols: bool = True,
) -> pd.DataFrame:
    """Extract per-atom feature columns across selected fort.7 frames."""
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
        base_cols = ["frame_idx", "iter", "atom_idx"] if add_index_cols else []
        wanted_cols = list(columns) if not isinstance(columns, str) else [columns]
        return pd.DataFrame(columns=base_cols + wanted_cols)
    return pd.concat(out_frames, ignore_index=True)


def extract_fort7_data_summaries(
    handler,
    columns: Union[str, Sequence[str]],
    frames: Indexish = None,
    iterations: Indexish = None,
    regex: bool = False,
    add_index_cols: bool = True,
) -> pd.DataFrame:
    """Extract per-iteration summary columns from fort.7."""
    sim_df = handler.dataframe()
    idx_list = resolve_indices(handler, frames, iterations)

    if len(sim_df) == 0:
        return pd.DataFrame()

    cols = _resolve_columns(sim_df, columns, regex=regex, must_exist=True)
    part = sim_df.iloc[idx_list][cols].reset_index(drop=True)

    if add_index_cols and "iter" not in part.columns:
        meta_df = sim_df.loc[idx_list, ["iter"]].reset_index(drop=True)
        meta_df.insert(0, "frame_idx", idx_list)
        part = pd.concat([meta_df, part], axis=1)
    elif add_index_cols and "iter" in part.columns:
        part.insert(0, "frame_idx", idx_list)

    return part


__all__ = [
    "extract_fort7_data_per_atom",
    "extract_fort7_data_summaries",
]
