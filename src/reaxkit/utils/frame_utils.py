"""Utilities to parse and resolve frame/atom selections into concrete DataFrame/handler indices.

ReaxFF analyses frequently require flexible user input for selecting frames
(e.g., "0:100:5", "10,20,30") and atom indices. This module provides the common
parsing and resolution logic used across handlers and analyzers:

  • parse_frames(): interprets slice-like or comma-separated frame strings,
  • select_frames(): applies frame selections to DataFrames,
  • resolve_indices(): converts user-specified frames or iteration numbers into
    concrete frame indices using handler metadata,
  • parse_atoms(): parses atom-index lists,
  • helper utilities for range construction and backwards compatibility.

These functions ensure that all modules can accept flexible frame/atom notation
(including CLI inputs) and resolve them into consistent, ordered Python
indices before analysis.

"""

from __future__ import annotations
from typing import Optional, Sequence, Union, Iterable, List
import pandas as pd

FramesT = Optional[Union[slice, Sequence[int]]]

def parse_frames(arg: Optional[str]) -> FramesT:
    """
    Parse frame selection string:
      - 'start:stop[:step]' -> slice
      - 'i,j,k'             -> list of ints
      - None or ''          -> None (keep all)
    """
    if arg is None or str(arg).strip() == "":
        return None
    s = str(arg).strip()
    if ":" in s:
        parts = [p.strip() for p in s.split(":")]
        start = int(parts[0]) if parts[0] else None
        stop  = int(parts[1]) if len(parts) > 1 and parts[1] else None
        step  = int(parts[2]) if len(parts) > 2 and parts[2] else None
        return slice(start, stop, step)
    return [int(p.strip()) for p in s.split(",") if p.strip()]

# Back-compat aliases requested in project notes
_parse_frames = parse_frames

def select_frames(df: pd.DataFrame, frames: FramesT) -> pd.DataFrame:
    """Apply a slice or explicit index list to a DataFrame (row-position based)."""
    if frames is None:
        return df
    if isinstance(frames, slice):
        return df.iloc[frames]
    return df.iloc[list(frames)]

def _select_frames(xh, start: Optional[int], stop: Optional[int], every: int) -> range:
    """Return a range of frame indices [start, stop] with stride 'every' for a handler."""
    try:
        n_frames = xh.n_frames()
    except Exception:
        # fallback from dataframe length
        df = xh.dataframe()
        n_frames = len(df)
    s = 0 if start is None else max(0, int(start))
    e = (n_frames - 1) if stop is None else min(n_frames - 1, int(stop))
    ev = max(1, int(every))
    if e < s:
        return range(0, 0)  # empty
    return range(s, e + 1, ev)

def parse_atoms(arg: Optional[str]) -> Optional[List[int]]:
    """Parse comma/space separated atom indices into list[int], or None."""
    if arg is None or str(arg).strip() == "":
        return None
    parts = [p for chunk in str(arg).split(",") for p in chunk.split()]
    out: List[int] = []
    for p in parts:
        try:
            out.append(int(p))
        except ValueError:
            continue
    return out or None

def resolve_indices(handler, frames: FramesT = None, iterations: Optional[Iterable[int]] = None, step: Optional[int] = None) -> list[int]:
    """
    Return an ordered list of frame indices selected by either:
      - frames: slice or explicit indices (preferred if provided)
      - iterations: iter numbers mapped via handler.dataframe()['iter']
    Optionally decimate with 'step'.
    """
    sim_df = handler.dataframe()
    n = len(sim_df)
    all_idx = list(range(n))

    # Start from frames
    if frames is None:
        chosen = set(all_idx)
    elif isinstance(frames, slice):
        chosen = set(range(n)[frames])
    else:
        chosen = set(int(i) for i in frames)

    # Filter by iteration numbers if provided
    if iterations is not None:
        iters = set(int(i) for i in (iterations if not isinstance(iterations, int) else [iterations]))
        iter_to_idx = {int(sim_df.iloc[i]["iter"]): i for i in all_idx}
        chosen &= {iter_to_idx[it] for it in iters if it in iter_to_idx}

    idx = sorted(i for i in chosen if 0 <= i < n)
    if step and int(step) > 1:
        idx = idx[::int(step)]
    return idx
