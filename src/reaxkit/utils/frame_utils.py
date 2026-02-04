"""
Frame and atom selection utilities for ReaxKit analyses.

This module provides common helpers for parsing flexible user input
(e.g., CLI arguments or configuration strings) that specify frame and
atom selections, and for resolving those selections into concrete,
ordered indices usable by handlers and analyzers.

Typical use cases include:

- parsing frame ranges such as ``"0:100:5"`` or explicit lists like ``"10,20,30"``
- selecting subsets of rows from DataFrames by frame index
- resolving iteration numbers into frame indices via handler metadata
- parsing atom index lists for per-atom analyses
"""


from __future__ import annotations
from typing import Optional, Sequence, Union, Iterable, List
import pandas as pd

FramesT = Optional[Union[slice, Sequence[int]]]

def parse_frames(arg: Optional[str]) -> FramesT:
    """
    Parse a frame-selection string into a slice or index list.

    Supported formats are:
    - ``"start:stop[:step]"`` → ``slice``
    - ``"i,j,k"`` → list of integers
    - ``None`` or empty string → ``None`` (select all frames)

    Parameters
    ----------
    arg : str or None
        Frame selection string.

    Returns
    -------
    slice or list[int] or None
        Parsed frame selection.
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
    """
    Select rows from a DataFrame based on frame indices.

    Selection is performed using row-position indexing.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing per-frame data.
    frames : slice or list[int] or None
        Frame selection returned by ``parse_frames``.

    Returns
    -------
    pandas.DataFrame
        DataFrame restricted to the selected frames.
        """
    if frames is None:
        return df
    if isinstance(frames, slice):
        return df.iloc[frames]
    return df.iloc[list(frames)]

def _select_frames(xh, start: Optional[int], stop: Optional[int], every: int) -> range:
    """
    Construct a range of frame indices for a handler.

    Notes
    -----
    This is an internal helper used to support legacy workflows and
    handler-based frame iteration.
    """
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
    """
    Parse an atom-index selection string.

    Parameters
    ----------
    arg : str or None
        Comma- or space-separated atom indices.

    Returns
    -------
    list[int] or None
        Parsed atom indices, or ``None`` if no selection is provided.
        """
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
    Resolve user-specified frame or iteration selections into frame indices.

    Frame selection is resolved in the following order:
    1. Explicit frame indices or slices (if provided)
    2. Iteration numbers mapped to frame indices via ``handler.dataframe()['iter']``

    An optional stride may be applied to decimate the result.

    Parameters
    ----------
    handler
        Handler providing access to per-frame simulation data.
    frames : slice or list[int], optional
        Explicit frame selection.
    iterations : iterable of int, optional
        Iteration numbers to map to frame indices.
    step : int, optional
        Stride applied to the resolved frame indices.

    Returns
    -------
    list[int]
        Ordered list of resolved frame indices.
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
