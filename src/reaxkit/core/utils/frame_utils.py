"""
Frame and atom selection utilities for ReaxKit analyses.

This module provides common helpers for parsing flexible user input
(e.g., CLI arguments or configuration strings) that specify frame and
atom selections, and for resolving those selections into concrete,
ordered indices usable by handlers and analyzers.
"""

from __future__ import annotations

from typing import Optional, Sequence, Union, Iterable, List, Any
import re

import pandas as pd

FramesT = Optional[Union[slice, Sequence[int]]]
_INT_RANGE_RE = re.compile(r"^\s*(-?\d+)\s*-\s*(-?\d+)\s*$")


def _as_tokens(arg: Any) -> list[str]:
    """Normalize a frame selector input into non-empty text tokens."""
    if arg is None:
        return []
    if isinstance(arg, (list, tuple, set)):
        out: list[str] = []
        for item in arg:
            token = str(item).strip()
            if token:
                out.append(token)
        return out
    token = str(arg).strip()
    return [token] if token else []


def _expand_range_token(token: str) -> list[int] | None:
    """
    Expand one range token to explicit indices.

    Supported:
    - ``start-stop`` (inclusive)
    - ``start:stop[:step]`` (python-style stop-exclusive)
    """
    m = _INT_RANGE_RE.match(token)
    if m is not None:
        start = int(m.group(1))
        stop = int(m.group(2))
        step = 1 if stop >= start else -1
        return list(range(start, stop + step, step))

    if ":" in token:
        parts = [p.strip() for p in token.split(":")]
        if len(parts) > 3:
            raise ValueError(f"Invalid frame selector '{token}'. Use start:stop[:step].")
        start = int(parts[0]) if parts[0] else 0
        if len(parts) < 2 or not parts[1]:
            raise ValueError(f"Invalid frame selector '{token}'. Stop is required.")
        stop = int(parts[1])
        if len(parts) > 2 and parts[2]:
            step = int(parts[2])
        else:
            step = 1 if stop >= start else -1
        if step == 0:
            raise ValueError("Frame selector step cannot be zero.")
        return list(range(start, stop, step))
    return None


def parse_frames(arg: Any) -> FramesT:
    """
    Parse a frame selector into a ``slice`` or explicit index list.

    Supported formats include:
    - ``start:stop[:step]`` -> ``slice`` (single-token only)
    - ``start-stop`` -> inclusive integer list
    - ``i,j,k`` -> integer list
    - ``i j k`` -> integer list
    - ``None`` / empty -> ``None`` (select all)
    """
    tokens = _as_tokens(arg)
    if not tokens:
        return None

    # Keep legacy behavior: a lone colon-expression yields a slice.
    if len(tokens) == 1:
        s = tokens[0]
        if ":" in s:
            parts = [p.strip() for p in s.split(":")]
            if len(parts) > 3:
                raise ValueError(f"Invalid frame selector '{s}'. Use start:stop[:step].")
            start = int(parts[0]) if parts[0] else None
            stop = int(parts[1]) if len(parts) > 1 and parts[1] else None
            step = int(parts[2]) if len(parts) > 2 and parts[2] else None
            return slice(start, stop, step)

        expanded = _expand_range_token(s)
        if expanded is not None:
            return expanded

        parts = [p.strip() for p in re.split(r"[,\s]+", s) if p.strip()]
        return [int(p) for p in parts]

    # Multi-token selector: expand each token to explicit indices.
    out: list[int] = []
    for tok in tokens:
        expanded = _expand_range_token(tok)
        if expanded is not None:
            out.extend(expanded)
            continue
        parts = [p.strip() for p in tok.split(",") if p.strip()]
        out.extend(int(p) for p in parts)
    return out


def parse_frame_indices(arg: Any) -> list[int] | None:
    """
    Parse any supported frame selector into explicit frame indices.

    This is intended for CLI/UI request normalization where request objects
    should store ``list[int] | None``.
    """
    sel = parse_frames(arg)
    if sel is None:
        return None
    if isinstance(sel, slice):
        start = 0 if sel.start is None else int(sel.start)
        if sel.stop is None:
            raise ValueError("Open-ended frame range requires an explicit stop (for example 0:20).")
        stop = int(sel.stop)
        step = 1 if sel.step is None else int(sel.step)
        if step == 0:
            raise ValueError("Frame selector step cannot be zero.")
        return list(range(start, stop, step))
    return [int(v) for v in sel]


# Back-compat alias requested in project notes
_parse_frames = parse_frames


def select_frames(df: pd.DataFrame, frames: FramesT) -> pd.DataFrame:
    """Select rows from a DataFrame based on frame indices."""
    if frames is None:
        return df
    if isinstance(frames, slice):
        return df.iloc[frames]
    return df.iloc[list(frames)]


def _select_frames(xh, start: Optional[int], stop: Optional[int], every: int) -> range:
    """
    Construct a range of frame indices for a handler.

    This is an internal helper used to support legacy workflows and
    handler-based frame iteration.
    """
    try:
        n_frames = xh.n_frames()
    except Exception:
        df = xh.dataframe()
        n_frames = len(df)
    s = 0 if start is None else max(0, int(start))
    e = (n_frames - 1) if stop is None else min(n_frames - 1, int(stop))
    ev = max(1, int(every))
    if e < s:
        return range(0, 0)
    return range(s, e + 1, ev)


def parse_atoms(arg: Optional[str]) -> Optional[List[int]]:
    """Parse a comma/space-separated atom-index selector."""
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


def resolve_indices(
    handler,
    frames: FramesT = None,
    iterations: Optional[Iterable[int]] = None,
    step: Optional[int] = None,
) -> list[int]:
    """Resolve user-specified frame or iteration selections into frame indices."""
    sim_df = handler.dataframe()
    n = len(sim_df)
    all_idx = list(range(n))

    if frames is None:
        chosen = set(all_idx)
    elif isinstance(frames, slice):
        chosen = set(range(n)[frames])
    else:
        chosen = set(int(i) for i in frames)

    if iterations is not None:
        iters = set(int(i) for i in (iterations if not isinstance(iterations, int) else [iterations]))
        iter_to_idx = {int(sim_df.iloc[i]["iter"]): i for i in all_idx}
        chosen &= {iter_to_idx[it] for it in iters if it in iter_to_idx}

    idx = sorted(i for i in chosen if 0 <= i < n)
    if step and int(step) > 1:
        idx = idx[::int(step)]
    return idx
