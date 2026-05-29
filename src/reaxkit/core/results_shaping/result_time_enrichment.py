"""
Shared helpers to enrich analysis result tables with time from iteration.

**Usage context**

- Import these helpers from ReaxKit core modules when implementing CLI and workflow logic.
- Reuse the public APIs here to keep behavior consistent across commands and engines.
"""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from typing import Any

import numpy as np
import pandas as pd

from reaxkit.presentation.convert import convert_xaxis


def _iter_time_mapping_from_simulation(simulation: Any) -> dict[int, float] | None:
    """
    Iter time mapping from simulation.
    """
    if simulation is None:
        return None
    iterations = getattr(simulation, "iterations", None)
    time_values = getattr(simulation, "time", None)
    if iterations is None or time_values is None:
        return None
    try:
        it = np.asarray(iterations, dtype=int).reshape(-1)
        tv = np.asarray(time_values, dtype=float).reshape(-1)
    except Exception:
        return None
    if it.size == 0 or tv.size == 0 or it.shape[0] != tv.shape[0]:
        return None
    return {int(i): float(t) for i, t in zip(it, tv, strict=False)}


def _iter_time_mapping_from_data(data: Any) -> dict[int, float] | None:
    """
    Iter time mapping from data.
    """
    candidates: list[Any] = []

    sim = getattr(data, "simulation", None)
    if sim is not None:
        candidates.append(sim)

    trajectory = getattr(data, "trajectory", None)
    if trajectory is not None:
        candidates.append(getattr(trajectory, "simulation", None))

    charges = getattr(data, "charges", None)
    if charges is not None:
        candidates.append(getattr(charges, "simulation", None))

    connectivity = getattr(data, "connectivity", None)
    if connectivity is not None:
        candidates.append(getattr(connectivity, "simulation", None))

    for cand in candidates:
        mapping = _iter_time_mapping_from_simulation(cand)
        if mapping:
            return mapping
    return None


def _attach_time_to_frame(
    frame: pd.DataFrame,
    *,
    iter_to_time: dict[int, float] | None,
    control_file: str,
) -> pd.DataFrame:
    """
    Attach time to frame.
    """
    if "time" in frame.columns or "iter" not in frame.columns or frame.empty:
        return frame

    out = frame.copy()
    iter_values = pd.to_numeric(out["iter"], errors="coerce")
    valid_mask = iter_values.notna()
    if not valid_mask.any():
        return out

    if iter_to_time:
        vals = np.full((len(out),), np.nan, dtype=float)
        iter_int = iter_values[valid_mask].astype(int).to_numpy()
        vals[valid_mask.to_numpy()] = np.asarray([iter_to_time.get(int(v), np.nan) for v in iter_int], dtype=float)
        out["time"] = vals
        return out

    try:
        converted, _ = convert_xaxis(
            iter_values[valid_mask].astype(int).to_numpy(),
            "time",
            control_file=control_file,
        )
    except Exception:
        return out

    vals = np.full((len(out),), np.nan, dtype=float)
    vals[valid_mask.to_numpy()] = np.asarray(converted, dtype=float)
    out["time"] = vals
    return out


def enrich_result_with_time(result: Any, data: Any, *, control_file: str = "control") -> Any:
    """
    Attach a ``time`` column to tabular result frames when ``iter`` exists.
    
    This function is part of the ReaxKit core API and performs the operation
    described by its name and arguments.
    
    Parameters
    -----
    result : Any
        Input parameter used by this function.
    data : Any
        Input parameter used by this function.
    control_file : str, optional
        Input parameter used by this function.
    
    Returns
    -----
    Any
        Value produced by this function call.
    
    Examples
    -----
    ```python
    from reaxkit.core.results_shaping.result_time_enrichment import enrich_result_with_time
    # Configure required arguments for your case.
    result = enrich_result_with_time(...)
    print(type(result).__name__)
    ```
    Sample output:
    ```text
    str
    ```
    The output type reflects the return contract for this API call.
    """
    iter_to_time = _iter_time_mapping_from_data(data)

    def _enrich_attr(obj: Any, attr: str) -> None:
        value = getattr(obj, attr, None)
        if isinstance(value, pd.DataFrame):
            setattr(
                obj,
                attr,
                _attach_time_to_frame(value, iter_to_time=iter_to_time, control_file=control_file),
            )

    if is_dataclass(result):
        for f in fields(result):
            _enrich_attr(result, f.name)
    elif hasattr(result, "__dict__"):
        for key in vars(result).keys():
            _enrich_attr(result, str(key))
    return result

