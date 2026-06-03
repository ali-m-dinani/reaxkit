"""Template numerical utility functions for ReaxKit.

This module demonstrates the general structure used by reusable utility files:
small pure functions, clear typing, deterministic numeric behavior, and no
workflow/task coupling.

**Usage context**

- Shared math: Centralize reusable numerical formulas and transforms.
- Cross-module reuse: Support analyzers, engines, and workflows via pure APIs.
- Testing-friendly design: Keep utilities stateless and side-effect free.

Notes
-----
- Prefer explicit units in argument names when a quantity is unit-sensitive.
- Keep conversion constants local or imported from canonical constants modules.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np

__all__ = [
    "template_series_transform",
    "template_scalar_metric",
]


def _as_float_array(values: Iterable[float]) -> np.ndarray:
    """Convert an iterable of numeric values to a 1D float array."""
    arr = np.asarray(list(values), dtype=float).reshape(-1)
    return arr


def template_series_transform(
    values: Iterable[float],
    *,
    scale: float = 1.0,
    offset: float = 0.0,
) -> np.ndarray:
    """Apply a linear transform to a numeric series.

    This helper is a template for utilities that transform vector-like input
    and return NumPy arrays suitable for downstream plotting/fitting logic.

    Parameters
    -----
    values : Iterable[float]
        Input numeric sequence to transform.
    scale : float
        Multiplicative factor applied to each input value.
    offset : float
        Additive term applied after scaling.

    Returns
    -----
    numpy.ndarray
        Transformed 1D array where `out = values * scale + offset`.

    Examples
    -----
    ```python
    out = template_series_transform([1.0, 2.0, 3.0], scale=2.0, offset=-1.0)
    print(out.tolist())
    ```
    Sample output:
    `[1.0, 3.0, 5.0]`
    Meaning:
    Each element was multiplied by 2 and then shifted by -1.
    """
    arr = _as_float_array(values)
    return arr * float(scale) + float(offset)


def template_scalar_metric(
    values: Iterable[float],
    *,
    reference: float = 0.0,
) -> float:
    """Compute a simple scalar metric from a numeric series.

    This function illustrates scalar-return utility patterns commonly used for
    summary statistics or objective values in higher-level modules.

    Notes
    -----
    Returns `0.0` for empty inputs to keep downstream call sites robust.

    Parameters
    -----
    values : Iterable[float]
        Input numeric sequence used to compute the metric.
    reference : float
        Reference value used to shift the series before reduction.

    Returns
    -----
    float
        Mean squared deviation from `reference`.

    Examples
    -----
    ```python
    m = template_scalar_metric([1.0, 2.0, 3.0], reference=2.0)
    print(round(m, 6))
    ```
    Sample output:
    `0.666667`
    Meaning:
    The metric is the average squared distance from the reference value 2.0.
    """
    arr = _as_float_array(values)
    if arr.size == 0:
        return 0.0
    delta = arr - float(reference)
    return float(np.mean(delta * delta))
