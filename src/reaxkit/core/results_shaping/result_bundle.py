"""
Helpers for packaging multi-table analysis outputs for presentation/persistence.

**Usage context**

- Import these helpers from ReaxKit core modules when implementing CLI and workflow logic.
- Reuse the public APIs here to keep behavior consistent across commands and engines.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class DualTableResultBundle:
    """
    Result container with stable field names used by persistence.
    
    
    Fields
    -----
    table : pd.DataFrame
        Field value used by this structured record.
    tract_table : pd.DataFrame
        Field value used by this structured record.
    summary : dict[str, Any] | None, optional
        Field value used by this structured record.
    request : Any, optional
        Field value used by this structured record.
    soap_descriptors : np.ndarray | None, optional
        Field value used by this structured record.
    """

    table: pd.DataFrame
    tract_table: pd.DataFrame
    summary: dict[str, Any] | None = None
    request: Any = None
    soap_descriptors: np.ndarray | None = None


def bundle_canonical_and_tract_tables(
    result: Any,
    *,
    canonical_attr: str = "table",
    tract_attr: str = "tract_table",
) -> DualTableResultBundle:
    """
    Attach canonical + TRACT tables under stable names for one output bundle.
    
    This function is part of the ReaxKit core API and performs the operation
    described by its name and arguments.
    
    Parameters
    -----
    result : Any
        Input parameter used by this function.
    canonical_attr : str, optional
        Input parameter used by this function.
    tract_attr : str, optional
        Input parameter used by this function.
    
    Returns
    -----
    DualTableResultBundle
        Value produced by this function call.
    
    Examples
    -----
    ```python
    from reaxkit.core.results_shaping.result_bundle import bundle_canonical_and_tract_tables
    # Configure required arguments for your case.
    result = bundle_canonical_and_tract_tables(...)
    print(type(result).__name__)
    ```
    Sample output:
    ```text
    str
    ```
    The output type reflects the return contract for this API call.
    """
    canonical = getattr(result, canonical_attr, None)
    tract = getattr(result, tract_attr, None)
    if not isinstance(canonical, pd.DataFrame):
        raise TypeError(
            f"Expected `{canonical_attr}` to be a pandas DataFrame; got {type(canonical).__name__}."
        )
    if not isinstance(tract, pd.DataFrame):
        raise TypeError(
            f"Expected `{tract_attr}` to be a pandas DataFrame; got {type(tract).__name__}."
        )
    return DualTableResultBundle(
        table=canonical,
        tract_table=tract,
        summary=getattr(result, "summary", None),
        request=getattr(result, "request", None),
        soap_descriptors=getattr(result, "soap_descriptors", None),
    )


__all__ = ["DualTableResultBundle", "bundle_canonical_and_tract_tables"]
