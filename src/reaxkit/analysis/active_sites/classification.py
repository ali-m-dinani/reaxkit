"""Provide site-label classification helpers for active-site analysis outputs.

This module maps per-atom structural signals to TRACT-parity site labels used
by active-site workflows. It is scoped to label assignment and does not compute
geometric descriptors directly.

**Usage context**

- Label assignment: Convert structural flags into stable site categories.
- TRACT parity: Keep label priority rules consistent across analyzers.
- Reporting inputs: Supply compact site labels for figures and summaries.
"""

from __future__ import annotations

import numpy as np


def site_labels_tract_parity(
    elements: list[str],
    carbon_element: str,
    is_undercoord: np.ndarray,
    in_non6_ring: np.ndarray,
    defect_type: np.ndarray,
    edge_label_by_global: dict[int, str],
) -> np.ndarray:
    """Assign TRACT-style site labels using ordered priority rules.

    Parameters
    -----
    elements : list[str]
        Element labels for each atom index.
    carbon_element : str
        Carbon element symbol used by upstream workflows.
    is_undercoord : np.ndarray
        Boolean mask indicating under-coordinated atoms.
    in_non6_ring : np.ndarray
        Boolean mask indicating membership in non-hexagonal rings.
    defect_type : np.ndarray
        Per-atom defect-type labels.
    edge_label_by_global : dict[int, str]
        Optional edge labels keyed by global atom index.

    Returns
    -----
    np.ndarray
        Per-atom label array with values such as `basal`, `defect`,
        `edge_zigzag`, `edge_armchair`, or `under_coordinated`.

    Examples
    -----
    ```python
    labels = site_labels_tract_parity(
        elements=["C", "C"],
        carbon_element="C",
        is_undercoord=np.array([False, True]),
        in_non6_ring=np.array([False, False]),
        defect_type=np.array(["none", "none"], dtype=object),
        edge_label_by_global={1: "edge_armchair"},
    )
    ```
    Sample output:
    `array(["basal", "edge_armchair"], dtype=object)`
    Meaning:
    Labels follow undercoord/defect/edge priority ordering.
    """
    n = len(elements)
    _ = carbon_element
    out = np.full(n, "basal", dtype=object)
    for i in range(n):
        if bool(is_undercoord[i]):
            out[i] = str(edge_label_by_global.get(i, "under_coordinated"))
        elif str(defect_type[i]) != "none":
            out[i] = "defect"
        elif i in edge_label_by_global:
            out[i] = str(edge_label_by_global[i])
        elif bool(in_non6_ring[i]):
            out[i] = "defect"
        else:
            out[i] = "basal"
    return out
