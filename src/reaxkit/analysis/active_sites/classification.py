"""Site-label helpers for active-site analysis."""

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
    """Assign TRACT-like labels with edge typing priority."""
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
