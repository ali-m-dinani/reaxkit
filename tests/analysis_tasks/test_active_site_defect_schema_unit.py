"""Unit checks for active-site defect schema naming and priority."""

from __future__ import annotations

import networkx as nx
import pytest

from reaxkit.analysis.active_sites.defects import (
    DEFECT_TYPE_SCHEMA,
    classify_defect_clusters,
    normalize_defect_type,
    per_atom_defect_types,
    prefer_defect_type,
)
from reaxkit.analysis.active_sites import defects as defects_mod


def test_normalize_defect_type_enforces_schema():
    assert normalize_defect_type("SW_5775") == "SW_5775"
    assert normalize_defect_type("none") == "none"
    assert normalize_defect_type("unknown_label") == "non6_cluster"


def test_prefer_defect_type_follows_tract_priority():
    # Higher-priority incoming should replace current.
    assert prefer_defect_type("non6_cluster", "SV_5_9") == "SV_5_9"
    # Lower-priority incoming should not replace current.
    assert prefer_defect_type("DV_5_8_5", "haeckelite_like") == "DV_5_8_5"
    # Schema normalization should happen before resolution.
    assert prefer_defect_type("none", "made_up_type") == "non6_cluster"


def test_schema_contains_expected_values():
    expected = {
        "SW_5775",
        "SV_5_9",
        "DV_5_8_5",
        "DV_555_777",
        "GB_chain_5_7",
        "EDGE_reczag_57",
        "haeckelite_like",
        "non6_cluster",
        "none",
    }
    assert set(DEFECT_TYPE_SCHEMA) == expected


def test_sw_like_component_without_5775_cycle_falls_back_to_non6_cluster():
    ring_sizes = [5, 5, 7, 7]
    ring_adj = nx.Graph()
    ring_adj.add_edges_from([(0, 1), (1, 2), (2, 3)])  # connected path, no 4-cycle
    boundary_frac = [0.0, 0.0, 0.0, 0.0]

    labels = classify_defect_clusters(ring_sizes, ring_adj, boundary_frac)
    assert labels == {0: "non6_cluster", 1: "non6_cluster", 2: "non6_cluster", 3: "non6_cluster"}


def test_dv_555777_like_component_without_alternating_hexad_falls_back_to_non6_cluster():
    ring_sizes = [5, 5, 5, 7, 7, 7]
    ring_adj = nx.Graph()
    ring_adj.add_edges_from([(0, 3), (0, 4), (0, 5), (1, 3), (2, 4)])  # connected but no 6-cycle
    boundary_frac = [0.0] * 6

    labels = classify_defect_clusters(ring_sizes, ring_adj, boundary_frac)
    assert labels == {0: "non6_cluster", 1: "non6_cluster", 2: "non6_cluster", 3: "non6_cluster", 4: "non6_cluster", 5: "non6_cluster"}


def test_per_atom_defect_type_uses_first_labeled_ring_tract_semantics(monkeypatch: pytest.MonkeyPatch):
    rings = [(0, 1, 2), (0, 3, 4)]

    def _fake_classifier(*_args, **_kwargs):
        # Atom 0 belongs to ring 0 and ring 1. TRACT semantics picks ring 0 first.
        return {0: "non6_cluster", 1: "haeckelite_like"}

    monkeypatch.setattr(defects_mod, "classify_defect_clusters", _fake_classifier)
    out = per_atom_defect_types(n_atoms=5, rings=rings, boundary_nodes=None)
    assert str(out[0]) == "non6_cluster"
