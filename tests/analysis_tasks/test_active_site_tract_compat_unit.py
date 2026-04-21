"""Unit checks for TRACT-compatible active-site table formatters."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from reaxkit.analysis.active_sites import (
    TRACT_EVENTS_COLUMNS,
    TRACT_STRUCTURAL_COLUMNS,
    to_tract_events_table,
    to_tract_structural_table,
)


def test_structural_formatter_enforces_exact_column_order():
    src = pd.DataFrame(
        {
            "atom_id": [1, 2],
            "element": ["C", "C"],
            "x": [0.0, 1.0],
            "y": [0.0, 0.0],
            "z": [0.0, 0.0],
            "n_bonds": [3, 2],
            "n_bonds_c": [3, 1],
            "is_undercoord": [False, True],
            "has_hetero_bond": [False, True],
            "d_pyr": [0.1, 0.0],
            "d_ang_deg": [np.nan, 12.0],
            "local_roughness": [0.01, 0.02],
            "ring_size_min": [6, 5],
            "ring_size_max": [6, 7],
            "in_non6_ring": [False, True],
            "defect_type": ["none", "SW_5775"],
            "label": ["basal", "edge_armchair"],
            "seg_id": [-1, 0],
        }
    )

    out = to_tract_structural_table(src)
    assert tuple(out.columns) == TRACT_STRUCTURAL_COLUMNS
    # has_hetero_bond is mapped directly from canonical table.
    assert bool(out.loc[0, "has_hetero_bond"]) is False
    assert bool(out.loc[1, "has_hetero_bond"]) is True
    # Missing advanced columns should exist and be NaN/default.
    assert np.isnan(out.loc[0, "bond_strain"])
    assert int(out.loc[0, "grain_id"]) == -1


def test_events_formatter_maps_contact_columns_to_tract_names():
    src = pd.DataFrame(
        {
            "atom_id": [1, 2],
            "n_events_O": [1, 0],
            "n_events_Si": [0, 2],
            "first_event_frame_O": [100, -1],
            "first_event_frame_Si": [-1, 140],
            "is_reactive_O": [True, False],
            "is_reactive_Si": [False, True],
            "total_bound_frames_O": [30, 0],
            "total_bound_frames_Si": [0, 55],
            "mean_contact_O_when_bound": [0.95, np.nan],
            "mean_contact_Si_when_bound": [np.nan, 1.82],
        }
    )

    out = to_tract_events_table(src)
    assert tuple(out.columns) == TRACT_EVENTS_COLUMNS
    assert float(out.loc[0, "mean_r_O_when_bound"]) == 0.95
    assert float(out.loc[1, "mean_r_Si_when_bound"]) == 1.82


def test_structural_formatter_strict_mode_raises_on_missing_critical_columns():
    src = pd.DataFrame(
        {
            "atom_id": [1],
            "element": ["C"],
            "x": [0.0],
            "y": [0.0],
            "z": [0.0],
            "n_bonds": [3],
            "is_undercoord": [False],
            "d_pyr": [0.0],
            "label": ["basal"],
        }
    )

    with pytest.raises(ValueError, match="STRICT TRACT compatibility failed"):
        to_tract_structural_table(src, strict=True)


def test_events_formatter_strict_mode_raises_on_missing_critical_columns():
    src = pd.DataFrame(
        {
            "atom_id": [1],
            "n_events_O": [0],
            "n_events_Si": [0],
        }
    )

    with pytest.raises(ValueError, match="STRICT TRACT compatibility failed"):
        to_tract_events_table(src, strict=True)
