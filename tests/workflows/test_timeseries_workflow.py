from __future__ import annotations

import argparse
from types import SimpleNamespace

import pandas as pd

from reaxkit.workflows import timeseries_workflow


def _base_args(**overrides):
    values = {
        "plot": "single",
        "grid": None,
        "xaxis": "iter",
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_plot_payload_for_trajectory_coordinate_series_single_plot():
    result = SimpleNamespace(
        table=pd.DataFrame(
            {
                "frame_index": [0, 1, 0, 1],
                "iter": [100, 200, 100, 200],
                "atom_id": [1, 1, 2, 2],
                "atom_type": ["H", "H", "O", "O"],
                "dim": ["z", "z", "z", "z"],
                "coord": [-1.0, -0.5, 2.0, 2.5],
            }
        )
    )

    payload = timeseries_workflow._plot_payload(
        "trajectory_coordinate_series",
        result,
        _base_args(plot="single"),
    )

    assert payload is not None
    assert payload["plot_type"] == "single_plot"
    assert len(payload["series"]) == 2
    labels = {series["label"] for series in payload["series"]}
    assert "atom_id=1, atom_type=H, dim=z" in labels
    assert "atom_id=2, atom_type=O, dim=z" in labels
