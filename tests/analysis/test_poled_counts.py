from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from reaxkit.analysis.ferroelectrics.poled_counts import (
    POLED_COUNT_COLUMNS,
    directional_poled_counts,
)
from reaxkit.core.results_shaping.result_time_enrichment import _attach_time_to_frame
from reaxkit.domain.data_models import SimulationData, TrajectoryData


def test_directional_poled_counts_reports_time_counts_and_percentages() -> None:
    simulation = SimulationData(
        atom_ids=[1],
        iterations=np.asarray([10]),
        time=np.asarray([0.25]),
    )
    trajectory = TrajectoryData(
        positions=np.zeros((1, 1, 3)),
        elements=["Al"],
        atom_ids=[1],
        iterations=np.asarray([10]),
        simulation=simulation,
    )
    table = pd.DataFrame(
        {
            "frame_index": [4, 4, 4, 4],
            "iter": [10, 10, 10, 10],
            "vx": [2.0, -1.0, 0.0, np.nan],
            "vy": [1.0, 3.0, -2.0, -4.0],
            "vz": [0.0, 0.0, 0.0, 0.0],
        }
    )

    result = directional_poled_counts(
        table, trajectory, {"x": "vx", "y": "vy", "z": "vz"}
    )

    assert list(result.columns) == POLED_COUNT_COLUMNS
    result = result.set_index("direction")
    assert result.loc["x", "frame"] == 4
    assert result.loc["x", "iter"] == 10
    assert result.loc["x", "time"] == pytest.approx(0.25)
    assert result.loc["x", "count_poled_up"] == 1
    assert result.loc["x", "count_poled_down"] == 1
    assert result.loc["x", "count_all"] == 4
    assert result.loc["x", "percentage_poled_up"] == pytest.approx(25.0)
    assert result.loc["x", "percentage_poled_down"] == pytest.approx(25.0)
    assert result.loc["y", "percentage_poled_up"] == pytest.approx(50.0)
    assert result.loc["y", "percentage_poled_down"] == pytest.approx(50.0)
    assert result.loc["z", "count_poled_up"] == 0
    assert result.loc["z", "count_poled_down"] == 0


def test_time_enrichment_fills_reserved_missing_time_column() -> None:
    table = pd.DataFrame({"frame": [0], "iter": [10], "time": [np.nan]})

    result = _attach_time_to_frame(
        table, iter_to_time={10: 0.25}, control_file="unused"
    )

    assert list(result.columns) == ["frame", "iter", "time"]
    assert result.loc[0, "time"] == pytest.approx(0.25)
