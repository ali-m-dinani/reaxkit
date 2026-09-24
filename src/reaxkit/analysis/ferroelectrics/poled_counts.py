"""Compact frame-resolved counts of positive and negative vector components."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from reaxkit.domain.data_models import TrajectoryData


POLED_COUNT_COLUMNS = [
    "frame",
    "iter",
    "time",
    "direction",
    "count_poled_up",
    "count_poled_down",
    "count_all",
    "percentage_poled_up",
    "percentage_poled_down",
]


def _iteration_times(trajectory: TrajectoryData) -> Mapping[int, float]:
    simulation = trajectory.simulation
    if simulation is None or simulation.iterations is None or simulation.time is None:
        return {}
    iterations = np.asarray(simulation.iterations).reshape(-1)
    times = np.asarray(simulation.time, dtype=float).reshape(-1)
    if len(iterations) != len(times):
        return {}
    return {
        int(iteration): float(time)
        for iteration, time in zip(iterations, times, strict=True)
    }


def directional_poled_counts(
    table: pd.DataFrame,
    trajectory: TrajectoryData,
    value_columns: Mapping[str, str],
) -> pd.DataFrame:
    """Count positive and negative x/y/z values for every selected frame."""

    required = {"frame_index", "iter", *value_columns.values()}
    missing = sorted(required.difference(table.columns))
    if missing:
        raise ValueError(f"Cannot calculate poled counts; missing columns: {missing}.")

    iteration_times = _iteration_times(trajectory)
    rows: list[dict[str, object]] = []
    for (frame, iteration), group in table.groupby(
        ["frame_index", "iter"], sort=True
    ):
        count_all = len(group)
        time = iteration_times.get(int(iteration), np.nan)
        for direction in "xyz":
            values = pd.to_numeric(
                group[value_columns[direction]], errors="coerce"
            ).to_numpy(float)
            count_up = int(np.count_nonzero(values > 0.0))
            count_down = int(np.count_nonzero(values < 0.0))
            rows.append(
                {
                    "frame": int(frame),
                    "iter": int(iteration),
                    "time": time,
                    "direction": direction,
                    "count_poled_up": count_up,
                    "count_poled_down": count_down,
                    "count_all": count_all,
                    "percentage_poled_up": (
                        100.0 * count_up / count_all if count_all else np.nan
                    ),
                    "percentage_poled_down": (
                        100.0 * count_down / count_all if count_all else np.nan
                    ),
                }
            )
    return pd.DataFrame(rows, columns=POLED_COUNT_COLUMNS)


__all__ = ["POLED_COUNT_COLUMNS", "directional_poled_counts"]
