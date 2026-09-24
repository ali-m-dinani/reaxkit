from __future__ import annotations

import numpy as np
import pandas as pd

from reaxkit.core.runtime.artifacts import BufferedTableSink
from reaxkit.core.runtime.reducers import (
    CountSumReducer,
    HistogramReducer,
    PlotMatrixReducer,
    TableAccumulator,
)


def test_table_accumulator_flushes_details_without_retaining_them(tmp_path):
    sink = BufferedTableSink(tmp_path / "details.csv", "csv", overwrite=True)
    reducer = TableAccumulator(retain=False, sink=sink, flush_rows=2)

    reducer.add([{"frame_index": 0, "value": 1.0}])
    reducer.add([{"frame_index": 1, "value": 2.0}])
    result = reducer.finalize()
    sink.finalize()

    assert result.empty
    pd.testing.assert_frame_equal(
        pd.read_csv(tmp_path / "details.csv"),
        pd.DataFrame({"frame_index": [0, 1], "value": [1.0, 2.0]}),
    )


def test_count_sum_reducer_ignores_invalid_bins_and_values():
    reducer = CountSumReducer(3)
    reducer.add(np.asarray([0, 0, 2, 4]), np.asarray([1.0, 3.0, 5.0, 9.0]))
    reducer.add(np.asarray([1, 2]), np.asarray([np.nan, 1.0]))

    counts, sums, means = reducer.finalize()
    np.testing.assert_array_equal(counts, [2, 0, 2])
    np.testing.assert_allclose(sums, [4.0, 0.0, 6.0])
    np.testing.assert_allclose(means[[0, 2]], [2.0, 3.0])
    assert np.isnan(means[1])


def test_histogram_and_plot_matrix_reducers_are_bounded_by_output_shape():
    histogram = HistogramReducer(np.asarray([0.0, 1.0, 2.0]))
    histogram.add(np.asarray([0.25, 0.75, 1.5]))
    counts, edges = histogram.finalize()
    np.testing.assert_array_equal(counts, [2, 1])
    np.testing.assert_allclose(edges, [0.0, 1.0, 2.0])

    matrix = PlotMatrixReducer(2)
    matrix.add(10, np.asarray([1.0, 2.0]))
    matrix.add(20, np.asarray([3.0, 4.0]))
    frames, values = matrix.finalize()
    np.testing.assert_array_equal(frames, [10, 20])
    np.testing.assert_allclose(values, [[1.0, 2.0], [3.0, 4.0]])
