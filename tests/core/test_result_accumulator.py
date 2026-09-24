from dataclasses import dataclass
import weakref

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from reaxkit.domain.base_result import BaseResult
from reaxkit.core.runtime.result_stream import ResultAccumulator, combine_results


@dataclass
class ExampleResult(BaseResult):
    request: object
    table: pd.DataFrame
    mapping: pd.DataFrame
    frame_indices: np.ndarray
    iterations: np.ndarray


def _result(index):
    return ExampleResult(None, pd.DataFrame({"frame": [index], "value": [index / 3]}),
                         pd.DataFrame({"id": np.arange(200)}), np.array([index]), np.array([10 * index]))


def test_accumulator_matches_original_collector_and_releases_duplicate_mapping():
    results = [_result(index) for index in range(10)]
    expected = combine_results(results, "request")
    accumulator = ResultAccumulator(chunk_frames=3)
    for result in results:
        accumulator.add(result)
    mapping_ref = weakref.ref(result.mapping)
    del result, results
    assert mapping_ref() is None
    actual = accumulator.finish("request")
    assert actual.request == "request"
    assert_frame_equal(actual.table, expected.table)
    assert_frame_equal(actual.mapping, expected.mapping)
    np.testing.assert_array_equal(actual.frame_indices, expected.frame_indices)
    np.testing.assert_array_equal(actual.iterations, expected.iterations)


def test_empty_public_table_preserves_columns_and_dtypes():
    result = _result(0)
    result.table = result.table.iloc[:0]
    accumulator = ResultAccumulator()
    accumulator.add(result)
    assert_frame_equal(accumulator.finish(None).table, result.table)
