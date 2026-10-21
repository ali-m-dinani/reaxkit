from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from ase.io import read

from tests.analysis.test_hbn_al_neighborhoods import trajectory
from reaxkit.analysis.ferroelectrics.hbn_reference.polarization import REFERENCE_STRUCTURE_PATH
from reaxkit.analysis.ferroelectrics.hbn_reference.local_polarization import (
    HBNReferenceLocalPolarizationRequest, HBNReferenceLocalPolarizationTask,
    write_local_polarization_extxyz,
)
from reaxkit.analysis.ferroelectrics.hbn_reference.projected_polarity import (
    HBNReferenceProjectedPolarityRequest, HBNReferenceProjectedPolarityTask,
)
from reaxkit.core.results_shaping.result_time_enrichment import enrich_result_with_time, _attach_time_to_frame
from reaxkit.core.runtime.result_store import ResultStore, ResultTable


@pytest.mark.parametrize("kind", ["local", "projected"])
def test_native_times_are_ps_in_tables_extxyz_and_checkpoint(tmp_path, kind):
    data = trajectory(shifts=(0.0, 0.1, -0.1))
    data.simulation.time = np.array([0.0, 500.0, 2500.0])
    settings = dict(reference_path=REFERENCE_STRUCTURE_PATH, replication=(2, 1, 1))
    if kind == "local":
        task = HBNReferenceLocalPolarizationTask()
        request = HBNReferenceLocalPolarizationRequest(**settings)
    else:
        task = HBNReferenceProjectedPolarityTask()
        request = HBNReferenceProjectedPolarityRequest(
            **settings, write_extxyz=True, _output_path=str(tmp_path / "projected.extxyz")
        )
    result = task.run(data, request)
    enrich_result_with_time(result, data)
    enrich_result_with_time(result, data)
    for table in result.csv_tables.values():
        if "iter" in table:
            assert table.time_unit.eq("ps").all()
            np.testing.assert_allclose(table.time, table["iter"].map({0: 0.0, 10: 0.5, 20: 2.5}))
    if kind == "local":
        path = write_local_polarization_extxyz(result, tmp_path / "local.extxyz")
    else:
        path = tmp_path / "projected.extxyz"
    exported = read(path, index=":")
    np.testing.assert_allclose([atoms.info["time"] for atoms in exported], [0, 0.5, 2.5])
    assert all(atoms.info["time_unit"] == "ps" for atoms in exported)

    frames = []
    for index in range(3):
        simulation = replace(data.simulation, iterations=data.iterations[index:index + 1],
            time=data.simulation.time[index:index + 1], cell_lengths=data.simulation.cell_lengths[index:index + 1],
            cell_angles=data.simulation.cell_angles[index:index + 1])
        frames.append(replace(data, positions=data.positions[index:index + 1],
            iterations=data.iterations[index:index + 1], simulation=simulation, source_frame_indices=[index]))
    with ResultStore(tmp_path / "checkpoint", {}, batch_frames=1) as store:
        request._result_store = store
        streamed = task.run_stream(iter(frames), request)
        for name, expected in result.csv_tables.items():
            actual = streamed.csv_tables[name]
            if isinstance(actual, ResultTable):
                actual = actual.materialize()
            if "time" in expected:
                np.testing.assert_allclose(actual.time, expected.time)
                assert actual.time_unit.eq("ps").all()
    with ResultStore(tmp_path / "checkpoint", {}, resume=True) as store:
        request._result_store = store
        resumed = task.run_stream(iter(()), request)
        enrich_result_with_time(resumed, data)
        for table in resumed.csv_tables.values():
            if isinstance(table, ResultTable):
                table = table.materialize()
            if "iter" in table:
                np.testing.assert_allclose(table.time, table["iter"].map({0: 0.0, 10: 0.5, 20: 2.5}))


@pytest.mark.parametrize("label, value", [("Time (fs)", 500), ("Time (ps)", 0.5), ("Time (ns)", 0.0005)])
def test_control_fallback_is_always_ps_and_idempotent(monkeypatch, label, value):
    monkeypatch.setattr("reaxkit.core.results_shaping.result_time_enrichment.convert_xaxis",
                        lambda *args, **kwargs: (np.array([value]), label))
    original = pd.DataFrame({"iter": [100]})
    converted = _attach_time_to_frame(original, iter_to_time=None, control_file="control", time_unit="ps")
    assert converted.time.iloc[0] == pytest.approx(0.5)
    repeated = _attach_time_to_frame(converted, iter_to_time=None, control_file="control", time_unit="ps")
    pd.testing.assert_frame_equal(converted, repeated)
    legacy = _attach_time_to_frame(original, iter_to_time=None, control_file="control")
    assert legacy.time.iloc[0] == value
