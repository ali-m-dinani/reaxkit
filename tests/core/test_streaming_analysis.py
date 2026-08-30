"""Behavioral tests for bounded-memory frame analysis."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from reaxkit.analysis.active_sites import (
    ActiveSiteEventDiagnosticsRequest,
    ActiveSiteEventDiagnosticsTask,
    ActiveSiteEventsRequest,
    ActiveSiteEventsTask,
)
from reaxkit.analysis.electrostatics.electrostatics import (
    DipoleRequest,
    DipoleTask,
    PolarizationFieldRequest,
    PolarizationFieldTask,
)
from reaxkit.analysis.electrostatics.charges import ChargeTableRequest, ChargeTableTask
from reaxkit.analysis.connectivity.connectivity import ConnectionListRequest, ConnectionListTask
from reaxkit.analysis.molecular_analysis.isomer_representative_detection import (
    IsomerRepresentativeDetectionRequest,
    IsomerRepresentativeDetectionTask,
)
from reaxkit.analysis.timeseries.timeseries import (
    ChargeSeriesRequest,
    ChargeSeriesTask,
    TrajectoryCoordinateSeriesRequest,
    TrajectoryCoordinateSeriesTask,
    TrajectoryDisplacementSeriesRequest,
    TrajectoryDisplacementSeriesTask,
)
from reaxkit.core.runtime.analysis_executor import AnalysisExecutor
from reaxkit.domain.data_models import (
    ChargeData,
    ConnectivityData,
    ConnectivityTrajectoryData,
    ElectrostaticsData,
    ElectricFieldData,
    SimulationData,
    TrajectoryData,
)
from reaxkit.engine.reaxff.adapter import ReaxFFAdapter
from reaxkit.engine.reaxff.io.fort7_handler import Fort7Handler
from reaxkit.engine.reaxff.io.xmolout_handler import XmoloutHandler


FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "reaxff_isomer_representatives_detection"


def _trajectory_frames(positions: np.ndarray, elements: list[str], iterations: list[int]):
    atom_ids = list(range(1, len(elements) + 1))
    for source_index, iteration in enumerate(iterations):
        simulation = SimulationData(
            atom_ids=atom_ids,
            iterations=np.asarray([iteration]),
            elements=elements,
            cell_lengths=np.asarray([[20.0, 20.0, 20.0]]),
            cell_angles=np.asarray([[90.0, 90.0, 90.0]]),
        )
        yield TrajectoryData(
            positions=positions[source_index:source_index + 1],
            elements=elements,
            atom_ids=atom_ids,
            iterations=np.asarray([iteration]),
            simulation=simulation,
            source_frame_indices=np.asarray([source_index]),
        )


def test_reaxff_file_streams_do_not_populate_handler_frame_caches():
    xmol = XmoloutHandler(FIXTURE_DIR / "xmolout", frame_indices=[0, 1])
    fort7 = Fort7Handler(FIXTURE_DIR / "fort.7", frame_indices=[0, 1])

    coordinate_frames = list(xmol.stream_file_frames())
    connectivity_frames = list(fort7.stream_file_frames())

    assert [frame["source_index"] for frame in coordinate_frames] == [0, 1]
    assert [frame["source_index"] for frame in connectivity_frames] == [0, 1]
    assert xmol._frames == []
    assert fort7._frames == []
    assert not xmol._parsed
    assert not fort7._parsed


def test_charge_only_fort7_stream_recovers_fused_large_neighbor_ids(tmp_path):
    fort7_path = tmp_path / "fort.7"
    fort7_path.write_text(
        """    28880 slab Iteration: 0 #Bonds: 5
    1    2    3    8   8827364    0    1  0.537  0.546  0.546  0.545  0.000  4.013  0.000  1.188
 0.0 0.0 0.0 0.0
""",
        encoding="utf-8",
    )

    with np.testing.assert_raises(ValueError):
        list(Fort7Handler(fort7_path).stream_file_frames())

    records = list(Fort7Handler(fort7_path).stream_file_frames(charges_only=True))
    assert len(records) == 1
    assert records[0]["connectivity_incomplete"] is True
    row = records[0]["frame"].iloc[0]
    assert int(row["atom_num"]) == 1
    assert int(row["atom_type_num"]) == 2
    assert all(int(row[f"atom_cnn{slot}"]) == 0 for slot in range(1, 6))
    assert float(row["partial_charge"]) == 1.188


def test_dipole_stream_matches_materialized_result():
    positions = np.asarray(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        ]
    )
    charges = np.asarray([[-1.0, 1.0], [-1.0, 1.0]])
    elements = ["O", "H"]
    iterations = [0, 10]
    full_sim = SimulationData(atom_ids=[1, 2], iterations=np.asarray(iterations), elements=elements)
    full = ElectrostaticsData(
        trajectory=TrajectoryData(
            positions=positions,
            elements=elements,
            atom_ids=[1, 2],
            iterations=np.asarray(iterations),
            simulation=full_sim,
        ),
        charges=ChargeData(charges=charges, iterations=np.asarray(iterations), simulation=full_sim),
    )

    def stream():
        for index, trajectory in enumerate(_trajectory_frames(positions, elements, iterations)):
            yield ElectrostaticsData(
                trajectory=trajectory,
                charges=ChargeData(
                    charges=charges[index:index + 1],
                    iterations=np.asarray([iterations[index]]),
                    simulation=trajectory.simulation,
                ),
            )

    request = DipoleRequest(scope="total", frames=None)
    expected = DipoleTask().run(full, request).table
    actual = DipoleTask().run_stream(stream(), request).table
    pd.testing.assert_frame_equal(actual, expected)


def test_polarization_field_stream_matches_materialized_result():
    positions = np.asarray(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            [[0.0, 0.0, 0.0], [1.1, 0.0, 0.0], [0.0, 1.1, 0.0], [0.0, 0.0, 1.1]],
            [[0.0, 0.0, 0.0], [1.2, 0.0, 0.0], [0.0, 1.2, 0.0], [0.0, 0.0, 1.2]],
        ],
        dtype=float,
    )
    charges = np.asarray([[-1.0, 0.2, 0.3, 0.5]] * 3, dtype=float)
    elements = ["C", "H", "H", "H"]
    iterations = [0, 10, 20]
    simulation = SimulationData(atom_ids=[1, 2, 3, 4], iterations=np.asarray(iterations), elements=elements)
    field = ElectricFieldData(
        applied_field_values=np.asarray([[-0.1], [0.0], [0.1]]),
        applied_field_components=["field_z"],
        sampled_field_iterations=np.asarray(iterations),
    )
    full = ElectrostaticsData(
        trajectory=TrajectoryData(
            positions=positions,
            elements=elements,
            atom_ids=[1, 2, 3, 4],
            iterations=np.asarray(iterations),
            simulation=simulation,
        ),
        charges=ChargeData(charges=charges, iterations=np.asarray(iterations), simulation=simulation),
        electric_field=field,
    )

    def stream():
        for index, trajectory in enumerate(_trajectory_frames(positions, elements, iterations)):
            yield ElectrostaticsData(
                trajectory=trajectory,
                charges=ChargeData(
                    charges=charges[index:index + 1],
                    iterations=np.asarray([iterations[index]]),
                    simulation=trajectory.simulation,
                ),
                electric_field=field,
            )

    request = PolarizationFieldRequest(aggregate="mean", field_direction="z")
    expected = PolarizationFieldTask().run(full, request)
    actual = PolarizationFieldTask().run_stream(stream(), request)

    pd.testing.assert_frame_equal(actual.full_table, expected.full_table)
    pd.testing.assert_frame_equal(actual.aggregated_table, expected.aggregated_table)
    assert actual.polarization_zero_crossings == expected.polarization_zero_crossings
    assert actual.field_zero_crossings == expected.field_zero_crossings


def test_active_site_stream_preserves_persistence_across_frames():
    positions = np.asarray(
        [
            [[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [1.3, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [2.5, 0.0, 0.0]],
        ]
    )
    iterations = [0, 10, 20]
    simulation = SimulationData(
        atom_ids=[1, 2],
        iterations=np.asarray(iterations),
        elements=["C", "O"],
        cell_lengths=np.asarray([[20.0, 20.0, 20.0]] * 3),
        cell_angles=np.asarray([[90.0, 90.0, 90.0]] * 3),
    )
    full = TrajectoryData(
        positions=positions,
        elements=["C", "O"],
        atom_ids=[1, 2],
        iterations=np.asarray(iterations),
        simulation=simulation,
    )
    request = ActiveSiteEventsRequest(mode="dist", persist=2, r_CO=1.65)

    expected = ActiveSiteEventsTask().run(full, request)
    actual = ActiveSiteEventsTask().run_stream(
        _trajectory_frames(positions, ["C", "O"], iterations), request
    )

    pd.testing.assert_frame_equal(actual.table, expected.table)
    assert actual.summary == expected.summary


def test_active_site_diagnostics_stream_matches_materialized_result():
    positions = np.asarray(
        [
            [[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [1.3, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [2.5, 0.0, 0.0]],
        ]
    )
    iterations = [0, 10, 20]
    simulation = SimulationData(
        atom_ids=[1, 2],
        iterations=np.asarray(iterations),
        elements=["C", "O"],
        cell_lengths=np.asarray([[20.0, 20.0, 20.0]] * 3),
        cell_angles=np.asarray([[90.0, 90.0, 90.0]] * 3),
    )
    full = TrajectoryData(
        positions=positions,
        elements=["C", "O"],
        atom_ids=[1, 2],
        iterations=np.asarray(iterations),
        simulation=simulation,
    )
    request = ActiveSiteEventDiagnosticsRequest(r_probe=1.65, max_diag_frames=10)

    expected = ActiveSiteEventDiagnosticsTask().run(full, request)
    actual = ActiveSiteEventDiagnosticsTask().run_stream(
        _trajectory_frames(positions, ["C", "O"], iterations), request
    )

    pd.testing.assert_frame_equal(actual.distance_table, expected.distance_table)
    pd.testing.assert_frame_equal(actual.episode_table, expected.episode_table)
    assert actual.summary == expected.summary


def test_reaxff_isomer_stream_retains_only_representative_coordinates():
    adapter = ReaxFFAdapter()
    request = IsomerRepresentativeDetectionRequest(
        target_formula={"C": 8, "H": 13, "O": 3, "B": 5},
        max_representatives=3,
    )
    result = IsomerRepresentativeDetectionTask().run_stream(
        adapter.stream(
            ConnectivityTrajectoryData,
            {
                "xmolout": str(FIXTURE_DIR / "xmolout"),
                "fort7": str(FIXTURE_DIR / "fort.7"),
                "progress": False,
            },
        ),
        request,
    )

    assert len(result.records) == 3
    assert [record.frame_index for record in result.records] == [0, 2, 15]
    assert all(len(record.coordinates) == record.atom_count for record in result.records)


def test_executor_uses_streaming_for_all_frame_dipole():
    args = {
        "run_dir": str(FIXTURE_DIR),
        "xmolout": str(FIXTURE_DIR / "xmolout"),
        "fort7": str(FIXTURE_DIR / "fort.7"),
        "engine": "reaxff",
        "cache": False,
        "progress": False,
    }
    result = AnalysisExecutor().run(
        DipoleTask(),
        DipoleRequest(scope="total", frames=None),
        args,
    )

    assert args["_streaming"] is True
    assert len(result.table) > 1
    assert result.table["frame_index"].is_monotonic_increasing


def test_stream_cache_identity_includes_explicit_lammps_and_ams_paths(tmp_path):
    dump = tmp_path / "trajectory.dump"
    rkf = tmp_path / "ams.rkf"
    dump.write_text("dump", encoding="utf-8")
    rkf.write_text("rkf", encoding="utf-8")

    identity = AnalysisExecutor._stream_source_identity(
        ReaxFFAdapter(),
        TrajectoryData,
        {"dump": str(dump), "rkf": str(rkf), "run_dir": str(tmp_path / "missing")},
        (),
    )

    assert {Path(item["path"]) for item in identity["sources"]} == {
        dump.resolve(),
        rkf.resolve(),
    }


def test_reaxff_charge_stream_matches_materialized_charge_table():
    adapter = ReaxFFAdapter()
    args = {
        "xmolout": str(FIXTURE_DIR / "xmolout"),
        "fort7": str(FIXTURE_DIR / "fort.7"),
        "progress": False,
    }
    request = ChargeTableRequest(frames=None, every=7, atom_ids=[1, 2])

    expected = ChargeTableTask().run(adapter.load(ChargeData, args), request).table
    actual = ChargeTableTask().run_stream(adapter.stream(ChargeData, args), request).table

    pd.testing.assert_frame_equal(actual, expected)


def test_reaxff_connectivity_stream_matches_materialized_connection_list():
    adapter = ReaxFFAdapter()
    args = {"fort7": str(FIXTURE_DIR / "fort.7"), "progress": False}
    request = ConnectionListRequest(frames=None, every=11, min_bo=0.8, undirected=True)

    expected = ConnectionListTask().run(adapter.load(ConnectivityData, args), request).table
    actual = ConnectionListTask().run_stream(adapter.stream(ConnectivityData, args), request).table

    pd.testing.assert_frame_equal(actual, expected)


def test_reaxff_trajectory_series_streams_match_materialized_results():
    adapter = ReaxFFAdapter()
    args = {"xmolout": str(FIXTURE_DIR / "xmolout"), "progress": False}
    full = adapter.load(TrajectoryData, args)

    coordinate_request = TrajectoryCoordinateSeriesRequest(
        atom_ids=[1, 2], dims=["x", "xyz"], every=5
    )
    expected_coordinates = TrajectoryCoordinateSeriesTask().run(full, coordinate_request).table
    actual_coordinates = TrajectoryCoordinateSeriesTask().run_stream(
        adapter.stream(TrajectoryData, args), coordinate_request
    ).table
    pd.testing.assert_frame_equal(actual_coordinates, expected_coordinates)

    displacement_request = TrajectoryDisplacementSeriesRequest(
        atom_ids=[1, 2], dims=["x", "xyz"], reference_frame=2, every=5
    )
    expected_displacements = TrajectoryDisplacementSeriesTask().run(full, displacement_request).table
    actual_displacements = TrajectoryDisplacementSeriesTask().run_stream(
        adapter.stream(TrajectoryData, args), displacement_request
    ).table
    pd.testing.assert_frame_equal(actual_displacements, expected_displacements)


def test_reaxff_charge_series_stream_matches_materialized_result():
    adapter = ReaxFFAdapter()
    args = {
        "xmolout": str(FIXTURE_DIR / "xmolout"),
        "fort7": str(FIXTURE_DIR / "fort.7"),
        "progress": False,
    }
    request = ChargeSeriesRequest(atom_ids=[1, 2], every=6)

    expected = ChargeSeriesTask().run(adapter.load(ChargeData, args), request).table
    actual = ChargeSeriesTask().run_stream(adapter.stream(ChargeData, args), request).table

    pd.testing.assert_frame_equal(actual, expected)
