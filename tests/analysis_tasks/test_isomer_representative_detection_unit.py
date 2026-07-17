"""Tests for canonical isomer representative detection."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pytest

from reaxkit.analysis.molecular_analysis.isomer_representative_detection import (
    IsomerRepresentativeDetectionRequest,
    IsomerRepresentativeDetectionTask,
)
from reaxkit.domain.data_models import ConnectivityData, ConnectivityTrajectoryData, SimulationData, TrajectoryData
from reaxkit.engine.reaxff.adapter import ReaxFFAdapter
from reaxkit.engine.reaxff.io.fort7_handler import Fort7Handler
from reaxkit.workflows.file_tools import isomer_workflow


FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "reaxff_isomer_representatives_detection"


def _synthetic_canonical_data() -> ConnectivityTrajectoryData:
    """Build a two-frame canonical bundle with two H2O representatives."""
    positions = np.asarray(
        [
            [[0.0, 0.0, 0.0], [0.9, 0.0, 0.0], [-0.9, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [0.9, 0.0, 0.0], [1.8, 0.0, 0.0]],
        ],
        dtype=float,
    )
    atom_ids = [1, 2, 3]
    elements = ["O", "H", "H"]
    iterations = np.asarray([0, 100], dtype=int)
    molecule_nums = np.asarray([[1, 1, 1], [1, 1, 1]], dtype=int)
    simulation = SimulationData(
        atom_ids=atom_ids,
        iterations=iterations,
        elements=elements,
        molecule_nums=molecule_nums,
    )
    trajectory = TrajectoryData(
        positions=positions,
        elements=elements,
        atom_ids=atom_ids,
        iterations=iterations,
        simulation=simulation,
    )
    water_star = np.asarray(
        [
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ],
        dtype=float,
    )
    water_chain = np.asarray(
        [
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=float,
    )
    connectivity = ConnectivityData(
        connectivity=np.asarray([water_star, water_chain], dtype=float),
        bond_orders=np.asarray([water_star, water_chain], dtype=float),
        atom_ids=atom_ids,
        elements=elements,
        iterations=iterations,
        simulation=simulation,
    )
    return ConnectivityTrajectoryData(connectivity=connectivity, trajectory=trajectory)


def _fixture_paths() -> tuple[Path, Path, Path]:
    return (
        FIXTURE_DIR / "fort.7",
        FIXTURE_DIR / "xmolout",
        FIXTURE_DIR / "control_params",
    )


def test_task_detects_representatives_from_canonical_connectivity_trajectory_data() -> None:
    data = _synthetic_canonical_data()
    request = IsomerRepresentativeDetectionRequest(
        target_formula={"H": 2, "O": 1},
        structure_prefix="H2O",
    )

    result = IsomerRepresentativeDetectionTask().run(data, request)

    assert list(result.table["structure_name"]) == ["H2O_0", "H2O_1"]
    assert list(result.table["iteration"]) == [0, 100]
    assert list(result.table["molecule_id"]) == [1, 1]
    assert list(result.table["bond_signature"]) == ["H-O:2", "H-H:1;H-O:1"]


def test_task_honors_max_representatives() -> None:
    data = _synthetic_canonical_data()
    request = IsomerRepresentativeDetectionRequest(
        target_formula={"H": 2, "O": 1},
        structure_prefix="H2O",
        max_representatives=1,
    )

    result = IsomerRepresentativeDetectionTask().run(data, request)

    assert list(result.table["structure_name"]) == ["H2O_0"]
    assert list(result.table["iteration"]) == [0]


def test_task_requires_canonical_molecule_assignments() -> None:
    data = _synthetic_canonical_data()
    data.connectivity.simulation = SimulationData(
        atom_ids=[1, 2, 3],
        iterations=np.asarray([0, 100], dtype=int),
        elements=["O", "H", "H"],
    )
    request = IsomerRepresentativeDetectionRequest(target_formula={"H": 2, "O": 1})

    with pytest.raises(ValueError, match="molecule_nums"):
        IsomerRepresentativeDetectionTask().run(data, request)


def test_parse_legacy_control_params_for_file_workflow() -> None:
    _, _, control_path = _fixture_paths()

    control = isomer_workflow.parse_legacy_isomer_representative_control(control_path)

    assert control.atom_map == {"C": 1, "H": 2, "O": 3, "B": 4}
    assert control.input_formula == {"C": 8, "H": 13, "O": 3, "B": 5}
    assert control.isomer_run == 1
    assert control.isomer_prefixname == "C8H13O3B5"


def test_reaxff_adapter_fixture_feeds_canonical_task() -> None:
    fort7, xmolout, _ = _fixture_paths()
    data = ReaxFFAdapter().load_connectivity_trajectory({"fort7": str(fort7), "xmolout": str(xmolout)})
    request = IsomerRepresentativeDetectionRequest(
        target_formula={"C": 8, "H": 13, "O": 3, "B": 5},
        structure_prefix="C8H13O3B5",
    )

    result = IsomerRepresentativeDetectionTask().run(data, request)

    assert list(result.table["structure_name"]) == [
        "C8H13O3B5_0",
        "C8H13O3B5_1",
        "C8H13O3B5_2",
        "C8H13O3B5_3",
        "C8H13O3B5_4",
        "C8H13O3B5_5",
    ]
    assert list(result.table["iteration"]) == [0, 200, 1500, 2800, 3800, 15100]
    assert list(result.table["molecule_id"]) == [1, 1, 1, 1, 1, 1]


def test_file_workflow_runs_adapter_then_writes_outputs(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    fort7, xmolout, control = _fixture_paths()
    output_dir = tmp_path / "workflow_out"
    args = argparse.Namespace(
        fort7=str(fort7),
        xmolout=str(xmolout),
        control=str(control),
        output_dir=str(output_dir),
        write_isomer_dirs=True,
        no_isomer_dirs=False,
        max_representatives=3,
    )

    assert isomer_workflow.run_main("detect-isomer-representatives", args) == 0

    captured = capsys.readouterr()
    assert "detect-isomer-representatives: detected 3 isomer representatives" in captured.out
    assert (output_dir / "xmolout_isomers").is_file()
    assert (output_dir / "isomers" / "C8H13O3B5_2" / "xmolout").is_file()
    assert not (output_dir / "isomers" / "C8H13O3B5_3" / "xmolout").exists()


def test_file_workflow_validates_required_input_files(tmp_path: Path) -> None:
    fort7, xmolout, control = _fixture_paths()

    with pytest.raises(FileNotFoundError, match="fort.7 file not found"):
        isomer_workflow.detect_isomer_representatives_from_reaxff_files(
            fort7_path=tmp_path / "missing_fort.7",
            xmolout_path=xmolout,
            control_path=control,
            output_dir=tmp_path / "out",
        )

    with pytest.raises(FileNotFoundError, match="xmolout file not found"):
        isomer_workflow.detect_isomer_representatives_from_reaxff_files(
            fort7_path=fort7,
            xmolout_path=tmp_path / "missing_xmolout",
            control_path=control,
            output_dir=tmp_path / "out",
        )


def test_fort7_handler_accepts_unspaced_large_iteration_header(tmp_path: Path) -> None:
    fort7 = tmp_path / "fort.7"
    fort7.write_text(
        "\n".join(
            [
                "       1 H2                               Iteration:9999900 #Bonds:        0",
                "1 1 1 0.0 0.0 0.0",
                "0.0 0.0 0.0 0.0",
                "       1 H2                               Iteration:10000000 #Bonds:        0",
                "1 1 1 0.0 0.0 0.0",
                "0.0 0.0 0.0 0.0",
                "",
            ]
        ),
        encoding="utf-8",
    )

    handler = Fort7Handler(fort7)
    frame = handler.dataframe()

    assert handler.n_frames() == 2
    assert frame["iter"].tolist() == [9999900, 10000000]
