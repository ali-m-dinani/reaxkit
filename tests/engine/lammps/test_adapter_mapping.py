from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from reaxkit.domain.data_models import SimulationData, TrajectoryData
from reaxkit.engine.lammps.adapter import LAMMPSAdapter


def _example_dump_path() -> Path:
    repo_root = Path(__file__).resolve().parents[3]
    return repo_root / "examples_to_test" / "dump.xyz"


def _example_log_path() -> Path:
    repo_root = Path(__file__).resolve().parents[3]
    return repo_root / "examples_to_test" / "log.lammps"


def test_lammps_adapter_maps_dump_to_trajectory_data():
    dump_path = _example_dump_path()
    if not dump_path.exists():
        pytest.skip(f"LAMMPS dump file not found: {dump_path}")

    adapter = LAMMPSAdapter()
    out = adapter.load_trajectory({"dump": str(dump_path)})

    assert isinstance(out, TrajectoryData)
    assert out.positions.ndim == 3
    assert out.positions.shape[2] == 3
    assert len(out.atom_ids) == out.positions.shape[1]
    assert len(out.elements) == out.positions.shape[1]
    assert out.iterations is not None
    assert out.iterations.shape[0] == out.positions.shape[0]
    assert np.isfinite(out.positions).any()


def test_lammps_adapter_maps_log_to_simulation_data():
    log_path = _example_log_path()
    if not log_path.exists():
        pytest.skip(f"LAMMPS log file not found: {log_path}")

    try:
        from lammps.formats import LogFile  # noqa: F401
    except Exception:
        pytest.skip("LAMMPS backend unavailable: lammps.formats.LogFile is not installed.")

    adapter = LAMMPSAdapter()
    out = adapter.load_simulation({"log": str(log_path), "dump": str(_example_dump_path())})

    assert isinstance(out, SimulationData)
    assert out.iterations is not None
    assert out.iterations.size > 0
    assert out.temperature is not None
    assert out.pressure is not None
    assert out.density is not None
    assert out.potential_energy is not None
    assert len(out.iterations) == len(out.temperature)
    assert len(out.iterations) == len(out.pressure)
    assert len(out.iterations) == len(out.density)


def test_lammps_adapter_required_inputs_include_dump_xyz_for_trajectory():
    adapter = LAMMPSAdapter()
    required = adapter.required_input_files(TrajectoryData, {})
    assert required is not None
    assert "dump.xyz" in required
