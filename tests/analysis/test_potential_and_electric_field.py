from pathlib import Path

import numpy as np

from reaxkit.analysis.electrostatics.potential_and_electric_field.analysis import (
    PotentialElectricFieldRequest, calculate_potential_and_field,
)
from reaxkit.analysis.electrostatics.potential_and_electric_field.calculation import calculate_frame
from reaxkit.analysis.electrostatics.potential_and_electric_field.parameters import ReaxFFCoulombParameters
from reaxkit.analysis.electrostatics.potential_and_electric_field.physics import KCAL_PER_MOL_PER_EV, shielded_kernel
from reaxkit.analysis.electrostatics.potential_and_electric_field.trajectory import (
    PotentialElectricFieldTrajectoryRequest, PotentialElectricFieldTrajectoryTask,
)
from reaxkit.domain.data_models import ChargeData, ElectrostaticsData, TrajectoryData
from reaxkit.workflows.electrostatics.potential_and_electric_field.artifacts import write_binning


def _data():
    trajectory = TrajectoryData(
        positions=np.asarray([[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]]),
        elements=["Al", "N"], atom_ids=[1, 2], iterations=np.asarray([5]),
    )
    return ElectrostaticsData(trajectory, ChargeData(np.asarray([[1.0, -1.0]]), iterations=np.asarray([5])))


def test_two_atom_probe_potential_and_analytic_field_exclude_target():
    parameters = ReaxFFCoulombParameters(0.0, 10.0, {"al": 1.2, "n": 1.5})
    result = calculate_frame(
        np.asarray([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]), np.asarray([1.0, -1.0]),
        ["Al", "N"], parameters, probe_elements=["Al"], periodic=(False, False, False),
    )
    kernel = float(shielded_kernel(2.0, 1.2, 1.5,
        np.polyval(list(reversed(parameters.taper_coefficients)), 2.0)))
    assert np.isclose(result.probe_potential_v[0, 0], -kernel / KCAL_PER_MOL_PER_EV)
    kernel_al_al = float(shielded_kernel(2.0, 1.2, 1.2,
        np.polyval(list(reversed(parameters.taper_coefficients)), 2.0)))
    assert np.isclose(result.probe_potential_v[0, 1], kernel_al_al / KCAL_PER_MOL_PER_EV)
    assert np.isclose(result.per_atom_coulomb_kcal_per_mol.sum(), result.total_coulomb_kcal_per_mol)
    assert result.probe_field_v_per_angstrom.shape == (1, 2, 3)


def test_analysis_outputs_species_and_average_tables():
    request = PotentialElectricFieldRequest(
        lower_taper_radius=0.0, upper_taper_radius=10.0,
        gamma_by_symbol={"al": 1.2, "n": 1.5}, periodic=(False, False, False),
    )
    result = calculate_potential_and_field(_data(), request)
    assert set(result.probe_tables) == {"Al", "N"}
    assert len(result.table) == 2
    assert set(result.table["probe_element"]) == {"average"}
    assert "potential (V)" not in result.coulomb_table


def test_trajectory_writer_includes_local_vector_properties(tmp_path: Path):
    output = tmp_path / "local.extxyz"
    request = PotentialElectricFieldTrajectoryRequest(
        lower_taper_radius=0.0, upper_taper_radius=10.0,
        gamma_by_symbol={"al": 1.2, "n": 1.5}, periodic=(False, False, False),
        _output_path=str(output),
    )
    result = PotentialElectricFieldTrajectoryTask().run(_data(), request)
    text = output.read_text(encoding="utf-8")
    assert result.frame_indices.tolist() == [0]
    assert "local_potential_V:R:1" in text
    assert "local_electric_field_V_per_A:R:3" in text


def test_plot_request_writes_one_plot_per_frame_and_probe(tmp_path: Path):
    data = _data()
    data.trajectory.positions = np.repeat(data.trajectory.positions, 2, axis=0)
    data.trajectory.iterations = np.asarray([0, 5])
    data.charges.charges = np.repeat(data.charges.charges, 2, axis=0)
    data.charges.iterations = np.asarray([0, 5])
    request = PotentialElectricFieldRequest(
        lower_taper_radius=0.0, upper_taper_radius=10.0,
        gamma_by_symbol={"al": 1.2, "n": 1.5}, periodic=(False, False, False),
    )
    result = calculate_potential_and_field(data, request)
    paths = write_binning(result, tmp_path, axes="x", bins=[2], plot=True,
                          component="x", units="mv/cm")
    plots = [path for path in paths if path.suffix == ".png"]
    assert len(plots) == 6  # Al, N, and average for each of two frames.
    assert {path.name.rsplit("_", 1)[-1] for path in plots} == {"0.png", "1.png"}
