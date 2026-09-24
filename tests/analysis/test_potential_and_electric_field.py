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
from reaxkit.domain.data_models import ChargeData, ElectricFieldData, ElectrostaticsData, TrajectoryData
from reaxkit.presentation.persist import _write_csvs
from reaxkit.workflows.electrostatics.potential_and_electric_field.artifacts import write_binning, write_tables


def _data():
    trajectory = TrajectoryData(
        positions=np.asarray([[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]]),
        elements=["Al", "N"], atom_ids=[1, 2], iterations=np.asarray([5]),
    )
    field = ElectricFieldData(
        applied_field_values=np.asarray([[0.25, 0.0, 0.0]]),
        applied_field_components=("field_x", "field_y", "field_z"),
        sampled_field_iterations=np.asarray([5]),
    )
    return ElectrostaticsData(
        trajectory, ChargeData(np.asarray([[1.0, -1.0]]), iterations=np.asarray([5])),
        electric_field=field,
    )


def test_two_atom_probe_potential_and_analytic_field_exclude_target():
    parameters = ReaxFFCoulombParameters(0.0, 10.0, {"al": 1.2, "n": 1.5})
    result = calculate_frame(
        np.asarray([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]), np.asarray([1.0, -1.0]),
        ["Al", "N"], parameters, probe_elements=["Al"], periodic=(False, False, False),
    )
    kernel = float(shielded_kernel(2.0, 1.2, 1.5,
        np.polyval(list(reversed(parameters.taper_coefficients)), 2.0)))
    assert np.isclose(result.internal_probe_potential_v[0, 0], -kernel / KCAL_PER_MOL_PER_EV)
    kernel_al_al = float(shielded_kernel(2.0, 1.2, 1.2,
        np.polyval(list(reversed(parameters.taper_coefficients)), 2.0)))
    assert np.isclose(result.internal_probe_potential_v[0, 1], kernel_al_al / KCAL_PER_MOL_PER_EV)
    assert np.isclose(result.per_atom_coulomb_kcal_per_mol.sum(), result.total_coulomb_kcal_per_mol)
    assert result.internal_probe_field_v_per_angstrom.shape == (1, 2, 3)


def test_batched_analytic_field_matches_numerical_gradient_with_periodic_images():
    parameters = ReaxFFCoulombParameters(0.0, 6.0, {"al": 1.2, "n": 1.5})
    positions = np.asarray([
        [0.4, 0.5, 0.6], [2.1, 0.8, 1.2], [4.5, 4.6, 1.8], [1.2, 3.9, 3.2],
    ])
    charges = np.asarray([0.4, -0.3, 0.2, -0.3])
    labels = ["Al", "N", "Al", "N"]
    common = dict(probe_elements=["Al", "N"], cell=np.diag([5.0, 5.0, 8.0]),
                  periodic=(True, True, False))
    analytic = calculate_frame(positions, charges, labels, parameters, **common)
    numerical = calculate_frame(
        positions, charges, labels, parameters, field_method="numerical", field_step=1e-5,
        **common,
    )
    assert np.allclose(analytic.internal_probe_potential_v,
                       numerical.internal_probe_potential_v, rtol=0.0, atol=1e-12)
    assert np.allclose(analytic.internal_probe_field_v_per_angstrom,
                       numerical.internal_probe_field_v_per_angstrom, rtol=2e-7, atol=2e-8)


def test_analysis_outputs_species_and_average_tables():
    request = PotentialElectricFieldRequest(
        lower_taper_radius=0.0, upper_taper_radius=10.0,
        gamma_by_symbol={"al": 1.2, "n": 1.5}, periodic=(False, False, False),
    )
    result = calculate_potential_and_field(_data(), request)
    assert set(result.probe_tables) == {"Al", "N"}
    assert len(result.table) == 2
    assert set(result.table["probe_element"]) == {"average"}
    assert "internal_potential (V)" not in result.coulomb_table
    table = result.probe_tables["Al"].sort_values("atom_index")
    assert np.allclose(table["external_potential (V)"], [0.25, -0.25])
    assert np.allclose(table["external_electric_field_x (V/angstrom)"], 0.25)
    assert np.allclose(
        table["total_local_potential (V)"],
        table["internal_potential (V)"] + table["external_potential (V)"],
    )
    assert np.allclose(
        table["total_local_electric_field_x (V/angstrom)"],
        table["internal_electric_field_x (V/angstrom)"] + 0.25,
    )


def test_organized_csvs_suppress_duplicate_root_probe_tables(tmp_path: Path):
    request = PotentialElectricFieldRequest(
        lower_taper_radius=0.0, upper_taper_radius=10.0,
        gamma_by_symbol={"al": 1.2, "n": 1.5}, periodic=(False, False, False),
    )
    result = calculate_potential_and_field(_data(), request)
    paths = write_tables(result, tmp_path)
    recorded = _write_csvs(tmp_path, result)
    assert paths["probe_Al"].is_file()
    assert paths["probe_N"].is_file()
    assert paths["probe_average"].is_file()
    assert not (tmp_path / "probe_Al.csv").exists()
    assert not (tmp_path / "probe_N.csv").exists()
    assert not (tmp_path / "probe_average.csv").exists()
    assert set(recorded) == {
        "coulomb_per_atom.csv", "coulomb_totals.csv",
        str(Path("voltages_and_electric_fields") / "probe_Al_per_atom.csv"),
        str(Path("voltages_and_electric_fields") / "probe_N_per_atom.csv"),
        str(Path("voltages_and_electric_fields") / "probe_average_per_atom.csv"),
    }


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
    assert "internal_potential_V:R:1" in text
    assert "external_potential_V:R:1" in text
    assert "total_local_potential_V:R:1" in text
    assert "internal_electric_field_V_per_A:R:3" in text
    assert "external_electric_field_V_per_A:R:3" in text
    assert "total_local_electric_field_V_per_A:R:3" in text


def test_plot_request_writes_one_plot_per_frame_and_probe(tmp_path: Path):
    data = _data()
    data.trajectory.positions = np.repeat(data.trajectory.positions, 2, axis=0)
    data.trajectory.iterations = np.asarray([0, 5])
    data.charges.charges = np.repeat(data.charges.charges, 2, axis=0)
    data.charges.iterations = np.asarray([0, 5])
    data.electric_field.applied_field_values = np.asarray([[0.0, 0.0, 0.0], [0.25, 0.0, 0.0]])
    data.electric_field.sampled_field_iterations = np.asarray([0, 5])
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


def test_kymograph_writes_potential_and_field_across_frames_and_bins(tmp_path: Path):
    data = _data()
    data.trajectory.positions = np.asarray([
        [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
    ])
    data.trajectory.iterations = np.asarray([5, 10])
    data.charges.charges = np.repeat(data.charges.charges, 2, axis=0)
    data.charges.iterations = np.asarray([5, 10])
    data.electric_field.applied_field_values = np.asarray([[0.25, 0.0, 0.0], [0.5, 0.0, 0.0]])
    data.electric_field.sampled_field_iterations = np.asarray([5, 10])
    request = PotentialElectricFieldRequest(
        lower_taper_radius=0.0, upper_taper_radius=10.0,
        gamma_by_symbol={"al": 1.2, "n": 1.5}, periodic=(False, False, False),
    )
    result = calculate_potential_and_field(data, request)
    paths = write_binning(
        result, tmp_path, axes="x", bins=[3], plot=False,
        component="x", units="v/angstrom", kymograph=True,
        kymograph_values=("total-local-potential", "total-local-field"),
        kymograph_time_axis="iteration", dpi=72,
    )
    plots = [path for path in paths if path.suffix == ".png"]
    assert len(plots) == 6  # Al, N, and average for potential and field.
    assert all(path.is_file() for path in plots)
    assert all("_kymograph_x_" in path.name and "_by_iteration" in path.name for path in plots)


def test_fixed_midpoint_reference_is_reused_when_material_expands():
    data = _data()
    data.trajectory.positions = np.asarray([
        [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        [[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]],
    ])
    data.trajectory.iterations = np.asarray([5, 10])
    data.charges.charges = np.repeat(data.charges.charges, 2, axis=0)
    data.charges.iterations = np.asarray([5, 10])
    data.electric_field.applied_field_values = np.asarray([[0.25, 0.0, 0.0], [0.25, 0.0, 0.0]])
    data.electric_field.sampled_field_iterations = np.asarray([5, 10])
    common = dict(lower_taper_radius=0.0, upper_taper_radius=10.0,
                  gamma_by_symbol={"al": 1.2, "n": 1.5}, periodic=(False, False, False))
    dynamic = calculate_potential_and_field(data, PotentialElectricFieldRequest(**common))
    fixed = calculate_potential_and_field(
        data, PotentialElectricFieldRequest(**common, potential_reference_mode="fixed-midpoint")
    )
    dynamic_second = dynamic.table.loc[dynamic.table["frame_index"].eq(1)]
    fixed_second = fixed.table.loc[fixed.table["frame_index"].eq(1)]
    assert np.allclose(dynamic_second["potential_reference_x (angstrom)"], 2.0)
    assert np.allclose(fixed_second["potential_reference_x (angstrom)"], 1.0)
    assert np.allclose(fixed_second["external_potential (V)"], [0.25, -0.75])
