import numpy as np
import pandas as pd
import pytest
from scipy.constants import Boltzmann, epsilon_0
from scipy.integrate import trapezoid

from reaxkit.analysis.electrostatics.dielectric_constant import (
    DIPOLE_TO_COULOMB_METERS,
    VOLUME_TO_CUBIC_METERS,
    DielectricConstantRequest,
    DielectricConstantTask,
    calculate_dielectric_constant,
)
from reaxkit.analysis.electrostatics.electrostatics import (
    calculate_trajectory_volumes,
)
from reaxkit.domain.data_models import SimulationData, TrajectoryData


def _request(**overrides):
    values = {
        "temperature": 300.0,
        "volume": 1000.0,
        "time_unit": "fs",
        "dipole_unit": "debye",
        "volume_unit": "angstrom3",
    }
    values.update(overrides)
    return DielectricConstantRequest(**values)


def test_static_dielectric_matches_dipole_fluctuation_relation():
    time = np.arange(8, dtype=float)
    dipole = np.asarray([-2.0, -1.0, 0.0, 1.0, 2.0, 1.0, 0.0, -1.0])
    request = _request()

    result = calculate_dielectric_constant(time, dipole, request)

    variance_si = np.var(dipole) * DIPOLE_TO_COULOMB_METERS["debye"] ** 2
    volume_si = request.volume * VOLUME_TO_CUBIC_METERS["angstrom3"]
    expected = 1.0 + variance_si / (3.0 * epsilon_0 * volume_si * Boltzmann * 300.0)
    assert result.static_dielectric_constant == pytest.approx(expected)
    assert result.spectrum.iloc[0]["epsilon real"] == pytest.approx(expected)
    assert result.spectrum.iloc[0]["epsilon imaginary"] == pytest.approx(0.0)
    assert result.autocorrelation.iloc[0]["normalized autocorrelation"] == pytest.approx(1.0)


def test_complex_spectrum_matches_attached_kubo_green_integral():
    time = np.arange(8, dtype=float)
    dipole = np.asarray([-2.0, -1.0, 0.0, 1.0, 2.0, 1.0, 0.0, -1.0])
    request = _request()
    result = calculate_dielectric_constant(time, dipole, request)

    row = result.spectrum.iloc[1]
    frequency = float(row["frequency (Hz)"])
    lags = result.autocorrelation["lag (s)"].to_numpy()
    acf = result.autocorrelation["autocorrelation (C^2 m^2)"].to_numpy()
    integral = trapezoid(acf * np.exp(1j * 2.0 * np.pi * frequency * lags), x=lags)
    volume_si = request.volume * VOLUME_TO_CUBIC_METERS["angstrom3"]
    prefactor = 1.0 / (3.0 * epsilon_0 * volume_si * Boltzmann * request.temperature)
    expected = 1.0 + prefactor * (acf[0] + 1j * 2.0 * np.pi * frequency * integral)

    assert row["epsilon real"] == pytest.approx(expected.real)
    assert row["epsilon imaginary"] == pytest.approx(expected.imag)


def test_component_response_uses_directional_prefactor():
    time = np.arange(8, dtype=float)
    dipole = np.asarray([-2.0, -1.0, 0.0, 1.0, 2.0, 1.0, 0.0, -1.0])
    isotropic = calculate_dielectric_constant(time, dipole, _request(dipole_kind="total"))
    component = calculate_dielectric_constant(time, dipole, _request(dipole_kind="component"))

    assert component.static_dielectric_constant - 1.0 == pytest.approx(
        3.0 * (isotropic.static_dielectric_constant - 1.0)
    )


def test_time_dipole_and_volume_unit_conversions_are_physically_equivalent():
    time_fs = 2.0 * np.arange(8, dtype=float)
    dipole_debye = np.asarray([-2.0, -1.0, 0.0, 1.0, 2.0, 1.0, 0.0, -1.0])
    result_debye = calculate_dielectric_constant(time_fs, dipole_debye, _request())

    debye_per_e_angstrom = (
        DIPOLE_TO_COULOMB_METERS["debye"]
        / DIPOLE_TO_COULOMB_METERS["e-angstrom"]
    )
    result_e_angstrom = calculate_dielectric_constant(
        time_fs / 1000.0,
        dipole_debye * debye_per_e_angstrom,
        _request(
            time_unit="ps",
            dipole_unit="e-angstrom",
            volume=1.0,
            volume_unit="nm3",
        ),
    )

    assert result_e_angstrom.static_dielectric_constant == pytest.approx(
        result_debye.static_dielectric_constant
    )
    assert result_e_angstrom.spectrum["frequency (Hz)"].to_numpy() == pytest.approx(
        result_debye.spectrum["frequency (Hz)"].to_numpy()
    )


def test_task_selects_named_dataframe_columns():
    data = pd.DataFrame({"time_ps": np.arange(6), "mu": [0.0, 1.0, 0.0, -1.0, 0.0, 1.0]})
    request = _request(time_unit="ps", time_column="time_ps", dipole_column="mu")

    result = DielectricConstantTask().run(data, request)

    assert len(result.autocorrelation) == len(data)
    assert result.summary.loc[0, "samples"] == len(data)


def test_nonuniform_time_grid_is_rejected():
    with pytest.raises(ValueError, match="uniformly spaced"):
        calculate_dielectric_constant(
            np.asarray([0.0, 1.0, 2.1, 3.0]),
            np.asarray([0.0, 1.0, 0.0, -1.0]),
            _request(),
        )


def test_trajectory_volume_supports_hull_bbox_and_cell():
    cube = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
        ]
    )
    positions = np.stack([cube, 2.0 * cube])
    simulation = SimulationData(
        atom_ids=list(range(1, 9)),
        iterations=np.asarray([0, 1]),
        cell_lengths=np.asarray([[10.0, 10.0, 10.0], [20.0, 20.0, 20.0]]),
        cell_angles=np.full((2, 3), 90.0),
    )
    trajectory = TrajectoryData(
        positions=positions,
        elements=["C"] * 8,
        atom_ids=list(range(1, 9)),
        iterations=np.asarray([0, 1]),
        simulation=simulation,
    )

    assert calculate_trajectory_volumes(trajectory, "hull") == pytest.approx([1.0, 8.0])
    assert calculate_trajectory_volumes(trajectory, "bbox") == pytest.approx([1.0, 8.0])
    assert calculate_trajectory_volumes(trajectory, "cell") == pytest.approx([1000.0, 8000.0])
