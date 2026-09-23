import numpy as np
import pytest

from reaxkit.analysis.ferroelectrics.switching_kinetics import (
    KAI_CHARACTERISTIC_FRACTION,
    estimate_kai_t0,
    fit_switching_model,
    kai_fraction,
    nls_fraction,
    snng_fraction,
    snng_fraction_physical,
)


def test_model_endpoints_and_physical_snng_prefactor_agree():
    time = np.linspace(0.0, 10.0, 51)
    assert kai_fraction(time, 2.0, 2.0)[0] == 0.0
    assert nls_fraction(time, 2.0, 0.4, 2.0)[0] == 0.0
    assert snng_fraction(time, 0.3, 0.8, 1.2)[0] == 0.0

    physical = snng_fraction_physical(time, 2.0, 3.0, 4.0, 0.8, 1.2)
    combined = snng_fraction(time, 2.0 * np.pi * 2.0 * 3.0**2 * 4.0, 0.8, 1.2)
    np.testing.assert_allclose(physical, combined)


def test_kai_fit_recovers_synthetic_parameters():
    time = np.linspace(0.0, 8.0, 81)
    fraction = kai_fraction(time, t0=2.4, n=1.7)
    result = fit_switching_model(time, fraction, "kai", n_starts=4)

    assert result.parameters["t0"] == pytest.approx(2.4, rel=1e-4)
    assert result.parameters["n"] == pytest.approx(1.7, rel=1e-4)
    assert result.metrics["rmse"] < 1e-8
    assert result.metrics["r_squared"] > 0.999999


def test_kai_t0_is_interpolated_at_characteristic_fraction():
    time = np.linspace(0.0, 8.0, 81)
    fraction = kai_fraction(time, t0=2.4, n=1.7)

    assert KAI_CHARACTERISTIC_FRACTION == pytest.approx(0.6321205588)
    assert estimate_kai_t0(time, fraction) == pytest.approx(2.4, rel=2e-4)

    with pytest.raises(ValueError, match="never reach"):
        estimate_kai_t0(time[:10], fraction[:10])


def test_nls_fit_supports_literature_fixed_exponent():
    time = np.linspace(0.0, 12.0, 80)
    fraction = nls_fraction(time, t1=3.0, width=0.35, n=2.0)
    result = fit_switching_model(
        time,
        fraction,
        "nls",
        fixed={"n": 2.0, "amplitude": 1.0},
        n_starts=4,
    )

    assert result.parameters["t1"] == pytest.approx(3.0, rel=2e-3)
    assert result.parameters["width"] == pytest.approx(0.35, rel=2e-2)
    assert result.metrics["rmse"] < 1e-5
    assert "n" not in result.standard_errors


def test_snng_fit_recovers_prefactor_and_alpha_when_m_is_fixed():
    time = np.linspace(0.0, 6.0, 61)
    fraction = snng_fraction(time, prefactor=0.42, alpha=0.75, m=1.3)
    result = fit_switching_model(
        time,
        fraction,
        "snng",
        fixed={"m": 1.3},
        initial={"prefactor": 0.5, "alpha": 0.8},
        n_starts=4,
    )

    assert result.parameters["prefactor"] == pytest.approx(0.42, rel=2e-3)
    assert result.parameters["alpha"] == pytest.approx(0.75, rel=2e-3)
    assert result.metrics["rmse"] < 1e-7
