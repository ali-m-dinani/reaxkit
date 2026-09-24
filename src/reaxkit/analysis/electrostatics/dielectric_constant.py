"""Kubo--Green dielectric response from a scalar dipole time series.

The implementation follows the relation supplied with this feature::

    epsilon(f) - 1 = 1 / (g epsilon_0 V k_B T)
        [C(0) + i 2 pi f integral_0^inf exp(i 2 pi f t) C(t) dt]

where ``C(t)`` is the autocorrelation of the mean-centered dipole moment.
``g`` is three for an isotropic total-dipole observable and one for a single
Cartesian component.  The finite trajectory replaces the infinite integral
with a trapezoidal discrete transform.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from scipy.constants import Boltzmann, c, elementary_charge, epsilon_0
from scipy.signal import correlate

from reaxkit.analysis.base import AnalysisTask
from reaxkit.core.registry.analysis_task_registry import register_task
from reaxkit.domain.base_request import BaseRequest
from reaxkit.domain.base_result import BaseResult

TimeUnit = Literal["s", "ms", "us", "ns", "ps", "fs"]
DipoleUnit = Literal["c-m", "debye", "e-angstrom"]
VolumeUnit = Literal["m3", "cm3", "nm3", "angstrom3"]
FrequencyUnit = Literal["hz", "khz", "mhz", "ghz", "thz", "cm-1"]
DipoleKind = Literal["total", "component"]
DielectricVolumeMethod = Literal["manual", "hull", "bbox", "cell"]

TIME_TO_SECONDS: dict[str, float] = {
    "s": 1.0,
    "ms": 1.0e-3,
    "us": 1.0e-6,
    "ns": 1.0e-9,
    "ps": 1.0e-12,
    "fs": 1.0e-15,
}
DIPOLE_TO_COULOMB_METERS: dict[str, float] = {
    "c-m": 1.0,
    "debye": 3.33564e-30,
    "e-angstrom": elementary_charge * 1.0e-10,
}
VOLUME_TO_CUBIC_METERS: dict[str, float] = {
    "m3": 1.0,
    "cm3": 1.0e-6,
    "nm3": 1.0e-27,
    "angstrom3": 1.0e-30,
}
FREQUENCY_FROM_HZ: dict[str, float] = {
    "hz": 1.0,
    "khz": 1.0e-3,
    "mhz": 1.0e-6,
    "ghz": 1.0e-9,
    "thz": 1.0e-12,
    "cm-1": 1.0 / (c * 100.0),
}


@dataclass
class DielectricConstantRequest(BaseRequest):
    """Inputs and physical units for a Kubo--Green dielectric calculation."""

    temperature: float
    volume: float
    time_unit: TimeUnit
    dipole_unit: DipoleUnit
    volume_unit: VolumeUnit
    time_column: str = "t"
    dipole_column: str = "dipole"
    dipole_kind: DipoleKind = "total"
    frequency_unit: FrequencyUnit = "thz"
    max_lag: float | None = None
    max_frequency: float | None = None
    volume_method: DielectricVolumeMethod = "manual"
    volume_frame_count: int | None = None
    volume_min: float | None = None
    volume_max: float | None = None


@dataclass
class DielectricConstantResult(BaseResult):
    """Static permittivity, complex spectrum, and dipole autocorrelation."""

    static_dielectric_constant: float
    table: pd.DataFrame
    autocorrelation: pd.DataFrame
    summary: pd.DataFrame
    request: DielectricConstantRequest

    @property
    def spectrum(self) -> pd.DataFrame:
        """Return the frequency-domain table under a descriptive alias."""
        return self.table


def _require_supported(value: str, mapping: dict[str, float], label: str) -> float:
    try:
        return mapping[value]
    except KeyError as exc:  # pragma: no cover - argparse normally guards this
        choices = ", ".join(mapping)
        raise ValueError(f"Unsupported {label} '{value}'. Choose one of: {choices}.") from exc


def _validate_series(time: np.ndarray, dipole: np.ndarray) -> float:
    if time.ndim != 1 or dipole.ndim != 1:
        raise ValueError("Time and dipole inputs must be one-dimensional columns.")
    if time.size != dipole.size:
        raise ValueError("Time and dipole columns must contain the same number of rows.")
    if time.size < 3:
        raise ValueError("At least three time/dipole samples are required.")
    if not np.all(np.isfinite(time)) or not np.all(np.isfinite(dipole)):
        raise ValueError("Time and dipole columns must contain only finite numeric values.")

    steps = np.diff(time)
    if np.any(steps <= 0.0):
        raise ValueError("Time values must be strictly increasing.")
    time_step = float(np.mean(steps))
    if not np.allclose(steps, time_step, rtol=1.0e-5, atol=max(abs(time_step) * 1.0e-8, 1.0e-30)):
        raise ValueError(
            "Kubo--Green FFT integration requires uniformly spaced time values; "
            "resample the input series onto a uniform grid."
        )
    return time_step


def _unbiased_autocorrelation(values: np.ndarray) -> np.ndarray:
    """Return the non-negative-lag, time-origin averaged autocorrelation."""
    count = values.size
    numerator = correlate(values, values, mode="full", method="fft")[count - 1 :]
    denominator = np.arange(count, 0, -1, dtype=float)
    return np.asarray(numerator / denominator, dtype=float)


def calculate_dielectric_constant(
    time: np.ndarray,
    dipole: np.ndarray,
    request: DielectricConstantRequest,
) -> DielectricConstantResult:
    """Calculate static and frequency-dependent relative permittivity.

    The input dipole mean is removed because equilibrium dielectric response is
    governed by dipole fluctuations.  A one-column input cannot retain vector
    orientation; ``dipole_kind='total'`` therefore treats the scalar series as
    the total-dipole observable in the supplied isotropic relation, while
    ``dipole_kind='component'`` calculates the directional response.
    """
    time_values = np.asarray(time, dtype=float)
    dipole_values = np.asarray(dipole, dtype=float)
    raw_time_step = _validate_series(time_values, dipole_values)

    if not np.isfinite(request.temperature) or request.temperature <= 0.0:
        raise ValueError("Temperature must be a finite value greater than zero kelvin.")
    if not np.isfinite(request.volume) or request.volume <= 0.0:
        raise ValueError("Volume must be a finite value greater than zero.")
    if request.dipole_kind not in ("total", "component"):
        raise ValueError("dipole_kind must be 'total' or 'component'.")

    time_scale = _require_supported(request.time_unit, TIME_TO_SECONDS, "time unit")
    dipole_scale = _require_supported(
        request.dipole_unit, DIPOLE_TO_COULOMB_METERS, "dipole unit"
    )
    volume_scale = _require_supported(request.volume_unit, VOLUME_TO_CUBIC_METERS, "volume unit")
    frequency_scale = _require_supported(
        request.frequency_unit, FREQUENCY_FROM_HZ, "frequency unit"
    )

    dipole_si = dipole_values * dipole_scale
    fluctuation = dipole_si - float(np.mean(dipole_si))
    time_step_seconds = raw_time_step * time_scale
    autocorrelation_si = _unbiased_autocorrelation(fluctuation)
    lags_seconds = np.arange(autocorrelation_si.size, dtype=float) * time_step_seconds

    if request.max_lag is not None:
        if not np.isfinite(request.max_lag) or request.max_lag <= 0.0:
            raise ValueError("max_lag must be a finite value greater than zero.")
        max_lag_seconds = float(request.max_lag) * time_scale
        keep_lags = lags_seconds <= max_lag_seconds * (1.0 + 1.0e-12)
        autocorrelation_si = autocorrelation_si[keep_lags]
        lags_seconds = lags_seconds[keep_lags]
        if autocorrelation_si.size < 2:
            raise ValueError("max_lag must include at least two autocorrelation samples.")

    normalization_factor = 3.0 if request.dipole_kind == "total" else 1.0
    volume_m3 = float(request.volume) * volume_scale
    response_prefactor = 1.0 / (
        normalization_factor * epsilon_0 * volume_m3 * Boltzmann * float(request.temperature)
    )
    static_dielectric = 1.0 + response_prefactor * float(autocorrelation_si[0])

    frequencies_hz = np.fft.rfftfreq(autocorrelation_si.size, d=time_step_seconds)
    trapezoid_weights = np.ones(autocorrelation_si.size, dtype=float)
    trapezoid_weights[[0, -1]] = 0.5
    # rfft uses exp(-i omega t); conjugation implements the image's
    # exp(+i omega t) convention for a real autocorrelation sequence.
    correlation_integral = time_step_seconds * np.conjugate(
        np.fft.rfft(autocorrelation_si * trapezoid_weights)
    )
    omega = 2.0 * np.pi * frequencies_hz
    epsilon = 1.0 + response_prefactor * (
        autocorrelation_si[0] + 1j * omega * correlation_integral
    )

    frequencies_output = frequencies_hz * frequency_scale
    if request.max_frequency is not None:
        if not np.isfinite(request.max_frequency) or request.max_frequency < 0.0:
            raise ValueError("max_frequency must be a finite value greater than or equal to zero.")
        keep_frequency = frequencies_output <= float(request.max_frequency) * (1.0 + 1.0e-12)
        frequencies_hz = frequencies_hz[keep_frequency]
        frequencies_output = frequencies_output[keep_frequency]
        omega = omega[keep_frequency]
        epsilon = epsilon[keep_frequency]

    frequency_label = request.frequency_unit
    spectrum = pd.DataFrame(
        {
            f"frequency ({frequency_label})": frequencies_output,
            "frequency (Hz)": frequencies_hz,
            "angular frequency (rad/s)": omega,
            "epsilon real": np.real(epsilon),
            "epsilon imaginary": np.imag(epsilon),
            "dielectric loss": np.imag(epsilon),
        }
    )

    dipole_squared_scale = dipole_scale**2
    if autocorrelation_si[0] == 0.0:
        normalized_acf = np.zeros_like(autocorrelation_si)
    else:
        normalized_acf = autocorrelation_si / autocorrelation_si[0]
    autocorrelation = pd.DataFrame(
        {
            f"lag ({request.time_unit})": lags_seconds / time_scale,
            "lag (s)": lags_seconds,
            f"autocorrelation ({request.dipole_unit}^2)": (
                autocorrelation_si / dipole_squared_scale
            ),
            "autocorrelation (C^2 m^2)": autocorrelation_si,
            "normalized autocorrelation": normalized_acf,
        }
    )

    summary = pd.DataFrame(
        [
            {
                "static dielectric constant": static_dielectric,
                "temperature (K)": float(request.temperature),
                f"volume ({request.volume_unit})": float(request.volume),
                "volume (m3)": volume_m3,
                "volume method": request.volume_method,
                "volume frames": request.volume_frame_count,
                f"minimum volume ({request.volume_unit})": request.volume_min,
                f"maximum volume ({request.volume_unit})": request.volume_max,
                "samples": int(time_values.size),
                f"time step ({request.time_unit})": raw_time_step,
                "time step (s)": time_step_seconds,
                f"mean dipole ({request.dipole_unit})": float(np.mean(dipole_values)),
                f"dipole variance ({request.dipole_unit}^2)": float(
                    np.mean((dipole_values - np.mean(dipole_values)) ** 2)
                ),
                "dipole kind": request.dipole_kind,
                "normalization factor": normalization_factor,
                "frequency convention": "exp(+i*2*pi*f*t)",
            }
        ]
    )
    return DielectricConstantResult(
        static_dielectric_constant=float(static_dielectric),
        table=spectrum,
        autocorrelation=autocorrelation,
        summary=summary,
        request=request,
    )


@register_task("get-dielectric-constant", label="Dielectric Constant")
class DielectricConstantTask(AnalysisTask):
    """Analyze a DataFrame containing time and scalar dipole columns."""

    required_data = pd.DataFrame

    def run(
        self,
        data: pd.DataFrame,
        request: DielectricConstantRequest,
        reporter=None,
    ) -> DielectricConstantResult:
        missing = [
            column
            for column in (request.time_column, request.dipole_column)
            if column not in data.columns
        ]
        if missing:
            available = ", ".join(str(column) for column in data.columns)
            raise ValueError(
                f"Missing Excel column(s): {', '.join(missing)}. Available columns: {available}."
            )
        time = pd.to_numeric(data[request.time_column], errors="coerce").to_numpy(dtype=float)
        dipole = pd.to_numeric(data[request.dipole_column], errors="coerce").to_numpy(dtype=float)
        if callable(reporter):
            reporter("analyze", 0, len(time), "Calculating dipole autocorrelation")
        result = calculate_dielectric_constant(time, dipole, request)
        if callable(reporter):
            reporter("analyze", len(time), len(time), "Calculated dielectric response")
        return result


__all__ = [
    "DIPOLE_TO_COULOMB_METERS",
    "FREQUENCY_FROM_HZ",
    "TIME_TO_SECONDS",
    "VOLUME_TO_CUBIC_METERS",
    "DielectricConstantRequest",
    "DielectricConstantResult",
    "DielectricConstantTask",
    "calculate_dielectric_constant",
]
