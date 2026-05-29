"""
ReaxFF electric-field regime (eregime.in) generators.

This module provides deterministic utilities for generating ReaxFF
``eregime.in`` files, which define time-dependent external electric
field schedules applied during MD simulations.

**Usage context**

- Template generation: Produce canonical text payloads for ReaxFF artifacts.
- File writing: Persist generated outputs to disk with stable formatting.
- Workflow integration: Support higher-level ReaxKit workflow commands.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence
import math

import numpy as np


__all__ = [
    "HEADER_LINES",
    "ExplicitERegimeSpec",
    "SinusoidalERegimeSpec",
    "SmoothPulseERegimeSpec",
    "FunctionalERegimeSpec",
    "EREGIME_GENERATOR_REGISTRY",
    "gen_eregime",
]


HEADER_LINES: Sequence[str] = (
    "#Electric field regimes\n",
    "#start #V direction Magnitude(V/A)\n",
)


@dataclass(frozen=True)
class ExplicitERegimeSpec:
    """Represent ExplicitERegimeSpec.

    Public class used by ReaxFF generator components.

    Fields
    ------
    rows : tuple[tuple[int, int, str, float], ...]
        Dataclass field.
    """
    rows: tuple[tuple[int, int, str, float], ...]


@dataclass(frozen=True)
class SinusoidalERegimeSpec:
    """Represent SinusoidalERegimeSpec.

    Public class used by ReaxFF generator components.

    Fields
    ------
    max_magnitude : float
        Dataclass field.
    step_angle : float
        Dataclass field.
    iteration_step : int
        Dataclass field.
    num_cycles : float
        Dataclass field.
    direction : str
        Dataclass field.
    voltage_idx : int
        Dataclass field.
    phase : float
        Dataclass field.
    dc_offset : float
        Dataclass field.
    start_iter : int
        Dataclass field.
    """
    max_magnitude: float
    step_angle: float
    iteration_step: int
    num_cycles: float
    direction: str = "z"
    voltage_idx: int = 1
    phase: float = 0.0
    dc_offset: float = 0.0
    start_iter: int = 0


@dataclass(frozen=True)
class SmoothPulseERegimeSpec:
    """Represent SmoothPulseERegimeSpec.

    Public class used by ReaxFF generator components.

    Fields
    ------
    amplitude : float
        Dataclass field.
    width : float
        Dataclass field.
    period : float
        Dataclass field.
    slope : float
        Dataclass field.
    iteration_step : int
        Dataclass field.
    num_of_cycles : int | float
        Dataclass field.
    step_size : float
        Dataclass field.
    direction : str
        Dataclass field.
    voltage_idx : int
        Dataclass field.
    baseline : float
        Dataclass field.
    start_iter : int
        Dataclass field.
    """
    amplitude: float
    width: float
    period: float
    slope: float
    iteration_step: int
    num_of_cycles: int | float
    step_size: float = 0.1
    direction: str = "z"
    voltage_idx: int = 1
    baseline: float = 0.0
    start_iter: int = 0


@dataclass(frozen=True)
class FunctionalERegimeSpec:
    """Represent FunctionalERegimeSpec.

    Public class used by ReaxFF generator components.

    Fields
    ------
    func : Callable[[float], float]
        Dataclass field.
    t_end : float
        Dataclass field.
    dt : float
        Dataclass field.
    iteration_step : int
        Dataclass field.
    direction : str
        Dataclass field.
    voltage_idx : int
        Dataclass field.
    start_iter : int
        Dataclass field.
    """
    func: Callable[[float], float]
    t_end: float
    dt: float
    iteration_step: int
    direction: str = "z"
    voltage_idx: int = 1
    start_iter: int = 0


def _normalize_direction(direction: str) -> str:
    """Normalize direction."""
    d = direction.strip().lower()
    if d not in {"x", "y", "z"}:
        raise ValueError(f"direction must be one of 'x','y','z'; got {direction!r}")
    return d


def _format_eregime_text(rows: Iterable[tuple[int, int, str, float]]) -> str:
    """Format eregime text."""
    lines = list(HEADER_LINES)
    for it, v, d, mag in rows:
        d = _normalize_direction(d)
        lines.append(f"{int(it):6d}     {int(v):d}        {d:<2}              {float(mag): .6f}\n")
    return "".join(lines)


def _gen_eregime_text(spec: ExplicitERegimeSpec) -> str:
    """
    Generate ``eregime.in`` text from explicit row definitions.
    """
    return _format_eregime_text(spec.rows)


def _generate_a_given_eregime(spec: ExplicitERegimeSpec) -> str:
    """
    Backward-compatible alias for ``gen_eregime``.
    """
    return _gen_eregime_text(spec)


def _write_eregime_rows(
    file_path: str | Path,
    rows: Iterable[tuple[int, int, str, float]],
) -> Path:
    """
    Write ``eregime.in`` text from already prepared rows.
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(_format_eregime_text(rows), encoding="utf-8")
    return file_path


def _write_a_given_eregime(
    file_path: str | Path,
    rows: Iterable[tuple[int, int, str, float]],
) -> Path:
    """
    Backward-compatible wrapper for writing explicit ``eregime.in`` rows.
    """
    return _write_eregime_rows(file_path=file_path, rows=rows)


def _generate_eregime_sinusoidal(spec: SinusoidalERegimeSpec) -> str:
    """
    Generate a sinusoidal electric-field schedule as ``eregime.in`` text.
    """
    if spec.step_angle <= 0:
        raise ValueError("step_angle must be > 0")
    if spec.iteration_step <= 0:
        raise ValueError("iteration_step must be > 0")

    direction = _normalize_direction(spec.direction)
    npts = int(round((2.0 * spec.num_cycles * math.pi) / spec.step_angle)) + 1

    rows: list[tuple[int, int, str, float]] = []
    for k in range(npts):
        ang = spec.phase + k * spec.step_angle
        mag = spec.dc_offset + spec.max_magnitude * math.sin(ang)
        it = spec.start_iter + k * spec.iteration_step
        rows.append((it, spec.voltage_idx, direction, float(mag)))

    return _format_eregime_text(rows)


def _write_eregime_sinusoidal(
    file_path: str | Path,
    *,
    max_magnitude: float,
    step_angle: float,
    iteration_step: int,
    num_cycles: float,
    direction: str = "z",
    voltage_idx: int = 1,
    phase: float = 0.0,
    dc_offset: float = 0.0,
    start_iter: int = 0,
) -> Path:
    """
    Backward-compatible wrapper for writing a sinusoidal ``eregime.in`` file.
    """
    spec = SinusoidalERegimeSpec(
        max_magnitude=max_magnitude,
        step_angle=step_angle,
        iteration_step=iteration_step,
        num_cycles=num_cycles,
        direction=direction,
        voltage_idx=voltage_idx,
        phase=phase,
        dc_offset=dc_offset,
        start_iter=start_iter,
    )
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    text = _generate_eregime_sinusoidal(spec)
    file_path.write_text(text, encoding="utf-8")
    return file_path


def _generate_eregime_smooth_pulse(spec: SmoothPulseERegimeSpec) -> str:
    """
    Generate smooth bipolar electric-field pulses as ``eregime.in`` text.
    """
    if spec.period <= 0 or spec.step_size <= 0 or spec.slope < 0 or spec.width < 0:
        raise ValueError("period>0, step_size>0, slope>=0, width>=0 are required")
    if 2 * spec.slope + spec.width > (spec.period / 2):
        raise ValueError("Each half-period must satisfy 2*slope + width <= period/2")
    if spec.iteration_step <= 0:
        raise ValueError("iteration_step must be > 0")

    direction = _normalize_direction(spec.direction)
    total_time = float(spec.num_of_cycles) * spec.period
    t = np.arange(0.0, total_time + spec.step_size, spec.step_size)
    half_period = spec.period / 2.0

    def half_profile(tin: float) -> float:
        """Half profile.

        Parameters
        ----------
        tin : float
            Input parameter.

        Returns
        -------
        float
            Return value.

        Examples
        --------
        ```python
        # Example
        half_profile(...)
        ```
        """
        if tin < spec.slope:
            return (
                spec.baseline + (spec.amplitude / spec.slope) * tin
                if spec.slope > 0
                else spec.baseline + spec.amplitude
            )
        if tin < spec.slope + spec.width:
            return spec.baseline + spec.amplitude
        if tin < 2.0 * spec.slope + spec.width:
            return (
                spec.baseline
                + spec.amplitude
                - (spec.amplitude / spec.slope) * (tin - (spec.slope + spec.width))
                if spec.slope > 0
                else spec.baseline
            )
        return spec.baseline

    rows: list[tuple[int, int, str, float]] = []
    for idx, ti in enumerate(t):
        in_cycle = ti % spec.period
        sign = 1.0 if in_cycle < half_period else -1.0
        tin = in_cycle if sign > 0 else (in_cycle - half_period)
        mag = sign * (half_profile(tin) - spec.baseline) + spec.baseline
        it = spec.start_iter + idx * spec.iteration_step
        rows.append((it, spec.voltage_idx, direction, float(mag)))

    return _format_eregime_text(rows)


def _write_eregime_smooth_pulse(
    file_path: str | Path,
    *,
    amplitude: float,
    width: float,
    period: float,
    slope: float,
    iteration_step: int,
    num_of_cycles: int | float,
    step_size: float = 0.1,
    direction: str = "z",
    voltage_idx: int = 1,
    baseline: float = 0.0,
    start_iter: int = 0,
) -> Path:
    """
    Backward-compatible wrapper for writing a smooth-pulse ``eregime.in`` file.
    """
    spec = SmoothPulseERegimeSpec(
        amplitude=amplitude,
        width=width,
        period=period,
        slope=slope,
        iteration_step=iteration_step,
        num_of_cycles=num_of_cycles,
        step_size=step_size,
        direction=direction,
        voltage_idx=voltage_idx,
        baseline=baseline,
        start_iter=start_iter,
    )
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    text = _generate_eregime_smooth_pulse(spec)
    file_path.write_text(text, encoding="utf-8")
    return file_path


def _generate_eregime_from_function(spec: FunctionalERegimeSpec) -> str:
    """
    Generate an electric-field regime by sampling an arbitrary function.
    """
    if spec.dt <= 0 or spec.t_end < 0:
        raise ValueError("dt must be > 0 and t_end >= 0")
    if spec.iteration_step <= 0:
        raise ValueError("iteration_step must be > 0")

    direction = _normalize_direction(spec.direction)
    t = np.arange(0.0, spec.t_end + spec.dt, spec.dt)

    rows: list[tuple[int, int, str, float]] = []
    for i, ti in enumerate(t):
        mag = float(spec.func(float(ti)))
        it = spec.start_iter + i * spec.iteration_step
        rows.append((it, spec.voltage_idx, direction, mag))

    return _format_eregime_text(rows)


def _write_eregime_from_function(
    file_path: str | Path,
    *,
    func: Callable[[float], float],
    t_end: float,
    dt: float,
    iteration_step: int,
    direction: str = "z",
    voltage_idx: int = 1,
    start_iter: int = 0,
) -> Path:
    """
    Backward-compatible wrapper for writing a function-sampled ``eregime.in`` file.
    """
    spec = FunctionalERegimeSpec(
        func=func,
        t_end=t_end,
        dt=dt,
        iteration_step=iteration_step,
        direction=direction,
        voltage_idx=voltage_idx,
        start_iter=start_iter,
    )
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    text = _generate_eregime_from_function(spec)
    file_path.write_text(text, encoding="utf-8")
    return file_path


def gen_eregime(
    out_path: str | Path = "eregime.in",
    *,
    profile_type: str,
    iteration_step: int,
    direction: str = "z",
    voltage_idx: int = 1,
    start_iter: int = 0,
    max_magnitude: float | None = None,
    step_angle: float | None = None,
    num_cycles: float | None = None,
    phase: float = 0.0,
    dc_offset: float = 0.0,
    amplitude: float | None = None,
    width: float | None = None,
    period: float | None = None,
    slope: float | None = None,
    step_size: float = 0.1,
    baseline: float = 0.0,
    func: Callable[[float], float] | None = None,
    t_end: float | None = None,
    dt: float | None = None,
) -> Path:
    """Gen eregime.

    Parameters
    ----------
    out_path : str | Path, optional
        Input parameter.
    profile_type : str
        Keyword-only parameter.
    iteration_step : int
        Keyword-only parameter.
    direction : str, optional
        Keyword-only parameter.
    voltage_idx : int, optional
        Keyword-only parameter.
    start_iter : int, optional
        Keyword-only parameter.
    max_magnitude : float | None, optional
        Keyword-only parameter.
    step_angle : float | None, optional
        Keyword-only parameter.
    num_cycles : float | None, optional
        Keyword-only parameter.
    phase : float, optional
        Keyword-only parameter.
    dc_offset : float, optional
        Keyword-only parameter.
    amplitude : float | None, optional
        Keyword-only parameter.
    width : float | None, optional
        Keyword-only parameter.
    period : float | None, optional
        Keyword-only parameter.
    slope : float | None, optional
        Keyword-only parameter.
    step_size : float, optional
        Keyword-only parameter.
    baseline : float, optional
        Keyword-only parameter.
    func : Callable[[float], float] | None, optional
        Keyword-only parameter.
    t_end : float | None, optional
        Keyword-only parameter.
    dt : float | None, optional
        Keyword-only parameter.

    Returns
    -------
    Path
        Return value.

    Examples
    --------
    ```python
    # Example
    gen_eregime(...)
    ```
    """
    kind = str(profile_type).strip().lower()
    if kind == "sin":
        if max_magnitude is None or step_angle is None or num_cycles is None:
            raise ValueError("sin profile requires max_magnitude, step_angle, and num_cycles.")
        return _write_eregime_sinusoidal(
            out_path,
            max_magnitude=max_magnitude,
            step_angle=step_angle,
            iteration_step=iteration_step,
            num_cycles=num_cycles,
            direction=direction,
            voltage_idx=voltage_idx,
            phase=phase,
            dc_offset=dc_offset,
            start_iter=start_iter,
        )
    if kind == "pulse":
        required = {"amplitude": amplitude, "width": width, "period": period, "slope": slope, "num_cycles": num_cycles}
        missing = [name for name, value in required.items() if value is None]
        if missing:
            raise ValueError(f"pulse profile missing required arguments: {', '.join(missing)}")
        return _write_eregime_smooth_pulse(
            out_path,
            amplitude=amplitude,
            width=width,
            period=period,
            slope=slope,
            iteration_step=iteration_step,
            num_of_cycles=num_cycles,
            step_size=step_size,
            direction=direction,
            voltage_idx=voltage_idx,
            baseline=baseline,
            start_iter=start_iter,
        )
    if kind == "func":
        if func is None or t_end is None or dt is None:
            raise ValueError("func profile requires func, t_end, and dt.")
        return _write_eregime_from_function(
            out_path,
            func=func,
            t_end=t_end,
            dt=dt,
            iteration_step=iteration_step,
            direction=direction,
            voltage_idx=voltage_idx,
            start_iter=start_iter,
        )
    raise ValueError(f"Unsupported profile_type: {profile_type!r}. Choose from sin|pulse|func.")


EREGIME_GENERATOR_REGISTRY: dict[str, dict[str, Any]] = {
    "eregime_explicit": {
        "label": "Electric Field Regime From Rows",
        "default_filename": "eregime.in",
        "spec_type": ExplicitERegimeSpec,
        "generate": _gen_eregime_text,
        "write": _write_a_given_eregime,
    },
    "eregime_sinusoidal": {
        "label": "Electric Field Regime Sinusoidal",
        "default_filename": "eregime.in",
        "spec_type": SinusoidalERegimeSpec,
        "generate": _generate_eregime_sinusoidal,
        "write": _write_eregime_sinusoidal,
    },
    "eregime_smooth_pulse": {
        "label": "Electric Field Regime Smooth Pulse",
        "default_filename": "eregime.in",
        "spec_type": SmoothPulseERegimeSpec,
        "generate": _generate_eregime_smooth_pulse,
        "write": _write_eregime_smooth_pulse,
    },
    "eregime_from_function": {
        "label": "Electric Field Regime From Function",
        "default_filename": "eregime.in",
        "spec_type": FunctionalERegimeSpec,
        "generate": _generate_eregime_from_function,
        "write": _write_eregime_from_function,
    },
}
