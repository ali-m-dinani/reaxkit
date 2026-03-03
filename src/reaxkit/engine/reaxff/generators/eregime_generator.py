"""
ReaxFF electric-field regime (eregime.in) generators.

This module provides deterministic utilities for generating ReaxFF
``eregime.in`` files, which define time-dependent external electric
field schedules applied during MD simulations.
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
    "generate_a_given_eregime",
    "generate_eregime_sinusoidal",
    "generate_eregime_smooth_pulse",
    "generate_eregime_from_function",
    "write_eregime_rows",
    "write_a_given_eregime",
    "write_eregime_sinusoidal",
    "write_eregime_smooth_pulse",
    "write_eregime_from_function",
]


HEADER_LINES: Sequence[str] = (
    "#Electric field regimes\n",
    "#start #V direction Magnitude(V/A)\n",
)


@dataclass(frozen=True)
class ExplicitERegimeSpec:
    rows: tuple[tuple[int, int, str, float], ...]


@dataclass(frozen=True)
class SinusoidalERegimeSpec:
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
    func: Callable[[float], float]
    t_end: float
    dt: float
    iteration_step: int
    direction: str = "z"
    voltage_idx: int = 1
    start_iter: int = 0


def _normalize_direction(direction: str) -> str:
    d = direction.strip().lower()
    if d not in {"x", "y", "z"}:
        raise ValueError(f"direction must be one of 'x','y','z'; got {direction!r}")
    return d


def _format_eregime_text(rows: Iterable[tuple[int, int, str, float]]) -> str:
    lines = list(HEADER_LINES)
    for it, v, d, mag in rows:
        d = _normalize_direction(d)
        lines.append(f"{int(it):6d}     {int(v):d}        {d:<2}              {float(mag): .6f}\n")
    return "".join(lines)


def generate_a_given_eregime(spec: ExplicitERegimeSpec) -> str:
    """
    Generate ``eregime.in`` text from explicit row definitions.
    """
    return _format_eregime_text(spec.rows)


def write_eregime_rows(
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


def write_a_given_eregime(
    file_path: str | Path,
    rows: Iterable[tuple[int, int, str, float]],
) -> Path:
    """
    Backward-compatible wrapper for writing explicit ``eregime.in`` rows.
    """
    return write_eregime_rows(file_path=file_path, rows=rows)


def generate_eregime_sinusoidal(spec: SinusoidalERegimeSpec) -> str:
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


def write_eregime_sinusoidal(
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
    text = generate_eregime_sinusoidal(spec)
    file_path.write_text(text, encoding="utf-8")
    return file_path


def generate_eregime_smooth_pulse(spec: SmoothPulseERegimeSpec) -> str:
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


def write_eregime_smooth_pulse(
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
    text = generate_eregime_smooth_pulse(spec)
    file_path.write_text(text, encoding="utf-8")
    return file_path


def generate_eregime_from_function(spec: FunctionalERegimeSpec) -> str:
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


def write_eregime_from_function(
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
    text = generate_eregime_from_function(spec)
    file_path.write_text(text, encoding="utf-8")
    return file_path


EREGIME_GENERATOR_REGISTRY: dict[str, dict[str, Any]] = {
    "eregime_explicit": {
        "label": "Electric Field Regime From Rows",
        "default_filename": "eregime.in",
        "spec_type": ExplicitERegimeSpec,
        "generate": generate_a_given_eregime,
        "write": write_a_given_eregime,
    },
    "eregime_sinusoidal": {
        "label": "Electric Field Regime Sinusoidal",
        "default_filename": "eregime.in",
        "spec_type": SinusoidalERegimeSpec,
        "generate": generate_eregime_sinusoidal,
        "write": write_eregime_sinusoidal,
    },
    "eregime_smooth_pulse": {
        "label": "Electric Field Regime Smooth Pulse",
        "default_filename": "eregime.in",
        "spec_type": SmoothPulseERegimeSpec,
        "generate": generate_eregime_smooth_pulse,
        "write": write_eregime_smooth_pulse,
    },
    "eregime_from_function": {
        "label": "Electric Field Regime From Function",
        "default_filename": "eregime.in",
        "spec_type": FunctionalERegimeSpec,
        "generate": generate_eregime_from_function,
        "write": write_eregime_from_function,
    },
}
