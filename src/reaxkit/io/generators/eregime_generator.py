"""
ReaxFF electric-field regime (eregime.in) generators.

This module provides deterministic utilities for generating ReaxFF
``eregime.in`` files, which define time-dependent external electric
field schedules applied during MD simulations.

Generators in this module:
---

- write fully formatted ``eregime.in`` files
- do not parse simulation output
- do not run simulations or perform analysis
"""


from __future__ import annotations
from pathlib import Path
from typing import Callable, Iterable, Tuple, Union, Sequence
import math
import numpy as np

__all__ = [
    "_write_a_given_eregime",
    "write_eregime_sinusoidal",
    "write_eregime_smooth_pulse",
    "write_eregime_from_function",
]

HEADER_LINES: Sequence[str] = (
    "#Electric field regimes\n",
    "#start #V direction Magnitude(V/A)\n",
)


def _normalize_direction(direction: str) -> str:
    d = direction.strip().lower()
    if d not in {"x", "y", "z"}:
        raise ValueError(f"direction must be one of 'x','y','z'; got {direction!r}")
    return d


def _write_header(f) -> None:
    for line in HEADER_LINES:
        f.write(line)


def _write_a_given_eregime(
    file_path: Union[str, Path],
    rows: Iterable[Tuple[int, int, str, float]],
) -> Path:
    """
    Write an ``eregime.in`` file from explicit row definitions.

    Each row defines the electric-field magnitude and direction applied
    at a given iteration.

    Parameters
    ----------
    file_path : str | Path
        Output path (e.g. ``"eregime.in"``).
    rows : iterable of (iteration, V_index, direction, magnitude)
        Electric-field schedule entries, where:
        - iteration : int
            Simulation iteration number.
        - V_index : int
            Voltage index column expected by ReaxFF.
        - direction : {"x","y","z"}
            Field direction.
        - magnitude : float
            Field magnitude in V/Å.

    Returns
    -------
    Path
        The resolved path of the written ``eregime.in`` file.

    Examples
    ---
    >>> rows = [(0, 1, "z", 0.01), (100, 1, "z", -0.01)]
    >>> _write_a_given_eregime("eregime.in", rows)
    PosixPath('eregime.in')
    """
    file_path = Path(file_path)
    with open(file_path, "w") as f:
        _write_header(f)
        for it, v, d, mag in rows:
            d = _normalize_direction(d)
            f.write(f"{int(it):6d}     {int(v):d}        {d:<2}              {float(mag): .6f}\n")
    return file_path


# -----------------------------------------------------------------------------
# Generators
# -----------------------------------------------------------------------------

def write_eregime_sinusoidal(
    file_path: Union[str, Path],
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
    Generate a sinusoidal electric-field schedule and write ``eregime.in``.

    The generated field follows:
        E(t) = dc_offset + max_magnitude · sin(phase + k · step_angle)

    sampled at fixed angular increments.

    Parameters
    ----------
    max_magnitude : float
        Peak field amplitude in V/Å.
    step_angle : float
        Angular step size in radians.
    iteration_step : int
        Iteration increment between successive samples.
    num_cycles : float
        Total number of sinusoidal cycles.
    direction : {"x","y","z"}, optional
        Field direction (default: ``"z"``).
    voltage_idx : int, optional
        Voltage index column value (default: 1).
    phase : float, optional
        Phase offset in radians.
    dc_offset : float, optional
        Constant offset added to the field (V/Å).
    start_iter : int, optional
        Starting iteration index.

    Returns
    -------
    Path
        The written ``eregime.in`` file path.

    Examples
    ---
    >>> write_eregime_sinusoidal(
    ...     "eregime.in",
    ...     max_magnitude=0.05,
    ...     step_angle=0.1,
    ...     iteration_step=10,
    ...     num_cycles=2
    ... )
    """
    if step_angle <= 0:
        raise ValueError("step_angle must be > 0")
    if iteration_step <= 0:
        raise ValueError("iteration_step must be > 0")

    direction = _normalize_direction(direction)
    npts = int(round((2.0 * num_cycles * math.pi) / step_angle)) + 1

    rows = []
    for k in range(npts):
        ang = phase + k * step_angle
        mag = dc_offset + max_magnitude * math.sin(ang)
        it = start_iter + k * iteration_step
        rows.append((it, voltage_idx, direction, float(mag)))

    out = _write_a_given_eregime(file_path, rows)
    print(f"[Done] Sinusoidal eregime saved to {out} ({npts} entry rows).")
    return out


def write_eregime_smooth_pulse(
    file_path: Union[str, Path],
    *,
    amplitude: float,
    width: float,
    period: float,
    slope: float,
    iteration_step: int,
    num_of_cycles: Union[int, float],
    step_size: float = 0.1,
    direction: str = "z",
    voltage_idx: int = 1,
    baseline: float = 0.0,
    start_iter: int = 0,
) -> Path:
    """
    Generate smooth bipolar electric-field pulses and write ``eregime.in``.

    Each cycle consists of a positive half-cycle followed by a mirrored
    negative half-cycle, with linear ramps and flat plateaus.

    Parameters
    ----------
    amplitude : float
        Peak field magnitude in V/Å (positive value).
    width : float
        Flat-top duration at peak amplitude.
    period : float
        Full cycle duration.
    slope : float
        Ramp-up and ramp-down duration.
    iteration_step : int
        Iteration increment per sample.
    num_of_cycles : int | float
        Number of cycles to generate.
    step_size : float, optional
        Time resolution for sampling.
    direction : {"x","y","z"}, optional
        Field direction.
    voltage_idx : int, optional
        Voltage index column value.
    baseline : float, optional
        Baseline field offset in V/Å.
    start_iter : int, optional
        Starting iteration index.

    Returns
    -------
    Path
        The written ``eregime.in`` file path.

    Examples
    ---
    >>> write_eregime_smooth_pulse(
    ...     "eregime.in",
    ...     amplitude=0.1,
    ...     width=5.0,
    ...     period=20.0,
    ...     slope=2.0,
    ...     iteration_step=10,
    ...     num_of_cycles=3
    ... )
    """
    if period <= 0 or step_size <= 0 or slope < 0 or width < 0:
        raise ValueError("period>0, step_size>0, slope>=0, width>=0 are required")
    if 2 * slope + width > (period / 2):
        raise ValueError("Each half-period must satisfy 2*slope + width <= period/2")
    if iteration_step <= 0:
        raise ValueError("iteration_step must be > 0")

    direction = _normalize_direction(direction)

    total_time = float(num_of_cycles) * period
    t = np.arange(0.0, total_time + step_size, step_size)
    halfT = period / 2.0

    def half_profile(tin: float) -> float:
        if tin < slope:
            return baseline + (amplitude / slope) * tin if slope > 0 else baseline + amplitude
        if tin < slope + width:
            return baseline + amplitude
        if tin < 2.0 * slope + width:
            return baseline + amplitude - (amplitude / slope) * (tin - (slope + width)) if slope > 0 else baseline
        return baseline

    rows = []
    for idx, ti in enumerate(t):
        in_cycle = ti % period
        sign = 1.0 if in_cycle < halfT else -1.0
        tin = in_cycle if sign > 0 else (in_cycle - halfT)
        mag = sign * (half_profile(tin) - baseline) + baseline
        it = start_iter + idx * iteration_step
        rows.append((it, voltage_idx, direction, float(mag)))

    out = _write_a_given_eregime(file_path, rows)
    print(f"[Done] Smooth pulse eregime saved to {out} ({len(rows)} entry rows).")
    return out


def write_eregime_from_function(
    file_path: Union[str, Path],
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
    Generate an electric-field regime from an arbitrary function and
    write ``eregime.in``.

    The function ``func(t)`` is sampled uniformly in time and mapped
    to simulation iterations.

    Parameters
    ----------
    func : Callable[[float], float]
        Function returning electric-field magnitude (V/Å) at time ``t``.
    t_end : float
        End time for sampling.
    dt : float
        Time step for sampling.
    iteration_step : int
        Iteration increment per sample.
    direction : {"x","y","z"}, optional
        Field direction.
    voltage_idx : int, optional
        Voltage index column value.
    start_iter : int, optional
        Starting iteration index.

    Returns
    -------
    Path
        The written ``eregime.in`` file path.

    Examples
    ---
    >>> f = lambda t: 0.02 * t
    >>> write_eregime_from_function(
    ...     "eregime.in",
    ...     func=f,
    ...     t_end=10.0,
    ...     dt=0.5,
    ...     iteration_step=5
    ... )
    """
    if dt <= 0 or t_end < 0:
        raise ValueError("dt must be > 0 and t_end >= 0")
    if iteration_step <= 0:
        raise ValueError("iteration_step must be > 0")

    direction = _normalize_direction(direction)

    t = np.arange(0.0, t_end + dt, dt)
    rows = []
    for i, ti in enumerate(t):
        mag = float(func(float(ti)))
        it = start_iter + i * iteration_step
        rows.append((it, voltage_idx, direction, mag))

    out = _write_a_given_eregime(file_path, rows)
    print(f"[Done] Functional eregime saved to {out} ({len(rows)} entry rows).")
    return out

