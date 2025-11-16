"""handler for generating an eregime.in file"""
from __future__ import annotations
from pathlib import Path
from typing import Callable, Iterable, Tuple, Union, Sequence
import math
import numpy as np

__all__ = [
    "write_eregime",
    "make_eregime_sinusoidal",
    "make_eregime_smooth_pulse",
    "make_eregime_from_function",
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


def write_eregime(
    file_path: Union[str, Path],
    rows: Iterable[Tuple[int, int, str, float]],
) -> Path:
    """Write an `eregime.in`-style file from rows.

    Parameters
    ----------
    file_path : str | Path
        Output path (e.g., "eregime.in").
    rows : iterable of (iteration, V_index, direction, magnitude)
        Direction ∈ {x,y,z}; magnitude in V/Å.

    Returns
    -------
    Path
        The written path.
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

def make_eregime_sinusoidal(
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
    """Generate a sinusoidal electric-field schedule and write to file.

    E(t) = dc_offset + max_magnitude * sin(phase + k * step_angle), sampled at fixed angle steps.

    Parameters
    ----------
    max_magnitude : float
        Peak amplitude in V/Å.
    step_angle : float
        Sampling step in radians.
    iteration_step : int
        Iteration increment between adjacent samples.
    num_cycles : float
        Total number of cycles.
    direction : {"x","y","z"}
        Field direction (default: "z").
    voltage_idx : int
        V index column value (AMS/ReaxFF expects integer; default: 1).
    phase : float
        Phase offset in radians.
    dc_offset : float
        Adds a constant offset (V/Å) to the waveform.
    start_iter : int
        Starting iteration (default: 0).
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

    out = write_eregime(file_path, rows)
    print(f"[Done] Sinusoidal eregime saved to {out} ({npts} points)")
    return out


def make_eregime_smooth_pulse(
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
    """Generate smooth bipolar pulses (+/- amplitude) and write to file.

    Half-cycle profile (time domain):
      0 → slope            : ramp  baseline → baseline + amplitude
      slope → slope+width  : hold  baseline + amplitude
      slope+width → 2*slope+width : ramp down to baseline
      remainder            : baseline (until half-period completes), then mirrored negative half-cycle

    Parameters
    ----------
    amplitude : float
        Peak amplitude in V/Å (positive number; sign handled by half-cycles).
    width : float
        Flat-top duration at peak (time units).
    period : float
        Full-cycle duration (time units).
    slope : float
        Ramp duration up/down (time units).
    iteration_step : int
        Iteration increment per sample.
    num_of_cycles : int | float
        Number of cycles to generate.
    step_size : float
        Temporal resolution (time units/sample).
    direction : {"x","y","z"}
        Field direction.
    voltage_idx : int
        V index column value.
    baseline : float
        Baseline magnitude (V/Å) added to the whole waveform.
    start_iter : int
        Starting iteration.
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

    out = write_eregime(file_path, rows)
    print(f"[Done] Smooth pulse eregime saved to {out} with {len(rows)} points.")
    return out


def make_eregime_from_function(
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
    """Generate an eregime from an arbitrary function `func(t)`.

    Parameters
    ----------
    func : Callable[[float], float]
        Function returning magnitude (V/Å) at time t.
    t_end : float
        End time (inclusive sampling with dt).
    dt : float
        Time step for sampling (must be > 0).
    iteration_step : int
        Iteration increment per sample.
    direction : {"x","y","z"}
        Field direction.
    voltage_idx : int
        V index column value.
    start_iter : int
        Starting iteration.
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

    out = write_eregime(file_path, rows)
    print(f"[Done] Functional eregime saved to {out} with {len(rows)} points.")
    return out

