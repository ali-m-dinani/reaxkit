"""
ReaxFF electric-field regime (eregime.in) generators.

This module provides deterministic utilities for generating ReaxFF
``eregime.in`` files, which define time-dependent external electric
field schedules applied during MD simulations.

Notes
-----
Worth mentioning that eregime.in can have a maximum of 100 entry lines. So, if you are working with
a relatively long eregime profile, you may need to decrease the number of sampled points, or run
multiple simulations using restart files.

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
        Ordered explicit schedule rows in ``(iteration, voltage_idx, direction, magnitude)``
        form used verbatim when writing ``eregime.in``.

    Examples
    --------
    ```python
    spec = ExplicitERegimeSpec(
        rows=(
            (0, 1, "z", 0.0),
            (500, 1, "z", 0.25),
            (1000, 1, "z", -0.25),
        )
    )
    ```
    This means you provide every row yourself: at iteration 0 the field is 0.0 V/A,
    at 500 it is +0.25 V/A, and at 1000 it is -0.25 V/A, all on voltage index 1
    and along the z direction.
    """
    rows: tuple[tuple[int, int, str, float], ...]


@dataclass(frozen=True)
class SinusoidalERegimeSpec:
    """Represent SinusoidalERegimeSpec.

    Public class used by ReaxFF generator components.

    Notes
    -----
    In points-per-cycle mode, adjacent cycles share their baseline boundary,
    so the total number of rows is
    ``num_cycles * (points_per_cycle - 1) + 1``. Magnitudes change linearly
    within each quarter-cycle. Legacy step-angle mode samples a true sine.

    Fields
    ------
    max_magnitude : float
        Peak sinusoidal amplitude (V/A) around ``dc_offset``.
    step_angle : float | None
        Legacy angular increment (radians) between consecutive samples. Use
        ``points_per_cycle`` for schedules that land exactly on each half-cycle
        and full-cycle baseline.
    iteration_step : int
        MD-iteration increment between consecutive output rows.
    num_cycles : float
        Number of sinusoidal cycles to generate.
    points_per_cycle : int | None
        Number of rows in one complete piecewise-linear cycle, counting both
        boundary rows. The start, half-cycle, and end are exactly
        ``dc_offset``. Odd values produce equally sampled positive and negative
        half-cycles; even values place the extra interval in the negative half.
    direction : str
        Field axis label (``"x"``, ``"y"``, or ``"z"``).
    voltage_idx : int
        ReaxFF voltage slot index written to the second ``eregime.in`` column.
    phase : float
        Initial phase offset in radians applied to the sine argument.
    dc_offset : float
        Constant baseline value (V/A) added to the sinusoid.
    start_iter : int
        Iteration number for the first generated row.

    Examples
    --------
    ```python
    spec = SinusoidalERegimeSpec(
        max_magnitude=0.4,
        step_angle=None,
        iteration_step=100,
        num_cycles=2.0,
        points_per_cycle=9,
        direction="z",
        voltage_idx=1,
        phase=0.0,
        dc_offset=0.0,
        start_iter=0,
    )
    ```
    Plain-language meaning of each value:
    ``max_magnitude=0.4`` means the signal gets as high or as low as about 0.4 V/A
    around the offset. ``points_per_cycle=9`` samples each complete
    piecewise-linear cycle at nine rows, including its boundaries.
    ``iteration_step=100`` means each sampled point is 100 MD steps apart.
    ``num_cycles=2.0`` means generate two full sine cycles.
    ``direction="z"`` applies the field along z.
    ``voltage_idx=1`` writes to voltage column/index 1 in ReaxFF.
    ``phase=0.0`` starts the sine at zero phase shift.
    ``dc_offset=0.0`` adds no constant bias.
    ``start_iter=0`` starts writing rows at MD iteration 0.
    """
    max_magnitude: float
    step_angle: float | None
    iteration_step: int
    num_cycles: float
    points_per_cycle: int | None = None
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
        Pulse height above ``baseline`` (V/A) during the positive half-cycle.
    width : float
        Duration of the flat pulse plateau within each half-cycle.
    period : float
        Full pulse period in the profile time domain.
    slope : float
        Rise/fall ramp duration used to smooth pulse edges.
    iteration_step : int
        MD-iteration increment between consecutive sampled rows.
    num_of_cycles : int | float
        Number of full pulse periods to generate.
    step_size : float
        Sampling interval in profile time units.
    direction : str
        Field axis label (``"x"``, ``"y"``, or ``"z"``).
    voltage_idx : int
        ReaxFF voltage slot index written to the second ``eregime.in`` column.
    baseline : float
        Base field value around which bipolar pulses oscillate.
    start_iter : int
        Iteration number for the first generated row.

    Examples
    --------
    ```python
    spec = SmoothPulseERegimeSpec(
        amplitude=0.6,
        width=1.0,
        period=6.0,
        slope=0.5,
        iteration_step=50,
        num_of_cycles=3,
        step_size=0.1,
        direction="z",
        voltage_idx=1,
        baseline=0.0,
        start_iter=0,
    )
    ```
    Plain-language meaning of each value:
    ``amplitude=0.6`` sets the pulse height above baseline.
    ``width=1.0`` is how long the flat top lasts.
    ``period=6.0`` is the full positive+negative cycle length.
    ``slope=0.5`` smooths edges by ramping up/down over 0.5 time units.
    ``iteration_step=50`` means output points are 50 MD steps apart.
    ``num_of_cycles=3`` generates three full pulse cycles.
    ``step_size=0.1`` controls time-resolution of sampling.
    ``direction="z"`` applies the field along z.
    ``voltage_idx=1`` writes to voltage column/index 1.
    ``baseline=0.0`` centers pulses around zero field.
    ``start_iter=0`` starts writing from iteration 0.
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
        Callable mapping sample time to field magnitude (V/A).
    t_end : float
        Final sample time (inclusive) in the function time domain.
    dt : float
        Sampling interval passed to ``func``.
    iteration_step : int
        MD-iteration increment between consecutive sampled rows.
    direction : str
        Field axis label (``"x"``, ``"y"``, or ``"z"``).
    voltage_idx : int
        ReaxFF voltage slot index written to the second ``eregime.in`` column.
    start_iter : int
        Iteration number for the first generated row.

    Examples
    --------
    ```python
    spec = FunctionalERegimeSpec(
        func=lambda t: 0.2 * math.sin(t),
        t_end=10.0,
        dt=0.1,
        iteration_step=20,
        direction="z",
        voltage_idx=1,
        start_iter=0,
    )
    ```
    Plain-language meaning of each value:
    ``func`` is the rule used to compute field magnitude at each sampled time.
    Here it is a sine with amplitude 0.2 V/A.
    ``t_end=10.0`` means sample up to time 10.0.
    ``dt=0.1`` means sample every 0.1 time units.
    ``iteration_step=20`` maps consecutive samples to rows 20 MD steps apart.
    ``direction="z"`` applies the field along z.
    ``voltage_idx=1`` writes to voltage column/index 1.
    ``start_iter=0`` starts row indexing at iteration 0.
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
    Generate a sine-sampled or equal-increment electric-field schedule.
    """
    if spec.iteration_step <= 0:
        raise ValueError("iteration_step must be > 0")
    if spec.points_per_cycle is not None and spec.step_angle is not None:
        raise ValueError("Use either points_per_cycle or step_angle, not both")
    if spec.points_per_cycle is None and spec.step_angle is None:
        raise ValueError("points_per_cycle or step_angle is required")

    direction = _normalize_direction(spec.direction)
    rows: list[tuple[int, int, str, float]] = []

    if spec.points_per_cycle is not None:
        points_per_cycle = spec.points_per_cycle
        if isinstance(points_per_cycle, bool) or not isinstance(points_per_cycle, int):
            raise ValueError("points_per_cycle must be an integer")
        if points_per_cycle < 5:
            raise ValueError(
                "points_per_cycle must be >= 5 to include a positive sample, "
                "a negative sample, and three baseline points"
            )
        if spec.num_cycles <= 0 or not float(spec.num_cycles).is_integer():
            raise ValueError("num_cycles must be a positive integer with points_per_cycle")
        if not math.isclose(
            math.remainder(spec.phase, 2.0 * math.pi),
            0.0,
            abs_tol=1.0e-12,
        ):
            raise ValueError(
                "phase must be 0 (or a whole multiple of 2*pi) with points_per_cycle "
                "so cycle and half-cycle rows return to dc_offset"
            )

        # Both halves include their endpoints and are divided into two linear
        # quarter-cycle ramps. This gives equal magnitude changes while rising
        # or falling, and includes the exact positive/negative extrema.
        positive_half_points = (points_per_cycle + 1) // 2
        negative_half_points = points_per_cycle - positive_half_points + 1

        def half_cycle_magnitudes(sign: float, point_count: int) -> list[float]:
            interval_count = point_count - 1
            first_quarter_intervals = interval_count // 2
            second_quarter_intervals = interval_count - first_quarter_intervals
            peak = spec.dc_offset + sign * spec.max_magnitude
            magnitudes = [
                spec.dc_offset
                + (peak - spec.dc_offset) * index / first_quarter_intervals
                for index in range(first_quarter_intervals + 1)
            ]
            magnitudes.extend(
                peak
                + (spec.dc_offset - peak) * index / second_quarter_intervals
                for index in range(1, second_quarter_intervals + 1)
            )
            return magnitudes

        cycle_magnitudes = half_cycle_magnitudes(1.0, positive_half_points)
        cycle_magnitudes.extend(half_cycle_magnitudes(-1.0, negative_half_points)[1:])

        sample_index = 0
        baseline_indexes = {0, positive_half_points - 1, points_per_cycle - 1}
        for cycle_index in range(int(spec.num_cycles)):
            for cycle_sample_index, magnitude in enumerate(cycle_magnitudes):
                if cycle_index > 0 and cycle_sample_index == 0:
                    continue
                if cycle_sample_index in baseline_indexes:
                    magnitude = spec.dc_offset
                iteration = spec.start_iter + sample_index * spec.iteration_step
                rows.append((iteration, spec.voltage_idx, direction, float(magnitude)))
                sample_index += 1
    else:
        assert spec.step_angle is not None
        if spec.step_angle <= 0:
            raise ValueError("step_angle must be > 0")
        npts = int(round((2.0 * spec.num_cycles * math.pi) / spec.step_angle)) + 1
        for k in range(npts):
            angle = spec.phase + k * spec.step_angle
            magnitude = spec.dc_offset + spec.max_magnitude * math.sin(angle)
            iteration = spec.start_iter + k * spec.iteration_step
            rows.append((iteration, spec.voltage_idx, direction, float(magnitude)))

    return _format_eregime_text(rows)


def _write_eregime_sinusoidal(
    file_path: str | Path,
    *,
    max_magnitude: float,
    iteration_step: int,
    num_cycles: float,
    points_per_cycle: int | None = None,
    step_angle: float | None = None,
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
        points_per_cycle=points_per_cycle,
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
    points_per_cycle: int | None = None,
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
        Destination file path for the generated ``eregime.in`` payload.
    profile_type : str
        Profile selector: ``"sin"``, ``"pulse"``, or ``"func"``.
    iteration_step : int
        MD-iteration spacing between successive rows in the schedule.
    direction : str, optional
        Field direction written to the regime file (``"x"``, ``"y"``, ``"z"``).
    voltage_idx : int, optional
        ReaxFF voltage slot index written in column two.
    start_iter : int, optional
        Iteration index assigned to the first row.
    max_magnitude : float | None, optional
        Sinusoidal amplitude (required when ``profile_type="sin"``).
    points_per_cycle : int | None, optional
        Preferred equal-increment sampling control. Counts both endpoints of
        one cycle and guarantees baseline rows at the start, half-cycle, and
        end. Adjacent cycles share their boundary row.
    step_angle : float | None, optional
        Legacy angular sample spacing in radians. Mutually exclusive with
        ``points_per_cycle``.
    num_cycles : float | None, optional
        Number of cycles for ``"sin"``/``"pulse"`` profiles.
    phase : float, optional
        Initial phase shift in radians for the sinusoidal profile.
    dc_offset : float, optional
        Constant offset added to the sinusoidal profile.
    amplitude : float | None, optional
        Pulse amplitude above baseline (required for ``"pulse"``).
    width : float | None, optional
        Flat-top duration for each pulse (required for ``"pulse"``).
    period : float | None, optional
        Pulse period in profile time units (required for ``"pulse"``).
    slope : float | None, optional
        Ramp duration for pulse rise/fall (required for ``"pulse"``).
    step_size : float, optional
        Time increment used to sample pulse profiles.
    baseline : float, optional
        Baseline field level for pulse profiles.
    func : Callable[[float], float] | None, optional
        Magnitude function ``f(t)`` (required when ``profile_type="func"``).
    t_end : float | None, optional
        Final sample time for function-based profiles.
    dt : float | None, optional
        Sampling interval for function-based profiles.

    Returns
    -------
    Path
        Path to the written ``eregime.in`` file.

    Examples
    --------
    ```python
    # Example
    gen_eregime(...)
    ```
    """
    kind = str(profile_type).strip().lower()
    if kind == "sin":
        if max_magnitude is None or num_cycles is None:
            raise ValueError("sin profile requires max_magnitude and num_cycles.")
        if points_per_cycle is None and step_angle is None:
            raise ValueError("sin profile requires points_per_cycle or legacy step_angle.")
        if points_per_cycle is not None and step_angle is not None:
            raise ValueError("Use either points_per_cycle or step_angle, not both.")
        return _write_eregime_sinusoidal(
            out_path,
            max_magnitude=max_magnitude,
            points_per_cycle=points_per_cycle,
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
