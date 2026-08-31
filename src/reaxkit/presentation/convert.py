"""
X-axis conversion utilities for ReaxKit plots and analyses.

This module provides helpers for converting iteration indices to alternative
x-axis representations such as simulation frames or physical time, based on
information read from a ReaxFF control file.

Typical use cases include:

- plotting observables versus simulation time instead of iteration number
- switching between iteration, frame, and time axes in workflows
- automatically choosing appropriate time units (fs, ps, ns)

**Usage context**

- Import these helpers from presentation workflows that produce tables, files, or plots.
- Reuse the public APIs here to keep output formatting and artifact behavior consistent.
"""

from pathlib import Path

import numpy as np


def _trajectory_frame_cadence(trajectory_file: str | Path) -> tuple[int, int]:
    """Return the first iteration and iteration spacing from an XYZ trajectory."""

    path = Path(trajectory_file)
    header_iterations: list[int] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        while len(header_iterations) < 2:
            count_line = handle.readline()
            if not count_line:
                break
            count_tokens = count_line.split()
            if len(count_tokens) != 1:
                continue
            try:
                atom_count = int(count_tokens[0])
            except ValueError:
                continue
            if atom_count < 0:
                continue

            header = handle.readline()
            if not header:
                break
            header_tokens = header.split()
            if len(header_tokens) < 2:
                raise ValueError(f"Malformed trajectory header in {path}: {header.rstrip()!r}")
            try:
                header_iterations.append(int(header_tokens[1]))
            except ValueError as exc:
                raise ValueError(
                    f"Could not read an iteration from trajectory header in {path}: "
                    f"{header.rstrip()!r}"
                ) from exc

            for _ in range(atom_count):
                if not handle.readline():
                    break

    if len(header_iterations) < 2:
        raise ValueError(
            f"At least two trajectory headers are needed to determine frames: {path}"
        )
    interval = header_iterations[1] - header_iterations[0]
    if interval <= 0:
        raise ValueError(
            f"Trajectory header iterations must increase; got {header_iterations[:2]} in {path}."
        )
    return header_iterations[0], interval


def _frame_axis_parameters(
    iters,
    *,
    control_file: str | Path,
    trajectory_file: str | Path | None,
    frame_count: int | None,
) -> tuple[int, float]:
    """Resolve frame origin/cadence using control, trajectory, then frame count."""

    from reaxkit.engine.reaxff.io.control_handler import ControlHandler

    control_path = Path(control_file)
    if control_path.is_file():
        handler = ControlHandler(control_path)
        iout2 = (
            handler.general_parameters.get("iout2")
            or handler.md_parameters.get("iout2")
        )
        if iout2 is None:
            raise ValueError(f"Could not find 'iout2' in control file: {control_path}")
        try:
            output_interval = int(iout2)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Control keyword 'iout2' must be a positive integer; got {iout2!r}."
            ) from exc
        if output_interval <= 0 or output_interval != iout2:
            raise ValueError(
                f"Control keyword 'iout2' must be a positive integer; got {iout2!r}."
            )
        return 0, float(output_interval)

    if trajectory_file is not None:
        trajectory_path = Path(trajectory_file)
        if not trajectory_path.is_file():
            raise FileNotFoundError(f"Trajectory frame source not found: {trajectory_path}")
        origin, interval = _trajectory_frame_cadence(trajectory_path)
        return origin, float(interval)

    if frame_count is not None:
        count = int(frame_count)
        if count < 1 or count != frame_count:
            raise ValueError(f"frame_count must be a positive integer; got {frame_count!r}.")
        iteration_values = np.asarray(iters, dtype=int)
        if iteration_values.size == 0:
            return 0, 1.0
        first = int(np.min(iteration_values))
        last = int(np.max(iteration_values))
        if count == 1:
            if first != last:
                raise ValueError("frame_count=1 is incompatible with multiple iterations.")
            return first, 1.0
        interval = (last - first) / (count - 1)
        if interval <= 0:
            raise ValueError(
                f"Could not infer a positive frame cadence from {count} frames "
                f"over iterations {first}..{last}."
            )
        return first, interval

    raise FileNotFoundError(
        "Cannot determine the frame axis: no control file or trajectory frame source "
        "was found. Pass --frame-count with the total number of trajectory frames."
    )


def convert_xaxis(
    iters,
    xaxis,
    control_file: str = "control",
    *,
    trajectory_file: str | Path | None = None,
    frame_count: int | None = None,
):
    """
    Convert iteration indices to a different x-axis representation.

    Supported target axes include iteration number, frame index, and physical
    simulation time. When converting to time, the function automatically
    selects appropriate units (fs, ps, or ns) based on the total time span.

    Parameters
    ----------
    iters : array-like
        Iteration indices to convert.
    xaxis : {'iter', 'frame', 'time'}
        Target x-axis representation.
    control_file : str, optional
        Path to the ReaxFF control file. Frame conversion uses ``iout2``;
        time conversion uses the time step.

    Returns
    -------
    tuple[numpy.ndarray, str]
        Converted x-axis values and a human-readable axis label.

    Raises
    ------
    ValueError
        If the requested x-axis is unknown or the required control keyword
        cannot be determined.

    Examples
    --------
    >>> x, label = convert_xaxis(iters, "time")
    >>> x, label = convert_xaxis(iters, "frame")
    """
    if xaxis == "iter":
        return iters, "iter"

    elif xaxis == "frame":
        origin, output_interval = _frame_axis_parameters(
            iters,
            control_file=control_file,
            trajectory_file=trajectory_file,
            frame_count=frame_count,
        )
        frames = (np.asarray(iters, dtype=float) - origin) / output_interval
        return frames, "Frame"

    elif xaxis == "time":
        from reaxkit.engine.reaxff.io.control_handler import ControlHandler

        handler = ControlHandler(control_file)
        tstep = (
            handler.general_parameters.get("tstep")
            or handler.md_parameters.get("tstep")
        )

        if tstep is None:
            raise ValueError("❌ Could not find 'tstep' in control file.")

        # Compute total time in femtoseconds
        time_fs = np.asarray(iters) * tstep

        # Automatically choose scale
        max_time = np.max(time_fs)
        if max_time >= 1e6:
            # Convert fs → ns
            time_scaled = time_fs / 1e6
            label = "Time (ns)"
        elif max_time >= 1e3:
            # Convert fs → ps
            time_scaled = time_fs / 1e3
            label = "Time (ps)"
        else:
            # Keep in fs
            time_scaled = time_fs
            label = "Time (fs)"

        return time_scaled, label

    else:
        raise ValueError(f"❌ Unknown xaxis: {xaxis}")
