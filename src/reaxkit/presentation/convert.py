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


def convert_xaxis(iters, xaxis, control_file: str = "control"):
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
        from reaxkit.engine.reaxff.io.control_handler import ControlHandler

        control_path = Path(control_file)
        if not control_path.is_file():
            raise FileNotFoundError(
                "A ReaxFF control file is required for --xaxis frame so "
                f"'iout2' can be read. Control file not found: {control_path}"
            )

        handler = ControlHandler(control_path)
        iout2 = (
            handler.general_parameters.get("iout2")
            or handler.md_parameters.get("iout2")
        )
        if iout2 is None:
            raise ValueError(
                f"Could not find 'iout2' in control file: {control_path}"
            )
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

        frames = np.floor_divide(np.asarray(iters, dtype=int), output_interval)
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
