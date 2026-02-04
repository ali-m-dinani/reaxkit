"""
X-axis conversion utilities for ReaxKit plots and analyses.

This module provides helpers for converting iteration indices to alternative
x-axis representations such as simulation frames or physical time, based on
information read from a ReaxFF control file.

Typical use cases include:

- plotting observables versus simulation time instead of iteration number
- switching between iteration, frame, and time axes in workflows
- automatically choosing appropriate time units (fs, ps, ns)
"""

import numpy as np

from reaxkit.io.handlers.control_handler import ControlHandler


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
        Path to the ReaxFF control file used to determine the time step.

    Returns
    -------
    tuple[numpy.ndarray, str]
        Converted x-axis values and a human-readable axis label.

    Raises
    ------
    ValueError
        If the requested x-axis is unknown or the time step cannot be
        determined from the control file.

    Examples
    --------
    >>> x, label = convert_xaxis(iters, "time")
    >>> x, label = convert_xaxis(iters, "frame")
    """
    if xaxis == "iter":
        return iters, "iter"

    elif xaxis == "frame":
        return np.arange(len(iters)), "Frame"

    elif xaxis == "time":
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
