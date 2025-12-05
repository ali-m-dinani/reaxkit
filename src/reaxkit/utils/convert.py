"""used to convert x-axis from iteration to time or frame, based on user's selection"""

import numpy as np

from reaxkit.io.control_handler import ControlHandler


def convert_xaxis(iters, xaxis, control_file: str = "control"):
    """
    Convert iter indices to another axis (time, frame, etc.),
    automatically scaling time units to fs, ps, or ns.

    Parameters
    ----------
    iters : array-like
        List or array of iter indices.
    xaxis : str
        Target x-axis: "iter", "frame", or "time".
    control_file : str, optional
        Path to the control file (default: "control").

    Returns
    -------
    tuple[np.ndarray, str]
        Converted x-values and corresponding x-axis label.
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
