from __future__ import annotations

import numpy as np
import pytest

from reaxkit.presentation.convert import convert_xaxis


def _write_control(path, iout2: str = "5"):
    path.write_text(f"# MD\n{iout2} iout2\n", encoding="utf-8")
    return path


def test_frame_axis_uses_iout2_and_starts_at_zero(tmp_path):
    control = _write_control(tmp_path / "control", "5")

    values, label = convert_xaxis([0, 1, 4, 5, 9, 10], "frame", control_file=control)

    np.testing.assert_allclose(values, [0, 0.2, 0.8, 1, 1.8, 2])
    assert label == "Frame"


def test_frame_axis_requires_a_frame_source_or_count(tmp_path):
    missing = tmp_path / "control"

    with pytest.raises(FileNotFoundError, match="Pass --frame-count"):
        convert_xaxis([0, 5], "frame", control_file=missing)


def test_frame_axis_falls_back_to_trajectory_headers(tmp_path):
    trajectory = tmp_path / "xmolout"
    trajectory.write_text(
        "2\nslab 20 -1 1 1 1 90 90 90\nAl 0 0 0\nN 0 0 1\n"
        "2\nslab 30 -2 1 1 1 90 90 90\nAl 0 0 0\nN 0 0 1\n",
        encoding="utf-8",
    )

    values, label = convert_xaxis(
        [20, 25, 30],
        "frame",
        control_file=tmp_path / "missing-control",
        trajectory_file=trajectory,
    )

    np.testing.assert_allclose(values, [0, 0.5, 1])
    assert label == "Frame"


def test_frame_axis_falls_back_to_explicit_frame_count(tmp_path):
    values, label = convert_xaxis(
        [0, 5, 10, 15, 20],
        "frame",
        control_file=tmp_path / "missing-control",
        frame_count=3,
    )

    np.testing.assert_allclose(values, [0, 0.5, 1, 1.5, 2])
    assert label == "Frame"


@pytest.mark.parametrize("iout2", ["0", "-2", "2.5"])
def test_frame_axis_requires_positive_integer_iout2(tmp_path, iout2):
    control = _write_control(tmp_path / "control", iout2)

    with pytest.raises(ValueError, match="positive integer"):
        convert_xaxis([0, 5], "frame", control_file=control)
