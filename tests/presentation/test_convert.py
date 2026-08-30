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

    np.testing.assert_array_equal(values, [0, 0, 0, 1, 1, 2])
    assert label == "Frame"


def test_frame_axis_requires_control_file(tmp_path):
    missing = tmp_path / "control"

    with pytest.raises(FileNotFoundError, match="required for --xaxis frame"):
        convert_xaxis([0, 5], "frame", control_file=missing)


@pytest.mark.parametrize("iout2", ["0", "-2", "2.5"])
def test_frame_axis_requires_positive_integer_iout2(tmp_path, iout2):
    control = _write_control(tmp_path / "control", iout2)

    with pytest.raises(ValueError, match="positive integer"):
        convert_xaxis([0, 5], "frame", control_file=control)
