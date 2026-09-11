from __future__ import annotations

import pandas as pd

from reaxkit.presentation.stress_strain import z_binned_strain_plots as plot_module
from reaxkit.presentation.stress_strain import (
    generate_deformation_gradient_plots,
    generate_top_bottom_plots,
)


def _table() -> pd.DataFrame:
    rows = []
    for frame in (0, 1):
        for bin_number in (1, 2):
            value = 0.01 * (frame + bin_number)
            rows.append({
                "frame": frame,
                "iter": frame * 100,
                "bin_number": bin_number,
                "bin_mean_z": float(bin_number),
                "span_change_x": value,
                "span_change_y": -2.0 * value,
                "span_change_z": 3.0 * value,
                "strain_xx": value,
                "strain_yy": -2.0 * value,
                "strain_zz": 3.0 * value,
                "gamma_xy": value,
                "gamma_xz": -2.0 * value,
                "gamma_yz": 3.0 * value,
            })
    return pd.DataFrame(rows)


def test_top_bottom_generates_complete_frame_and_bin_families(tmp_path) -> None:
    outputs = generate_top_bottom_plots(
        _table(), tmp_path, dpi=30, plot_every=2, y_scale="global"
    )

    assert len(outputs) == 6
    assert {path.parent.name for path in outputs} == {
        "span_change_vs_z_by_frame",
        "strain_vs_z_by_frame",
        "span_change_vs_frame_by_bin",
        "strain_vs_frame_by_bin",
    }
    assert all(path.is_file() for path in outputs)
    assert (tmp_path / "span_change_vs_z_by_frame" / "correlations.xlsx").is_file()
    assert (tmp_path / "span_change_vs_frame_by_bin" / "correlations.xlsx").is_file()


def test_deformation_gradient_generates_complete_frame_and_bin_families(tmp_path) -> None:
    outputs = generate_deformation_gradient_plots(
        _table(), tmp_path, dpi=30, plot_every=2, y_scale="global"
    )

    assert len(outputs) == 6
    assert {path.parent.name for path in outputs} == {
        "normal_strain_vs_z_by_frame",
        "normal_strain_vs_frame_by_bin",
        "engineering_shear_vs_z_by_frame",
        "engineering_shear_vs_frame_by_bin",
    }
    assert all(path.is_file() for path in outputs)


def test_global_mode_reuses_limits_across_frame_and_bin_views(monkeypatch, tmp_path) -> None:
    calls = []

    def capture(*_args, **kwargs):
        calls.append(kwargs)
        return []

    monkeypatch.setattr(plot_module, "_plot_family", capture)
    generate_top_bottom_plots(_table(), tmp_path, y_scale="global")

    span_calls = [call for call in calls if call["family"] == "span_change"]
    strain_calls = [call for call in calls if call["family"] == "strain"]
    assert span_calls[0]["primary_limits"] == span_calls[1]["primary_limits"]
    assert strain_calls[0]["primary_limits"] == strain_calls[1]["primary_limits"]
    assert strain_calls[0]["secondary_limits"] == strain_calls[1]["secondary_limits"]
