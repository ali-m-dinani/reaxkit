import numpy as np
import pandas as pd
import pytest

from reaxkit.cli.main import _ReaxKitArgumentParser
from reaxkit.workflows.ferroelectrics.hbn_reference.domain_wall_velocity_workflow import build_parser, run_main
from reaxkit.workflows.ferroelectrics.hbn_reference.fraction_growth_workflow import COMMAND, plot_fraction_growth
from reaxkit.analysis.ferroelectrics.hbn_reference.threshold_growth import instantaneous_velocity, threshold_velocity


@pytest.mark.parametrize("initial_sign", [None, 1, -1])
def test_threshold_workflow_tables_and_plots(tmp_path, initial_sign):
    source = tmp_path / "grid.csv"
    rows = []
    for time, fraction in zip([0, 1, 2, 4, 10], [0, 0.2, 0.98, 0.99, 1]):
        for x_bin in range(2):
            for z_bin in range(2):
                rows.append(dict(time=time, u_axis="x", v_axis="z", u_bin=x_bin, v_bin=z_bin,
                                 defined_group_count=10, **{"switched fraction": fraction,
                                 "u_min (angstrom)": x_bin * 10, "u_max (angstrom)": (x_bin + 1) * 10,
                                 "u_center (angstrom)": x_bin * 10 + 5,
                                 "v_min (angstrom)": z_bin * 5 - 10, "v_max (angstrom)": z_bin * 5 - 5,
                                 "v_center (angstrom)": z_bin * 5 - 7.5}))
    table = pd.DataFrame(rows)
    if initial_sign is not None:
        count_column = "positive_count" if initial_sign == 1 else "negative_count"
        table[count_column] = (1 - table.pop("switched fraction")) * table["defined_group_count"]
    table.to_csv(source, index=False)
    parser = build_parser(_ReaxKitArgumentParser(COMMAND), command=COMMAND)
    help_text = parser.format_help(full=True)
    assert "--threshold" in help_text
    assert "--time-min" not in help_text and "--time-max" not in help_text
    arguments = ["--input", str(source), "--threshold", "0.98", "--field-start-time", "0.5",
                              "--length-x", "100", "--length-z", "50", "--figure-dpi", "60",
                              "--output-dir", str(tmp_path / "output")]
    assert parser.parse_args(arguments).initial_sign == 1
    if initial_sign == -1:
        arguments += ["--initial-sign", "-1"]
    args = parser.parse_args(arguments)
    assert run_main(COMMAND, args) == 0
    output = tmp_path / "output"
    readme = (output / "README.md").read_text(encoding="utf-8")
    assert "speed_heterogeneity_index" in readme
    assert "sigma / (mu + sigma)" in readme
    assert "0.333" in readme
    assert "do not prove" in readme
    for axis in "xz":
        matrix = pd.read_csv(output / f"{axis}_bins_fraction_vs_time.csv")
        assert matrix.shape == (5, 3)
        assert matrix.columns[0] == "time"
        assert matrix.columns[1].startswith(f"{axis}_bin_0 ({axis}=")
        assert "Å" in matrix.columns[2]
    endpoints = pd.read_csv(output / "velocities.csv")
    assert len(endpoints) == 4
    z_bins = endpoints[endpoints.bin_axis == "z"].sort_values("bin")
    assert z_bins.bin_position.tolist() == ["low-z (bottom)", "high-z (top)"]
    assert z_bins.bin_center_angstrom.tolist() == [-7.5, -2.5]
    assert endpoints.threshold_time.eq(4).all()
    np.testing.assert_allclose(endpoints.field_fraction, 0.1)
    assert len(list((output / "plots" / "fraction").glob("*_fraction.png"))) == 4
    assert len(list((output / "plots" / "velocity").glob("*_velocity.png"))) == 4
    assert {path.name for path in (output / "plots").glob("*.png")} == {
        "x_all_bins_fraction.png", "z_all_bins_fraction.png",
    }
    intervals = pd.read_csv(output / "velocity_intervals.csv")
    assert intervals.time_end.max() == 10
    assert "bin_center_angstrom" in intervals
    summary = pd.read_csv(output / "velocity_summary.csv").set_index("velocity_axis")
    assert summary.index.tolist() == ["x", "z"]
    assert summary.valid_bin_count.eq(2).all()
    assert summary.excluded_bin_count.eq(0).all()
    np.testing.assert_allclose(summary.std_velocity_m_per_s, 0)
    np.testing.assert_allclose(summary.velocity_cv, 0)
    np.testing.assert_allclose(summary.speed_heterogeneity_index, 0)
    for direction in "xz":
        values = endpoints.loc[endpoints.velocity_axis == direction, "velocity_m_per_s"]
        for statistic in ("mean", "median", "min", "max"):
            assert summary.loc[direction, f"{statistic}_velocity_m_per_s"] == pytest.approx(getattr(values, statistic)())


def test_velocity_plot_has_dots_and_constant_average(tmp_path, monkeypatch):
    from matplotlib.figure import Figure

    samples = pd.DataFrame({"bin": [0] * 5, "time": [0, 1, 2, 4, 10],
                            "switched_fraction": [0, 0.5, 1, 1, 1]})
    endpoints = threshold_velocity(samples, threshold=0.98, field_start_time=0, length=20)
    intervals = instantaneous_velocity(samples, endpoints)
    observed = []

    def capture(figure, path, **kwargs):
        if str(path).endswith("_velocity.png"):
            observed.extend(figure.axes[0].lines)
            assert not any(line.get_visible() for line in figure.axes[0].get_xgridlines())

    monkeypatch.setattr(Figure, "savefig", capture)
    plot_fraction_growth(samples, endpoints, intervals, tmp_path, bin_axis="x", velocity_axis="z", dpi=60)
    assert observed[0].get_linestyle() == "None"
    assert observed[0].get_marker() == "o"
    assert observed[0].get_color() == "#0072B2"
    assert observed[1].get_color() == "#D62728"
    np.testing.assert_allclose(observed[1].get_ydata(), endpoints.iloc[0].velocity_m_per_s)
    assert observed[0].get_ydata()[-1] == 0


def test_all_bins_plot_uses_ordered_gradient(tmp_path, monkeypatch):
    from matplotlib.figure import Figure
    from matplotlib.colors import to_hex

    samples = pd.DataFrame({"bin": [2, 2, 0, 0, 1, 1], "time": [0, 1] * 3,
                            "switched_fraction": [0.2, 1, 0, 1, 0.1, 1]})
    endpoints = threshold_velocity(samples, threshold=0.98, field_start_time=0, length=20)
    intervals = instantaneous_velocity(samples, endpoints)
    captured = []

    def capture(figure, path, **kwargs):
        if path.name == "x_all_bins_fraction.png":
            curves = figure.axes[0].lines[:3]
            np.testing.assert_allclose([curve.get_ydata()[0] for curve in curves], [0, 0.1, 0.2])
            assert to_hex(curves[0].get_color()) == "#0000ff"
            assert to_hex(curves[-1].get_color()) == "#ff0000"
            assert len({to_hex(curve.get_color()) for curve in curves}) == 3
            assert figure.axes[1].get_ylabel() == "x-bin number"
            assert not any(line.get_visible() for line in figure.axes[0].get_xgridlines())
            captured.append(path)

    monkeypatch.setattr(Figure, "savefig", capture)
    written = plot_fraction_growth(samples, endpoints, intervals, tmp_path, bin_axis="x", velocity_axis="z", dpi=60)
    assert captured == [tmp_path / "x_all_bins_fraction.png"]
    assert len(written) == 7


@pytest.mark.parametrize("threshold, valid_count", [(0.98, 2), (1.0, 0)])
def test_velocity_summary_excludes_unfinished_bins(tmp_path, monkeypatch, threshold, valid_count):
    from reaxkit.workflows.ferroelectrics.hbn_reference import fraction_growth_workflow

    monkeypatch.setattr(fraction_growth_workflow, "plot_fraction_growth", lambda *args, **kwargs: [])
    source = tmp_path / "x.csv"
    pd.DataFrame({"x_bin": [0, 1, 2], "0": [0, 0, 0], "1": [1, 0.5, 0.2],
                  "2": [1, 1, 0.3]}).to_csv(source, index=False)
    parser = build_parser(_ReaxKitArgumentParser(COMMAND), command=COMMAND)
    output = tmp_path / "output"
    args = parser.parse_args(["--input", str(source), "--axis", "z", "--length", "10",
                              "--threshold", str(threshold), "--field-start-time", "0",
                              "--output-dir", str(output)])
    assert run_main(COMMAND, args) == 0
    summary = pd.read_csv(output / "velocity_summary.csv").iloc[0]
    assert summary.velocity_axis == "z"
    assert summary.total_bin_count == 3
    assert summary.valid_bin_count == valid_count
    assert summary.excluded_bin_count == 3 - valid_count
    if valid_count:
        assert summary.mean_velocity_m_per_s == pytest.approx(750)
        assert summary.median_velocity_m_per_s == pytest.approx(750)
        assert summary.min_velocity_m_per_s == pytest.approx(500)
        assert summary.max_velocity_m_per_s == pytest.approx(1000)
        assert summary.std_velocity_m_per_s == pytest.approx(250)
        assert summary.velocity_cv == pytest.approx(1 / 3)
        assert summary.speed_heterogeneity_index == pytest.approx(0.25)
    else:
        assert pd.isna(summary.mean_velocity_m_per_s)
        assert pd.isna(summary.median_velocity_m_per_s)
        assert pd.isna(summary.min_velocity_m_per_s)
        assert pd.isna(summary.max_velocity_m_per_s)
        assert pd.isna(summary.std_velocity_m_per_s)
        assert pd.isna(summary.velocity_cv)
        assert pd.isna(summary.speed_heterogeneity_index)


def test_single_matrix_cannot_generate_other_axis(tmp_path):
    source = tmp_path / "x.csv"
    pd.DataFrame({"x_bin/time(ps)": [0], "0": [0], "1": [1]}).to_csv(source, index=False)
    parser = build_parser(_ReaxKitArgumentParser(COMMAND), command=COMMAND)
    args = parser.parse_args(["--input", str(source), "--threshold", "0.98", "--field-start-time", "0",
                              "--length-x", "100", "--length-z", "50"])
    with pytest.raises(ValueError, match="z-bin"):
        run_main(COMMAND, args)


def test_lag_workflow_exports_rise_velocity_and_settings(tmp_path):
    import json

    source = tmp_path / "x.csv"
    pd.DataFrame({"x_bin": [0], "0": [0.1], "1": [0.15], "2": [0.2],
                  "4": [0.7], "5": [1]}).to_csv(source, index=False)
    output = tmp_path / "output"
    parser = build_parser(_ReaxKitArgumentParser(COMMAND), command=COMMAND)
    args = parser.parse_args(["--input", str(source), "--axis", "z", "--length", "50",
                              "--threshold", "0.6", "--lag-threshold", "0.05",
                              "--field-start-time", "0", "--figure-dpi", "60",
                              "--output-dir", str(output)])
    assert run_main(COMMAND, args) == 0
    row = pd.read_csv(output / "velocities.csv").iloc[0]
    assert row.lag_time == 2
    assert row.rise_time == 2
    assert row.velocity_m_per_s == pytest.approx(1250)
    summary = pd.read_csv(output / "velocity_summary.csv").iloc[0]
    assert summary.mean_velocity_m_per_s == pytest.approx(1250)
    settings = json.loads((output / "settings.json").read_text())
    assert settings["method"] == "lag_threshold_fraction_growth"
    assert settings["lag_threshold"] == 0.05
    assert "rise_start_fraction" in (output / "README.md").read_text()
    assert (output / "plots" / "fraction" / "x_bin_0_fraction.png").exists()
