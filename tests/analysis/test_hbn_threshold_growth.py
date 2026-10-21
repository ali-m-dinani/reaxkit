import pandas as pd
import pytest

from reaxkit.analysis.ferroelectrics.hbn_reference.threshold_growth import (
    instantaneous_velocity, read_fraction_tables, threshold_velocity,
)


def test_threshold_uses_actual_endpoint_not_late_plateau():
    samples = pd.DataFrame({"bin": [0] * 5, "time": [0, 1, 2, 4, 10],
                            "switched_fraction": [0, 0.2, 0.98, 0.99, 1]})
    result = threshold_velocity(samples, threshold=0.98, field_start_time=0.5, length=50)
    row = result.iloc[0]
    assert row.field_fraction == pytest.approx(0.1)
    assert row.baseline_interpolated
    assert row.threshold_time == 4
    assert row.threshold_fraction == 0.99
    assert row.velocity_m_per_s == pytest.approx(50 * 0.89 / 3.5 * 100)


def test_lag_velocity_uses_rise_distance_and_time():
    samples = pd.DataFrame({"bin": [0] * 7, "time": [0, 1, 2, 3, 4, 5, 6],
                            "switched_fraction": [0.1, 0.3, 0.25, 0.4, 0.6, 0.8, 1]})
    row = threshold_velocity(samples, threshold=0.6, field_start_time=0.5,
                             length=50, front_count=2, lag_threshold=0.1).iloc[0]
    assert row.field_fraction == pytest.approx(0.2)
    assert row.rise_start_time == 3
    assert row.lag_time == 2.5
    assert row.rise_time == 2
    assert row.elapsed_time == 4.5
    assert row.fraction_change == pytest.approx(0.4)
    assert row.velocity_m_per_s == pytest.approx(500)


@pytest.mark.parametrize("fractions, threshold, status, lag", [
    ([0, 0.05, 0.05], 0.6, "lag_not_reached", None),
    ([0, 0.1, 0.2], 0.6, "threshold_not_reached", 1),
    ([0, 0.8, 1], 0.6, "unresolved_rise", 1),
    ([0.55, 0.7, 1], 0.6, "lag_target_at_or_above_threshold", None),
    ([0.8, 0.9, 1], 0.6, "already_above_threshold", None),
])
def test_lag_unavailable_cases(fractions, threshold, status, lag):
    samples = pd.DataFrame({"bin": [0] * 3, "time": [0, 1, 2], "switched_fraction": fractions})
    row = threshold_velocity(samples, threshold=threshold, field_start_time=0,
                             length=50, lag_threshold=0.05).iloc[0]
    assert row.status == status
    assert pd.isna(row.velocity_m_per_s)
    assert pd.isna(row.lag_time) if lag is None else row.lag_time == lag


@pytest.mark.parametrize("lag", [0, -0.1, 1, float("nan"), float("inf")])
def test_invalid_lag_threshold(lag):
    samples = pd.DataFrame({"bin": [0, 0], "time": [0, 1], "switched_fraction": [0, 1]})
    with pytest.raises(ValueError, match="lag_threshold"):
        threshold_velocity(samples, threshold=0.6, field_start_time=0, length=50, lag_threshold=lag)


def test_threshold_reports_unavailable_measurements():
    samples = pd.DataFrame({"bin": [0, 0, 1, 1], "time": [0, 1, 0, 1],
                            "switched_fraction": [0, 0.5, 1, 1]})
    result = threshold_velocity(samples, threshold=0.98, field_start_time=0, length=50)
    assert result.status.tolist() == ["threshold_not_reached", "already_above_threshold"]
    assert result.velocity_m_per_s.isna().all()
    outside = threshold_velocity(samples, threshold=0.98, field_start_time=-1, length=50)
    assert outside.status.eq("onset_outside_data").all()


def test_threshold_respects_each_bins_endpoint_and_front_count():
    samples = pd.DataFrame({"bin": [0, 0, 0, 1, 1, 1], "time": [0, 1, 2, 0, 1, 2],
                            "switched_fraction": [0, 1, 1, 0.2, 0.5, 1]})
    result = threshold_velocity(samples, threshold=0.98, field_start_time=0, length=20, front_count=2)
    assert result.threshold_time.tolist() == [1, 2]
    assert result.velocity_m_per_s.tolist() == pytest.approx([1000, 400])


def test_instantaneous_rates_keep_plateau_and_negative_changes():
    samples = pd.DataFrame({"bin": [0] * 6, "time": [0, 1, 2, 4, 6, 10],
                            "switched_fraction": [0, 0.2, 1, 1, 0.8, 0.8]})
    endpoints = threshold_velocity(samples, threshold=0.98, field_start_time=0.5, length=20)
    rates = instantaneous_velocity(samples, endpoints)
    assert rates.time_start.iloc[0] == 0.5
    assert rates.velocity_m_per_s.tolist() == pytest.approx([400, 1600, 0, -200, 0])
    assert rates.average_velocity_m_per_s.nunique() == 1


def test_matrix_numeric_time_columns(tmp_path):
    source = tmp_path / "matrix.csv"
    pd.DataFrame({"x_bin/time(ps)": [0, 1], "0": [0, 0.1], "2": [1, 0.9]}).to_csv(source, index=False)
    tables = read_fraction_tables(source)
    assert set(tables) == {"x"}
    assert tables["x"].time.tolist() == [0, 2, 0, 2]
    assert tables["x"].switched_fraction.tolist() == [0, 1, 0.1, 0.9]


def test_user_grid_fraction_is_weighted_not_recomputed(tmp_path):
    source = tmp_path / "grid.csv"
    pd.DataFrame(dict(time=[0, 0], u_axis=["x", "x"], v_axis=["z", "z"], u_bin=[0, 0],
                      v_bin=[0, 1], switched_fraction=[0.2, 0.8], defined_group_count=[1, 3],
                      negative_count=[0, 0])).to_csv(source, index=False)
    tables = read_fraction_tables(source)
    assert tables["x"].iloc[0].switched_fraction == pytest.approx(0.65)
    assert tables["z"].switched_fraction.tolist() == pytest.approx([0.2, 0.8])


@pytest.mark.parametrize("initial_sign, expected", [(1, 0.7), (-1, 0.6)])
@pytest.mark.parametrize("denominator", ["defined_group_count", "defined_cell_count"])
def test_missing_fraction_uses_initial_sign_and_population_weights(tmp_path, initial_sign, expected, denominator):
    source = tmp_path / "counts.csv"
    pd.DataFrame(dict(time=[0, 0, 0], u_axis=["x"] * 3, v_axis=["z"] * 3,
                      u_bin=[0] * 3, v_bin=[0, 1, 2], positive_count=[2, 1, 0],
                      negative_count=[1, 3, 0], zero_count=[1, 2, 0],
                      **{denominator: [4, 6, 0]})).to_csv(source, index=False)
    with pytest.raises(ValueError, match="no defined groups"):
        read_fraction_tables(source, initial_sign=initial_sign)
    table = pd.read_csv(source).iloc[:2]
    table.to_csv(source, index=False)
    tables = read_fraction_tables(source, initial_sign=initial_sign)
    assert tables["x"].iloc[0].switched_fraction == pytest.approx(expected)
    if initial_sign == 1:
        assert read_fraction_tables(source)["x"].iloc[0].switched_fraction == pytest.approx(expected)
    table["switched_fraction"] = 0.25
    table.to_csv(source, index=False)
    assert read_fraction_tables(source, initial_sign=initial_sign)["x"].iloc[0].switched_fraction == 0.25


@pytest.mark.parametrize("count", [-1, 11, float("nan")])
def test_invalid_fallback_counts_are_rejected(tmp_path, count):
    source = tmp_path / "counts.csv"
    pd.DataFrame(dict(time=[0], u_axis=["x"], v_axis=["z"], u_bin=[0], v_bin=[0],
                      positive_count=[count], defined_group_count=[10])).to_csv(source, index=False)
    with pytest.raises(ValueError, match="count"):
        read_fraction_tables(source)
