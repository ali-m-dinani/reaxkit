from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from reaxkit.analysis.ferroelectrics.hbn_reference.domain_wall_velocity import (
    PREFIX, VelocityRequest, calculate_domain_wall_velocity,
)


def write_bins(folder, arrivals=(1.5, 2.5, 3.5), reverse=False):
    rows = []
    for frame in range(7):
        for spatial_bin, arrival in enumerate(arrivals):
            positive = int(np.clip(50 + 50 * (arrival - frame), 0, 100))
            if reverse:
                positive = 100 - positive
            rows.append(dict(frame_index=frame, iter=frame * 100, time=float(frame),
                             u_axis="x", v_axis="z", u_bin=0, v_bin=spatial_bin,
                             positive_count=positive, negative_count=100 - positive,
                             zero_count=0, defined_group_count=100,
                             **{"u_center (angstrom)": 0.0, "v_center (angstrom)": spatial_bin * 10.0}))
    table = pd.DataFrame(rows)
    table.to_csv(folder / f"{PREFIX}2d.csv", index=False)
    return table


def write_centers(folder, sign_frames):
    rows = []
    for frame, signs in enumerate(sign_frames):
        for group_id, sign in enumerate(signs):
            rows.append(dict(frame_index=frame, iter=frame * 100, time=float(frame),
                             local_group_id=group_id, polarity=sign,
                             **{"reference_center_x (angstrom)": float(group_id),
                                "reference_center_y (angstrom)": 0.0,
                                "reference_center_z (angstrom)": 0.0}))
    table = pd.DataFrame(rows)
    table.to_csv(folder / f"{PREFIX}centers.csv", index=False)
    return table


@pytest.mark.parametrize("axis", ["x", "z"])
@pytest.mark.parametrize("initial_sign", [1, -1])
def test_crossing_recovers_known_directional_velocity(tmp_path, axis, initial_sign):
    table = write_bins(tmp_path, reverse=initial_sign == -1)
    if axis == "x":
        table["u_axis"], table["v_axis"] = "z", "x"
        table.to_csv(tmp_path / f"{PREFIX}2d.csv", index=False)
    result = calculate_domain_wall_velocity(tmp_path, VelocityRequest("crossing", axis=axis, initial_sign=initial_sign))
    np.testing.assert_allclose(result.samples["time"], [1.5, 2.5, 3.5])
    assert result.velocities.iloc[0]["velocity_m_per_s"] == pytest.approx(1000)
    assert result.velocities.iloc[0]["status"] == "ok"


def test_crossing_reverse_motion_and_onset(tmp_path):
    write_bins(tmp_path, arrivals=(3.5, 2.5, 1.5))
    result = calculate_domain_wall_velocity(tmp_path, VelocityRequest("crossing", field_start_time=0.5))
    assert result.velocities.iloc[0]["velocity_m_per_s"] == pytest.approx(-1000)
    np.testing.assert_allclose(result.samples["time"], [3, 2, 1])


def test_crossing_flags_nonmonotonic_arrivals(tmp_path):
    write_bins(tmp_path, arrivals=(1.5, 3.5, 2.5))
    result = calculate_domain_wall_velocity(tmp_path, VelocityRequest("crossing"))
    assert result.velocities.iloc[0]["status"] == "nonmonotonic_arrivals"
    assert np.isnan(result.velocities.iloc[0]["velocity_m_per_s"])


def test_fraction_is_count_weighted_and_front_corrected(tmp_path):
    table = write_bins(tmp_path)
    table = table[table["v_bin"] < 2].copy()
    table["defined_group_count"] = np.where(table["v_bin"] == 0, 100, 300)
    table["negative_count"] = table["time"] * np.where(table["v_bin"] == 0, 10, 0)
    table["zero_count"] = 10
    table["positive_count"] = table["defined_group_count"] - table["negative_count"] - 10
    table.to_csv(tmp_path / f"{PREFIX}2d.csv", index=False)
    request = VelocityRequest("fraction-growth", length=100, front_count=2)
    result = calculate_domain_wall_velocity(tmp_path, request)
    np.testing.assert_allclose(result.samples["switched_fraction"], np.arange(7) * 0.025)
    assert result.velocities.iloc[0]["velocity_m_per_s"] == pytest.approx(125)


def test_time_join_and_iteration_conversion(tmp_path):
    table = write_bins(tmp_path)
    table.drop(columns="time").to_csv(tmp_path / f"{PREFIX}2d.csv", index=False)
    request = VelocityRequest("crossing")
    with pytest.raises(ValueError, match="No time column"):
        calculate_domain_wall_velocity(tmp_path, request)
    converted = calculate_domain_wall_velocity(tmp_path, replace(request, time_per_iteration=0.01))
    assert converted.velocities.iloc[0]["velocity_m_per_s"] == pytest.approx(1000)
    table[["frame_index", "iter", "time"]].drop_duplicates().to_csv(
        tmp_path / f"{PREFIX}whole_slab_summary.csv", index=False,
    )
    joined = calculate_domain_wall_velocity(tmp_path, request)
    pd.testing.assert_frame_equal(converted.samples, joined.samples)
    explicit_fallback = calculate_domain_wall_velocity(tmp_path, replace(request, time_per_iteration=99))
    pd.testing.assert_frame_equal(joined.samples, explicit_fallback.samples)


def test_tracking_recovers_wall_motion(tmp_path):
    write_centers(tmp_path, [[-1] * boundary + [1] * (8 - boundary) for boundary in [2, 3, 4, 5]])
    result = calculate_domain_wall_velocity(tmp_path, VelocityRequest("tracking", axis="x", max_step=1.1))
    assert result.samples["track_id"].nunique() == 1
    assert result.velocities.iloc[0]["velocity_m_per_s"] == pytest.approx(100)


def test_periodic_tracking_unwraps_seam(tmp_path):
    signs = []
    for shift in [0, 1, 2, 3]:
        signs.append([-1 if (site - shift) % 8 in {6, 7} else 1 for site in range(8)])
    write_centers(tmp_path, signs)
    request = VelocityRequest("tracking", axis="x", max_step=1.1, periodic_length=8)
    result = calculate_domain_wall_velocity(tmp_path, request)
    assert len(result.velocities) == 2
    np.testing.assert_allclose(result.velocities["velocity_m_per_s"], 100)


def test_tracking_ambiguity_does_not_join_walls(tmp_path):
    write_centers(tmp_path, [[-1, 1, -1, 1, -1, 1]] * 3)
    result = calculate_domain_wall_velocity(tmp_path, VelocityRequest("tracking", axis="x", max_step=5))
    assert result.velocities["status"].eq("insufficient_points").all()
    assert "ambiguous_match_new_track" in result.diagnostics["status"].values


def test_no_boundaries_has_readable_empty_result(tmp_path):
    write_centers(tmp_path, [[1, 1, 1], [1, 1, 1]])
    result = calculate_domain_wall_velocity(tmp_path, VelocityRequest("tracking", axis="x", max_step=1))
    assert result.velocities.empty
    assert "status" in result.velocities


def test_invalid_and_incomplete_inputs(tmp_path):
    table = write_bins(tmp_path)
    with pytest.raises(ValueError, match="length"):
        calculate_domain_wall_velocity(tmp_path, VelocityRequest("fraction-growth"))
    with pytest.raises(ValueError, match="max-step"):
        calculate_domain_wall_velocity(tmp_path, VelocityRequest("tracking"))
    table.iloc[1:].to_csv(tmp_path / f"{PREFIX}2d.csv", index=False)
    with pytest.raises(ValueError, match="complete set"):
        calculate_domain_wall_velocity(tmp_path, VelocityRequest("crossing"))


def test_persistence_rejects_transient_crossing(tmp_path):
    table = write_bins(tmp_path, arrivals=(1.5,))
    table["positive_count"] = [100, 0, 100, 100, 0, 0, 0]
    table["negative_count"] = 100 - table["positive_count"]
    table.to_csv(tmp_path / f"{PREFIX}2d.csv", index=False)
    result = calculate_domain_wall_velocity(tmp_path, VelocityRequest("crossing", persistence=2))
    assert result.samples.iloc[0]["time"] == pytest.approx(3.5)


def test_multiple_persistent_crossings_are_excluded(tmp_path):
    table = write_bins(tmp_path, arrivals=(1.5,))
    table["positive_count"] = [100, 0, 0, 100, 0, 0, 0]
    table["negative_count"] = 100 - table["positive_count"]
    table.to_csv(tmp_path / f"{PREFIX}2d.csv", index=False)
    result = calculate_domain_wall_velocity(tmp_path, VelocityRequest("crossing", persistence=2))
    assert result.samples.empty
    assert result.diagnostics.iloc[0]["status"] == "multiple_crossings"


@pytest.mark.parametrize("id_column", ["local_group_id", "local_neighborhood_id"])
def test_tracking_does_not_bridge_missing_sites(tmp_path, id_column):
    table = write_centers(tmp_path, [[-1] * boundary + [1] * (8 - boundary) for boundary in [2, 3, 4, 5]])
    table = table[~((table["frame_index"] == 1) & (table["local_group_id"] == 0))]
    table = table.rename(columns={"local_group_id": id_column})
    table.to_csv(tmp_path / f"{PREFIX}centers.csv", index=False)
    result = calculate_domain_wall_velocity(tmp_path, VelocityRequest("tracking", axis="x", max_step=2.1))
    assert result.samples["track_id"].nunique() == 2
    assert "missing_or_duplicate_sites" in result.diagnostics["status"].values


def test_empty_bin_blocks_crossing_interpolation(tmp_path):
    table = write_bins(tmp_path, arrivals=(1.5,))
    columns = ["positive_count", "negative_count", "zero_count", "defined_group_count"]
    table.loc[table["frame_index"] == 1, columns] = 0
    table.to_csv(tmp_path / f"{PREFIX}2d.csv", index=False)
    result = calculate_domain_wall_velocity(tmp_path, VelocityRequest("crossing"))
    assert result.samples.empty


def test_periodic_gate_cannot_exceed_half_box(tmp_path):
    with pytest.raises(ValueError, match="half"):
        calculate_domain_wall_velocity(tmp_path, VelocityRequest("tracking", max_step=4, periodic_length=8))


def test_time_unit_conversion(tmp_path):
    write_bins(tmp_path)
    result = calculate_domain_wall_velocity(tmp_path, VelocityRequest("crossing", time_unit="ns"))
    assert result.velocities.iloc[0]["velocity_m_per_s"] == pytest.approx(1.0)


def test_fraction_intervals_resolve_changing_rates_and_bins(tmp_path):
    table = write_bins(tmp_path, arrivals=(1.5, 2.5))
    fractions = np.array([0, 0.1, 0.3, 0.3, 0.5, 0.4, 0.6])
    table["negative_count"] = [int(round(fractions[int(frame)] * 100)) if spatial_bin == 0 else 0
                               for frame, spatial_bin in zip(table["frame_index"], table["v_bin"])]
    table["positive_count"] = 100 - table["negative_count"]
    table.to_csv(tmp_path / f"{PREFIX}2d.csv", index=False)
    result = calculate_domain_wall_velocity(tmp_path, VelocityRequest("fraction-growth", length=100, front_count=2))
    local = result.intervals.query("scope == 'bin' and bin == 0")
    np.testing.assert_allclose(local["velocity_m_per_s"], np.diff(fractions) * 2500)
    np.testing.assert_allclose(local["time_mid"], np.arange(6) + 0.5)
    np.testing.assert_allclose(local["frame_start"], np.arange(6))
    np.testing.assert_allclose(local["frame_end"], np.arange(1, 7))
    assert local["position_angstrom"].eq(0).all()
    assert result.intervals.query("scope == 'bin' and bin == 1")["velocity_m_per_s"].eq(0).all()
    np.testing.assert_allclose(local["speed_m_per_s"], local["velocity_m_per_s"].abs())
    slice_rates = result.intervals.query("scope == 'slice'")
    np.testing.assert_allclose(slice_rates["velocity_m_per_s"], np.diff(fractions) * 2500)


def test_fraction_intervals_flag_population_changes(tmp_path):
    table = write_bins(tmp_path, arrivals=(1.5,))
    table.loc[table["frame_index"] == 1, "positive_count"] += 1
    table.loc[table["frame_index"] == 1, "defined_group_count"] += 1
    table.to_csv(tmp_path / f"{PREFIX}2d.csv", index=False)
    result = calculate_domain_wall_velocity(tmp_path, VelocityRequest("fraction-growth", length=10))
    local = result.intervals.query("scope == 'bin'")
    assert local.iloc[:2]["status"].eq("changing_defined_population").all()
    assert local.iloc[:2]["velocity_m_per_s"].isna().all()
    assert local.iloc[2:]["status"].eq("ok").all()


@pytest.mark.parametrize("arrivals,expected", [((1.5, 2.5, 4.5), [1000, 500]), ((4.5, 2.5, 1.5), [-500, -1000])])
def test_crossing_intervals_capture_local_acceleration(tmp_path, arrivals, expected):
    write_bins(tmp_path, arrivals=arrivals)
    result = calculate_domain_wall_velocity(tmp_path, VelocityRequest("crossing"))
    np.testing.assert_allclose(result.intervals["velocity_m_per_s"], expected)
    np.testing.assert_allclose(result.intervals["position_mid_angstrom"], [5, 15])
    np.testing.assert_allclose(result.intervals["time_mid"], [(arrivals[0] + arrivals[1]) / 2, (arrivals[1] + arrivals[2]) / 2])
    assert result.intervals["lower_frame_before"].notna().all()
    assert (result.intervals["time_end"] > result.intervals["time_start"]).all()


def test_crossing_intervals_flag_simultaneous_and_missing_events(tmp_path):
    write_bins(tmp_path, arrivals=(1.5, 1.5, 20))
    result = calculate_domain_wall_velocity(tmp_path, VelocityRequest("crossing"))
    assert result.intervals["status"].tolist() == ["simultaneous_crossings", "missing_crossing"]
    assert result.intervals["velocity_m_per_s"].isna().all()
