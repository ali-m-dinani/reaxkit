"""Domain-wall velocity estimators operating exclusively on exported CSV tables."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd


PREFIX = "hbn_reference_projected_polarity_"
TIME_SECONDS = {"fs": 1e-15, "ps": 1e-12, "ns": 1e-9, "s": 1.0}


@dataclass(frozen=True)
class VelocityRequest:
    method: str
    axis: str = "z"
    time_unit: str = "ps"
    time_per_iteration: float | None = None
    field_start_time: float = 0.0
    time_min: float | None = None
    time_max: float | None = None
    initial_sign: int = 1
    length: float | None = None
    front_count: int = 1
    persistence: int = 2
    aggregate_transverse: bool = False
    max_step: float | None = None
    periodic_length: float | None = None
    coordinate_decimals: int = 4


@dataclass
class VelocityResult:
    samples: pd.DataFrame
    velocities: pd.DataFrame
    diagnostics: pd.DataFrame
    intervals: pd.DataFrame = field(default_factory=pd.DataFrame)


def _require(table, columns):
    missing = set(columns) - set(table.columns)
    if missing:
        raise ValueError(f"Missing CSV columns: {', '.join(sorted(missing))}")


def _validate(request):
    if request.method not in {"fraction-growth", "crossing", "tracking"}:
        raise ValueError("Unknown velocity method.")
    if request.axis not in "xyz" or len(request.axis) != 1:
        raise ValueError("axis must be x, y, or z.")
    if request.time_unit not in TIME_SECONDS or request.initial_sign not in {-1, 1}:
        raise ValueError("Choose a physical time unit and initial_sign +1 or -1.")
    for name in ("time_per_iteration", "length", "max_step", "periodic_length"):
        value = getattr(request, name)
        if value is not None and (not np.isfinite(value) or value <= 0):
            raise ValueError(f"{name} must be finite and positive.")
    for name in ("field_start_time", "time_min", "time_max"):
        value = getattr(request, name)
        if value is not None and not np.isfinite(value):
            raise ValueError(f"{name} must be finite.")
    if request.time_min is not None and request.time_max is not None:
        if request.time_max <= request.time_min:
            raise ValueError("time_max must exceed time_min.")
    if request.front_count < 1 or request.persistence < 1:
        raise ValueError("front_count and persistence must be positive integers.")
    if not 0 <= request.coordinate_decimals <= 10:
        raise ValueError("coordinate_decimals must be between 0 and 10.")
    if request.method == "fraction-growth" and request.length is None:
        raise ValueError("Fraction growth requires an explicit physical --length in angstrom.")
    if request.method == "tracking" and request.max_step is None:
        raise ValueError("Tracking requires --max-step in angstrom per saved frame.")
    if request.periodic_length is not None:
        if request.method != "tracking":
            raise ValueError("periodic_length is supported only for tracking.")
        if request.max_step >= request.periodic_length / 2:
            raise ValueError("max_step must be less than half the periodic length.")


def _times(table, folder, request):
    table = table.copy()
    _require(table, ["frame_index", "iter"])
    if "time" not in table:
        summary_path = folder / f"{PREFIX}whole_slab_summary.csv"
        if summary_path.exists():
            summary = pd.read_csv(summary_path)
            if "time" in summary:
                _require(summary, ["frame_index", "iter"])
                table = table.merge(
                    summary[["frame_index", "iter", "time"]],
                    on=["frame_index", "iter"], how="left", validate="many_to_one",
                )
    if "time" not in table:
        if request.time_per_iteration is None:
            raise ValueError("No time column. Supply --time-per-iteration in the selected time unit.")
        table["time"] = pd.to_numeric(table["iter"]) * request.time_per_iteration
    table["time"] = pd.to_numeric(table["time"], errors="raise")
    if not np.isfinite(table["time"]).all():
        raise ValueError("Every input row must have finite physical time.")
    frame_times = table[["frame_index", "iter", "time"]].drop_duplicates()
    if frame_times["frame_index"].duplicated().any():
        raise ValueError("Conflicting iteration/time values for a frame.")
    ordered = frame_times.sort_values("frame_index")
    if np.any(np.diff(ordered["time"]) <= 0):
        raise ValueError("Frame times must be strictly increasing.")
    for column in ("component", "polarity_component", "local_grouping", "bin_reference_frame"):
        if column in table and table[column].nunique(dropna=False) != 1:
            raise ValueError(f"Mixed {column} values are not supported in one analysis.")
    table["source_time"] = table["time"]
    table["time"] -= request.field_start_time
    lower = max(0.0, request.time_min if request.time_min is not None else 0.0)
    table = table[table["time"] >= lower]
    if request.time_max is not None:
        table = table[table["time"] <= request.time_max]
    if table["time"].nunique() < 2:
        raise ValueError("At least two frames must remain in the selected time window.")
    return table


def _fit(time, position, request, **metadata):
    time, position = np.asarray(time, float), np.asarray(position, float)
    valid = np.isfinite(time) & np.isfinite(position)
    time, position = time[valid], position[valid]
    row = dict(metadata, n_points=len(time), velocity_angstrom_per_time=np.nan,
               velocity_m_per_s=np.nan, r_squared=np.nan, status="insufficient_points")
    if len(time) < 2 or np.ptp(time) == 0:
        return row
    slope, intercept = np.polyfit(time - time[0], position, 1)
    residual = position - (intercept + slope * (time - time[0]))
    total = np.sum((position - position.mean()) ** 2)
    row.update(velocity_angstrom_per_time=slope,
               velocity_m_per_s=slope * 1e-10 / TIME_SECONDS[request.time_unit],
               r_squared=1 - np.sum(residual ** 2) / total if total > 0 else np.nan,
               time_start=time.min(), time_end=time.max(), status="ok")
    return row


def _bins(table, request):
    _require(table, ["u_axis", "v_axis", "u_bin", "v_bin", "positive_count",
                     "negative_count", "zero_count", "defined_group_count"])
    if table["u_axis"].nunique() != 1 or table["v_axis"].nunique() != 1:
        raise ValueError("Input must contain one projection plane.")
    direction = next((name for name in ("u", "v") if table[f"{name}_axis"].iloc[0] == request.axis), None)
    if direction is None:
        raise ValueError("Requested axis is absent from the projection plane.")
    transverse = "v" if direction == "u" else "u"
    _require(table, [f"{direction}_center (angstrom)"])
    table = table.copy()
    table["bin"] = table[f"{direction}_bin"]
    table["position"] = table[f"{direction}_center (angstrom)"]
    table["lane"] = table[f"{transverse}_bin"].astype(str)
    if table.duplicated(["frame_index", "lane", "bin"]).any():
        raise ValueError("Duplicate spatial bins within a frame.")
    expected_bins = len(table[["lane", "bin"]].drop_duplicates())
    if not table.groupby("frame_index").size().eq(expected_bins).all():
        raise ValueError("Every frame must contain the same complete set of spatial bins.")
    if not np.isfinite(table["position"].to_numpy(float)).all():
        raise ValueError("Bin-center coordinates must be finite.")
    counts = table[["positive_count", "negative_count", "zero_count", "defined_group_count"]].to_numpy(float)
    if not np.isfinite(counts).all() or (counts < 0).any() or not np.allclose(counts, np.round(counts)):
        raise ValueError("Polarity counts must be finite nonnegative integers.")
    if not np.allclose(counts[:, :3].sum(axis=1), counts[:, 3]):
        raise ValueError("Positive, negative and zero counts must sum to defined_group_count.")
    if table.groupby("bin")["position"].nunique().max() != 1:
        raise ValueError("Reference bin coordinates must remain fixed.")
    if request.aggregate_transverse:
        table["lane"] = "all"
        table = table.groupby(["frame_index", "iter", "time", "source_time", "lane", "bin", "position"], as_index=False)[
            ["positive_count", "negative_count", "zero_count", "defined_group_count"]
        ].sum()
    denominator = table["defined_group_count"].replace(0, np.nan)
    table["mean_polarity"] = (table["positive_count"] - table["negative_count"]) / denominator
    target = "negative_count" if request.initial_sign == 1 else "positive_count"
    table["switched_fraction"] = table[target] / denominator
    return table.sort_values(["lane", "bin", "time"])


def _fraction_intervals(table, request, *, scope, length):
    keys = ["lane", "bin"] if scope == "bin" else ["lane"]
    ordered = table.sort_values([*keys, "time"]).copy()
    previous = ordered.groupby(keys, sort=False)[
        ["time", "frame_index", "switched_fraction", "defined_group_count"]
    ].shift()
    intervals = pd.DataFrame({
        "scope": scope,
        "lane": ordered["lane"],
        "bin": ordered["bin"] if scope == "bin" else -1,
        "position_angstrom": ordered["position"] if scope == "bin" else np.nan,
        "frame_start": previous["frame_index"],
        "frame_end": ordered["frame_index"],
        "time_start": previous["time"],
        "time_end": ordered["time"],
        "fraction_start": previous["switched_fraction"],
        "fraction_end": ordered["switched_fraction"],
        "length_angstrom": length,
        "front_count": request.front_count,
    })
    intervals["time_mid"] = (intervals["time_start"] + intervals["time_end"]) / 2
    intervals["velocity_angstrom_per_time"] = (
        length / request.front_count
        * (intervals["fraction_end"] - intervals["fraction_start"])
        / (intervals["time_end"] - intervals["time_start"])
    )
    intervals["status"] = "ok"
    changed = ordered["defined_group_count"] != previous["defined_group_count"]
    intervals.loc[changed, "status"] = "changing_defined_population"
    undefined = ~np.isfinite(intervals["fraction_start"]) | ~np.isfinite(intervals["fraction_end"])
    intervals.loc[undefined, "status"] = "undefined_fraction"
    intervals.loc[intervals["status"] != "ok", "velocity_angstrom_per_time"] = np.nan
    intervals["velocity_m_per_s"] = intervals["velocity_angstrom_per_time"] * 1e-10 / TIME_SECONDS[request.time_unit]
    intervals["speed_m_per_s"] = intervals["velocity_m_per_s"].abs()
    return intervals[intervals["frame_start"].notna()].reset_index(drop=True)


def fraction_growth(table, request):
    """Fit count-weighted target-domain fractions within transverse slices."""
    target = "negative_count" if request.initial_sign == 1 else "positive_count"
    samples = table.groupby(["lane", "frame_index", "time"], as_index=False)[
        [target, "defined_group_count"]
    ].sum()
    samples["switched_fraction"] = samples[target] / samples["defined_group_count"].replace(0, np.nan)
    samples["effective_distance_angstrom"] = samples["switched_fraction"] * request.length / request.front_count
    rows = [_fit(group["time"], group["effective_distance_angstrom"], request, lane=lane,
                 estimate="effective_growth", length_angstrom=request.length, front_count=request.front_count)
            for lane, group in samples.groupby("lane", sort=False)]
    intervals = pd.concat([
        _fraction_intervals(table, request, scope="bin", length=request.length / table["bin"].nunique()),
        _fraction_intervals(samples, request, scope="slice", length=request.length),
    ], ignore_index=True)
    return VelocityResult(samples, pd.DataFrame(rows), pd.DataFrame(columns=["lane", "status"]), intervals)


def crossing_velocity(table, request):
    """Fit positions versus persistent sign-crossing times in each spatial lane."""
    events, diagnostics = [], []
    timeline = np.sort(table["time"].unique())
    frame_by_time = table.drop_duplicates("time").set_index("time")["frame_index"]
    for (lane, spatial_bin), group in table.groupby(["lane", "bin"], sort=False):
        series = group.set_index("time")["mean_polarity"].reindex(timeline).to_numpy() * request.initial_sign
        candidates = []
        for index in range(1, len(timeline)):
            if not (series[index - 1] > 0 and series[index] <= 0):
                continue
            after = series[index:index + request.persistence]
            if len(after) < request.persistence or not np.all(after <= 0):
                continue
            fraction = series[index - 1] / (series[index - 1] - series[index])
            candidates.append(dict(
                time=timeline[index - 1] + fraction * (timeline[index] - timeline[index - 1]),
                frame_before=frame_by_time.loc[timeline[index - 1]],
                frame_after=frame_by_time.loc[timeline[index]],
            ))
        status = "ok" if len(candidates) == 1 else "multiple_crossings" if candidates else "no_crossing"
        if not candidates and np.isfinite(series[0]) and series[0] <= 0:
            status = "already_crossed"
        diagnostics.append(dict(lane=lane, bin=spatial_bin, status=status, crossing_count=len(candidates)))
        if len(candidates) == 1:
            events.append(dict(lane=lane, bin=spatial_bin, position=group["position"].iloc[0], **candidates[0]))
    samples = pd.DataFrame(events, columns=["lane", "bin", "position", "time", "frame_before", "frame_after"])
    velocities = []
    intervals = []
    for lane in table["lane"].unique():
        group = samples[samples["lane"] == lane].sort_values("position")
        row = _fit(group["time"], group["position"], request, lane=lane, estimate="crossing_front")
        differences = np.diff(group["time"])
        if len(group) >= 2 and not (np.all(differences > 0) or np.all(differences < 0)):
            row.update(status="nonmonotonic_arrivals", velocity_angstrom_per_time=np.nan, velocity_m_per_s=np.nan)
        if len(group) >= 2 and np.any(np.diff(group["bin"]) != 1):
            row.update(status="noncontiguous_crossings", velocity_angstrom_per_time=np.nan, velocity_m_per_s=np.nan)
        velocities.append(row)
        spatial = table[table["lane"] == lane][["bin", "position"]].drop_duplicates().sort_values("position")
        spatial = spatial.merge(group, on=["bin", "position"], how="left").reset_index(drop=True)
        for index in range(1, len(spatial)):
            lower, upper = spatial.iloc[index - 1], spatial.iloc[index]
            interval = dict(
                lane=lane, bin_lower=lower["bin"], bin_upper=upper["bin"],
                position_lower_angstrom=lower["position"], position_upper_angstrom=upper["position"],
                position_mid_angstrom=(lower["position"] + upper["position"]) / 2,
                crossing_time_lower=lower["time"], crossing_time_upper=upper["time"],
                lower_frame_before=lower["frame_before"], lower_frame_after=lower["frame_after"],
                upper_frame_before=upper["frame_before"], upper_frame_after=upper["frame_after"],
                time_start=np.nan, time_end=np.nan, time_mid=np.nan,
                velocity_angstrom_per_time=np.nan, velocity_m_per_s=np.nan, speed_m_per_s=np.nan,
                status="missing_crossing",
            )
            if np.isfinite(lower["time"]) and np.isfinite(upper["time"]):
                delta_time = upper["time"] - lower["time"]
                interval.update(time_start=min(lower["time"], upper["time"]),
                                time_end=max(lower["time"], upper["time"]),
                                time_mid=(lower["time"] + upper["time"]) / 2,
                                status="simultaneous_crossings" if delta_time == 0 else row["status"])
                if interval["status"] == "ok":
                    velocity = (upper["position"] - lower["position"]) / delta_time
                    interval.update(velocity_angstrom_per_time=velocity,
                                    velocity_m_per_s=velocity * 1e-10 / TIME_SECONDS[request.time_unit])
                    interval["speed_m_per_s"] = abs(interval["velocity_m_per_s"])
            intervals.append(interval)
    interval_table = pd.DataFrame(intervals) if intervals else pd.DataFrame(columns=[
        "lane", "bin_lower", "bin_upper", "time_mid", "position_mid_angstrom",
        "velocity_m_per_s", "speed_m_per_s", "status",
    ])
    return VelocityResult(samples, pd.DataFrame(velocities), pd.DataFrame(diagnostics), interval_table)


def tracking_velocity(table, request):
    """Track isolated sign boundaries on reference-coordinate lines, rejecting ambiguous matches."""
    coordinates = [f"reference_center_{axis} (angstrom)" for axis in "xyz"]
    group_id_column = "local_neighborhood_id" if "local_neighborhood_id" in table else "local_group_id"
    _require(table, [group_id_column, "polarity", *coordinates])
    if table.duplicated(["frame_index", group_id_column]).any():
        raise ValueError("Duplicate local group within a frame.")
    if not np.isfinite(table[coordinates].to_numpy(float)).all():
        raise ValueError("Reference coordinates must be finite.")
    if (table.groupby(group_id_column)[coordinates].nunique().to_numpy() > 1).any():
        raise ValueError("Reference coordinates must be fixed for each group.")
    defined = table["polarity"].dropna()
    if not defined.isin([-1, 0, 1]).all():
        raise ValueError("polarity must be -1, 0, +1 or NaN.")
    table = table.copy()
    transverse = [column for column in coordinates if column != f"reference_center_{request.axis} (angstrom)"]
    table["lane"] = list(map(tuple, table[transverse].round(request.coordinate_decimals).to_numpy()))
    table["position"] = table[f"reference_center_{request.axis} (angstrom)"]
    frames = table[["frame_index", "time"]].drop_duplicates().sort_values("time")
    samples, diagnostics = [], []
    next_track = 0
    for lane, lane_table in table.groupby("lane", sort=False):
        previous = []
        expected = lane_table[group_id_column].nunique()
        for frame in frames.itertuples(index=False):
            group = lane_table[lane_table["frame_index"] == frame.frame_index].sort_values("position")
            positions, signs = group["position"].to_numpy(), group["polarity"].to_numpy()
            boundaries = []
            if len(group) != expected or len(np.unique(positions)) != len(positions):
                diagnostics.append(dict(lane=str(lane), frame_index=frame.frame_index, status="missing_or_duplicate_sites"))
                previous = []
                continue
            if request.periodic_length is not None and len(positions):
                if np.ptp(positions) >= request.periodic_length:
                    raise ValueError("periodic_length must exceed each lane's reference-coordinate span.")
                positions = np.append(positions, positions[0] + request.periodic_length)
                signs = np.append(signs, signs[0])
            for index in range(1, len(positions)):
                if signs[index - 1] * signs[index] == -1:
                    boundaries.append(dict(position=(positions[index - 1] + positions[index]) / 2,
                                           orientation=int(signs[index] - signs[index - 1])))
            distances = np.empty((len(previous), len(boundaries)))
            for old_index, old in enumerate(previous):
                for new_index, new in enumerate(boundaries):
                    delta = new["position"] - old["position"]
                    if request.periodic_length is not None:
                        delta = (delta + request.periodic_length / 2) % request.periodic_length - request.periodic_length / 2
                    distances[old_index, new_index] = delta if old["orientation"] == new["orientation"] else np.inf
            allowed = np.abs(distances) <= request.max_step
            for new_index, boundary in enumerate(boundaries):
                candidates = np.flatnonzero(allowed[:, new_index])
                if len(candidates) == 1 and allowed[candidates[0]].sum() == 1:
                    old_index = candidates[0]
                    boundary["track_id"] = previous[old_index]["track_id"]
                    boundary["unwrapped_position"] = previous[old_index]["unwrapped_position"] + distances[old_index, new_index]
                else:
                    boundary["track_id"] = next_track
                    next_track += 1
                    boundary["unwrapped_position"] = boundary["position"]
                    if len(candidates):
                        diagnostics.append(dict(lane=str(lane), frame_index=frame.frame_index, status="ambiguous_match_new_track"))
                samples.append(dict(lane=str(lane), frame_index=frame.frame_index, time=frame.time, **boundary))
            if np.any(~np.isfinite(signs)) or np.any(signs == 0):
                diagnostics.append(dict(lane=str(lane), frame_index=frame.frame_index, status="zero_or_undefined_sites"))
            previous = boundaries
    samples = pd.DataFrame(samples, columns=["lane", "frame_index", "time", "position", "orientation", "track_id", "unwrapped_position"])
    rows = [_fit(group["time"], group["unwrapped_position"], request, lane=group["lane"].iloc[0],
                 track_id=track, orientation=group["orientation"].iloc[0], estimate="local_boundary")
            for track, group in samples.groupby("track_id", sort=False)]
    velocities = pd.DataFrame(rows) if rows else pd.DataFrame(columns=[
        "lane", "track_id", "orientation", "estimate", "n_points", "status",
        "velocity_angstrom_per_time", "velocity_m_per_s", "r_squared",
    ])
    return VelocityResult(samples, velocities, pd.DataFrame(diagnostics, columns=["lane", "frame_index", "status"]))


def calculate_domain_wall_velocity(folder: Path | str, request: VelocityRequest) -> VelocityResult:
    """Read projected-polarity CSVs; never load a trajectory or rerun polarity."""
    _validate(request)
    folder = Path(folder)
    suffix = "centers" if request.method == "tracking" else "2d"
    source = folder / f"{PREFIX}{suffix}.csv"
    if not source.is_file():
        raise ValueError(f"Required CSV not found: {source}")
    table = _times(pd.read_csv(source), folder, request)
    if request.method == "tracking":
        result = tracking_velocity(table, request)
    else:
        table = _bins(table, request)
        result = fraction_growth(table, request) if request.method == "fraction-growth" else crossing_velocity(table, request)
    for output in (result.samples, result.velocities, result.diagnostics, result.intervals):
        output["method"] = request.method
        output["axis"] = request.axis
        output["time_unit"] = request.time_unit
    return result
