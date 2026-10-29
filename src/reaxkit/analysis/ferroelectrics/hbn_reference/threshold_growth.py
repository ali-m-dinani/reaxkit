"""Threshold-based effective growth velocities from user-defined fractions."""

from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd

from reaxkit.analysis.ferroelectrics.hbn_reference.domain_wall_velocity import TIME_SECONDS


def threshold_velocity(
    samples: pd.DataFrame,
    *,
    threshold: float,
    field_start_time: float,
    length: float,
    time_unit: str = "ps",
    front_count: int = 1,
    lag_threshold: float | None = None,
) -> pd.DataFrame:
    """Measure each bin from field onset to its first observed value above threshold.

    Input columns are bin, time, and switched_fraction. Times remain absolute.
    The baseline is interpolated only when the onset falls between two samples.
    The threshold endpoint is an actual sample, not an interpolated crossing.
    With lag_threshold, measure growth from the first sample above baseline plus
    that increment. elapsed_time remains onset-to-endpoint; rise_time excludes lag.
    """
    required = {"bin", "time", "switched_fraction"}
    if not required.issubset(samples.columns):
        raise ValueError("Expected bin, time, and switched_fraction columns.")
    if not np.isfinite(threshold) or not 0 < threshold <= 1:
        raise ValueError("threshold must be in (0, 1].")
    if lag_threshold is not None and (not np.isfinite(lag_threshold) or not 0 < lag_threshold < 1):
        raise ValueError("lag_threshold must be finite and in (0, 1).")
    if not np.isfinite(field_start_time) or not np.isfinite(length) or length <= 0:
        raise ValueError("Field onset must be finite and length must be finite and positive.")
    if time_unit not in TIME_SECONDS or front_count < 1:
        raise ValueError("Use a physical time unit and positive front count.")
    if samples.empty or samples["bin"].isna().any():
        raise ValueError("At least one nonempty bin is required.")
    values = samples[["time", "switched_fraction"]].to_numpy(float)
    if not np.isfinite(values).all() or np.any((values[:, 1] < 0) | (values[:, 1] > 1)):
        raise ValueError("Times and fractions must be finite, with fractions in [0, 1].")
    if samples.duplicated(["bin", "time"]).any():
        raise ValueError("Each bin/time pair must be unique.")
    rows = []
    for spatial_bin, group in samples.groupby("bin", sort=True):
        group = group.sort_values("time")
        times = group["time"].to_numpy(float)
        fractions = group["switched_fraction"].to_numpy(float)
        row = dict(bin=spatial_bin, threshold=threshold, field_start_time=field_start_time,
                   field_fraction=np.nan, baseline_interpolated=False,
                   threshold_time=np.nan, threshold_fraction=np.nan,
                   elapsed_time=np.nan, fraction_change=np.nan, length_angstrom=length,
                   front_count=front_count, time_unit=time_unit,
                   velocity_angstrom_per_time=np.nan, velocity_m_per_s=np.nan,
                   status="onset_outside_data")
        if times[0] <= field_start_time <= times[-1]:
            baseline = float(np.interp(field_start_time, times, fractions))
            row.update(field_fraction=baseline,
                       baseline_interpolated=not np.any(times == field_start_time),
                       status="threshold_not_reached")
            if baseline - threshold > 1e-12:
                row["status"] = "already_above_threshold"
            else:
                candidates = np.flatnonzero((times > field_start_time) & (fractions - threshold > 1e-12))
                if candidates.size:
                    index = candidates[0]
                    elapsed = times[index] - field_start_time
                    change = fractions[index] - baseline
                    velocity = length * change / elapsed / front_count
                    row.update(threshold_time=times[index], threshold_fraction=fractions[index],
                               elapsed_time=elapsed, fraction_change=change,
                               velocity_angstrom_per_time=velocity,
                               velocity_m_per_s=velocity * 1e-10 / TIME_SECONDS[time_unit], status="ok")
        if lag_threshold is not None:
            row.update(lag_threshold=lag_threshold, lag_target_fraction=np.nan,
                       rise_start_time=np.nan, rise_start_fraction=np.nan,
                       lag_time=np.nan, rise_time=np.nan)
            if np.isfinite(row["field_fraction"]):
                lag_target = row["field_fraction"] + lag_threshold
                row["lag_target_fraction"] = lag_target
                candidates = np.flatnonzero((times > field_start_time) & (fractions - lag_target > 1e-12))
                if row["status"] != "already_above_threshold":
                    if threshold - lag_target <= 1e-12:
                        row["status"] = "lag_target_at_or_above_threshold"
                    elif not candidates.size:
                        row["status"] = "lag_not_reached"
                    else:
                        start_index = candidates[0]
                        row.update(rise_start_time=times[start_index],
                                   rise_start_fraction=fractions[start_index],
                                   lag_time=times[start_index] - field_start_time)
                        if np.isfinite(row["threshold_time"]):
                            rise_time = row["threshold_time"] - times[start_index]
                            row["rise_time"] = rise_time
                            if rise_time <= 0:
                                row["status"] = "unresolved_rise"
                            else:
                                change = row["threshold_fraction"] - fractions[start_index]
                                velocity = length * change / rise_time / front_count
                                row.update(fraction_change=change, velocity_angstrom_per_time=velocity,
                                           velocity_m_per_s=velocity * 1e-10 / TIME_SECONDS[time_unit])
            if row["status"] != "ok":
                row.update(fraction_change=np.nan, velocity_angstrom_per_time=np.nan, velocity_m_per_s=np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


def read_fraction_tables(source: Path, *, fraction_column: str = "switched_fraction", initial_sign: int = 1) -> dict[str, pd.DataFrame]:
    """Read a user-supplied xz grid or an x/z-bin by time fraction matrix."""
    if initial_sign not in {-1, 1}:
        raise ValueError("initial_sign must be +1 or -1.")
    table = pd.read_csv(source)
    if table.empty:
        raise ValueError("The fraction CSV is empty.")
    table.columns = table.columns.str.strip()
    if "defined_group_count" not in table and "defined_cell_count" in table:
        table["defined_group_count"] = table["defined_cell_count"]
    if "time" in table and {"u_bin", "v_bin", "u_axis", "v_axis"}.issubset(table.columns):
        if fraction_column not in table and fraction_column == "switched_fraction" and "switched fraction" in table:
            fraction_column = "switched fraction"
        if fraction_column not in table:
            if fraction_column not in {"switched_fraction", "switched fraction"}:
                raise ValueError(f"Missing requested fraction column: {fraction_column!r}.")
            count_column = "positive_count" if initial_sign == 1 else "negative_count"
            required = {count_column, "defined_group_count"}
            if not required.issubset(table.columns):
                raise ValueError(f"Missing CSV columns for fraction calculation: {', '.join(sorted(required - set(table.columns)))}")
            counts = table[[count_column, "defined_group_count"]].apply(pd.to_numeric, errors="raise")
            if not np.isfinite(counts.to_numpy()).all() or (counts < 0).any().any():
                raise ValueError("Polarity and defined-group counts must be finite and nonnegative.")
            if (counts[count_column] > counts["defined_group_count"]).any():
                raise ValueError("Polarity count cannot exceed defined_group_count.")
            table[fraction_column] = (1 - counts[count_column] / counts["defined_group_count"].replace(0, np.nan)).fillna(0)
        if not table["u_axis"].eq("x").all() or not table["v_axis"].eq("z").all():
            raise ValueError("Expected an xz projection for x/z fraction tables.")
        for column in ("time", "u_bin", "v_bin", fraction_column):
            table[column] = pd.to_numeric(table[column], errors="raise")
        fractions = table[fraction_column].to_numpy(float)
        if not np.isfinite(fractions).all() or np.any((fractions < 0) | (fractions > 1)):
            raise ValueError("User-defined switched fractions must be finite and in [0, 1].")
        if not np.isfinite(table[["time", "u_bin", "v_bin"]].to_numpy()).all():
            raise ValueError("Time and bin identifiers must be finite.")
        if table.duplicated(["time", "u_bin", "v_bin"]).any():
            raise ValueError("Duplicate time/x-bin/z-bin rows.")
        expected = len(table[["u_bin", "v_bin"]].drop_duplicates())
        if expected != table["u_bin"].nunique() * table["v_bin"].nunique() or not table.groupby("time").size().eq(expected).all():
            raise ValueError("Each time must contain a complete rectangular grid.")
        weights = pd.to_numeric(table["defined_group_count"], errors="raise") if "defined_group_count" in table else pd.Series(1.0, index=table.index)
        if not np.isfinite(weights).all() or (weights < 0).any():
            raise ValueError("Defined-group counts must be finite and nonnegative.")
        table["weight"] = weights
        table["weighted_fraction"] = table[fraction_column] * weights
        outputs = {}
        for axis, column in (("x", "u_bin"), ("z", "v_bin")):
            grouped = table.groupby(["time", column], as_index=False)[["weight", "weighted_fraction"]].sum()
            if grouped["weight"].le(0).any():
                raise ValueError("Cannot aggregate a slice with no defined groups.")
            grouped["switched_fraction"] = (grouped["weighted_fraction"] / grouped["weight"]).clip(0, 1)
            outputs[axis] = grouped.rename(columns={column: "bin"})[["bin", "time", "switched_fraction"]]
            prefix = column[0]
            coordinate_columns = [f"{prefix}_{part} (angstrom)" for part in ("min", "max", "center")]
            if all(name in table for name in coordinate_columns):
                coordinates = table[[column, *coordinate_columns]].copy()
                coordinates[coordinate_columns] = coordinates[coordinate_columns].apply(pd.to_numeric, errors="raise")
                if not np.isfinite(coordinates[coordinate_columns].to_numpy()).all():
                    raise ValueError("Bin coordinates must be finite.")
                first = coordinates.groupby(column)[coordinate_columns].first()
                for spatial_bin, values in coordinates.groupby(column):
                    if not np.allclose(values[coordinate_columns], first.loc[spatial_bin].to_numpy(), rtol=0, atol=1e-8):
                        raise ValueError("Bin coordinates must remain fixed across times and transverse bins.")
                first.columns = ["bin_min_angstrom", "bin_max_angstrom", "bin_center_angstrom"]
                if (first.bin_min_angstrom > first.bin_center_angstrom).any() or (first.bin_center_angstrom > first.bin_max_angstrom).any():
                    raise ValueError("Bin centers must lie within their bounds.")
                first["bin_position"] = "interior"
                low = "low-z (bottom)" if axis == "z" else "low-x"
                high = "high-z (top)" if axis == "z" else "high-x"
                first.loc[first.bin_center_angstrom.idxmin(), "bin_position"] = low
                first.loc[first.bin_center_angstrom.idxmax(), "bin_position"] = high
                if len(first) == 1:
                    first["bin_position"] = f"{low} to {high}"
                outputs[axis] = outputs[axis].merge(first.rename_axis("bin").reset_index(), on="bin", validate="many_to_one")
        return outputs
    label = str(table.columns[0])
    match = re.fullmatch(r"([xz])_bin(?:/time\([^)]*\))?", label)
    if match is None:
        raise ValueError("Use an xz grid CSV with a switched_fraction column, or a matrix beginning with x_bin/time(ps) or z_bin/time(ps).")
    times = []
    for column in table.columns[1:]:
        normalized = re.sub(r"^time_(.+)_(?:fs|ps|ns|s)$", r"\1", column)
        try:
            times.append(float(normalized))
        except ValueError as exc:
            raise ValueError("Matrix time columns must contain fractions at each time; a single added switched_fraction column cannot represent a time series.") from exc
    if not times or not np.isfinite(times).all() or len(set(times)) != len(times):
        raise ValueError("Matrix time columns must be finite and unique.")
    table = table.rename(columns={label: "bin", **dict(zip(table.columns[1:], times))})
    samples = table.melt(id_vars="bin", var_name="time", value_name="switched_fraction")
    for column in samples:
        samples[column] = pd.to_numeric(samples[column], errors="raise")
    return {match[1]: samples.sort_values(["bin", "time"])}


def instantaneous_velocity(samples: pd.DataFrame, endpoints: pd.DataFrame) -> pd.DataFrame:
    """Return finite-difference rates after onset, retaining the final plateau."""
    rows = []
    for endpoint in endpoints.itertuples(index=False):
        group = samples[samples["bin"] == endpoint.bin].sort_values("time")
        group = group[group["time"] >= endpoint.field_start_time].copy()
        if endpoint.baseline_interpolated:
            group = pd.concat([pd.DataFrame([dict(bin=endpoint.bin, time=endpoint.field_start_time,
                                                 switched_fraction=endpoint.field_fraction)]), group], ignore_index=True)
        times = group["time"].to_numpy(float)
        fractions = group["switched_fraction"].to_numpy(float)
        for index in range(1, len(group)):
            velocity = endpoint.length_angstrom * (fractions[index] - fractions[index - 1]) / (
                times[index] - times[index - 1]) / endpoint.front_count
            rows.append(dict(bin=endpoint.bin, time_start=times[index - 1], time_end=times[index],
                             time_mid=(times[index - 1] + times[index]) / 2,
                             elapsed_time_mid=(times[index - 1] + times[index]) / 2 - endpoint.field_start_time,
                             fraction_start=fractions[index - 1], fraction_end=fractions[index],
                             velocity_angstrom_per_time=velocity,
                             velocity_m_per_s=velocity * 1e-10 / TIME_SECONDS[endpoint.time_unit],
                             average_velocity_m_per_s=endpoint.velocity_m_per_s,
                             time_unit=endpoint.time_unit))
    return pd.DataFrame(rows, columns=["bin", "time_start", "time_end", "time_mid", "elapsed_time_mid",
                                      "fraction_start", "fraction_end", "velocity_angstrom_per_time",
                                      "velocity_m_per_s", "average_velocity_m_per_s", "time_unit"])
