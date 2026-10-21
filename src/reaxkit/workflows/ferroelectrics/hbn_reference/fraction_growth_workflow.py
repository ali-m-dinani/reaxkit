"""Threshold growth analysis and per-bin plots from user-defined fraction CSVs."""

from __future__ import annotations

from pathlib import Path
import json
import re

import pandas as pd

from reaxkit.analysis.ferroelectrics.hbn_reference.threshold_growth import (
    instantaneous_velocity, read_fraction_tables, threshold_velocity,
)
from reaxkit.core.storage.storage_layout import default_project_root, generate_run_id
from reaxkit.presentation.plot_styles import plot_render_context, resolve_plot_style
from reaxkit.presentation.workflow_artifacts import write_workflow_csv


COMMAND = "get-hbn-domain-wall-fraction-velocity"

LAG_README = """With `--lag-threshold`, the baseline is the fraction at field onset
(interpolated if bracketed). The rise starts at the first recorded sample strictly
above baseline + lag_threshold; equality within 1e-12 does not count.
`lag_time = rise_start_time - field_start_time`.
`rise_time = threshold_time - rise_start_time`.
`elapsed_time` remains the total field-onset-to-threshold duration.
Velocity uses length * (threshold_fraction - rise_start_fraction) / rise_time /
front_count. Times use the declared input unit, lengths are angstrom, and speeds
are also converted to m/s. Neither crossing is interpolated.
The final threshold is absolute, not an increment from baseline.

Unreached lag targets have status `lag_not_reached`; lag targets at or above the
final threshold have status `lag_target_at_or_above_threshold`. If both crossings
occur in one saved sample, status is `unresolved_rise`. These bins have no valid
average velocity. A measured lag is retained even if the final threshold is not
reached. Interval rates still include the lag and late plateau for context.
Summary statistics use rise-only velocities from valid bins. Compare runs using
the same lag threshold and sampling interval; early noise can trigger this first
crossing rule, which does not impose persistence.
"""

OUTPUT_README = """# Switching-speed heterogeneity

## What is summarized?

`velocity_summary.csv` summarizes the onset-to-threshold average velocities
in `velocities.csv`, separately for each analyzed direction:

- x velocity: variation across z-bin slices.
- z velocity: variation across x-bin slices.

For each bin, velocity = length * (threshold_fraction - field_fraction) /
(threshold_time - field_start_time) / front_count, converted to m/s.
The threshold time is the first recorded time after field onset at which
the switched fraction exceeds the chosen threshold. These are endpoint
average velocities, not the instantaneous interval rates.

## Formulas

Only bins with status `ok` and a nonmissing velocity contribute. Each bin
has equal weight. For N valid bin velocities v_i:

- Mean: mu = sum(v_i) / N.
- Population standard deviation: sigma = sqrt(sum((v_i - mu)^2) / N).
- `std_velocity_m_per_s` = sigma, in m/s (population convention, ddof=0).
- `velocity_cv` = sigma / mu, the dimensionless coefficient of variation.
- `speed_heterogeneity_index` = sigma / (mu + sigma) = CV / (1 + CV).

## Interpreting the index

For nonnegative velocities and a positive mean, the index ranges from zero
toward one. It is a chosen descriptive measure, not a universal physical
classification of domain walls. CV itself can exceed one.

| Heterogeneity index | CV | Meaning |
|---|---|---|
| 0 | 0 | All valid bins have identical average speeds. |
| 0.091 (approximately) | 0.1 | Standard deviation is 10% of the mean. |
| 0.2 | 0.25 | Standard deviation is 25% of the mean. |
| 0.333 (approximately) | 0.5 | Standard deviation is 50% of the mean. |
| 0.5 | 1 | Standard deviation equals the mean. |
| 0.667 (approximately) | 2 | Standard deviation is twice the mean. |
| Approaching 1 | Very large | Strong relative speed variation. |

These examples explain the scale; they are not validated low/high cutoffs.
For example, mean = 100 m/s and standard deviation = 20 m/s give CV = 0.2
and heterogeneity = 20 / 120 = 0.167 (approximately).

## Limitations and missing values

- Inspect `total_bin_count`, `valid_bin_count`, and `excluded_bin_count`.
  Bins that do not reach the threshold are excluded, not assigned zero speed.
  Excluding slow or unswitched bins can make switching appear more uniform.
- With no valid bins, statistics are blank. With zero mean, CV and the index
  are blank rather than treated as uniform switching.
- One valid positive-speed bin gives zero variation mathematically but
  cannot establish spatial uniformity.
- Similar speeds do not prove that a domain wall is flat. A curved wall
  can move uniformly, and nucleation or multiple fronts can affect fractions.
- This index compares per-bin endpoint averages, not time-dependent wall
  roughness or instantaneous speed heterogeneity. Compare runs using
  consistent binning, threshold, field onset and front-count assumptions.

`velocity_intervals.csv` contains interval rates; `plots/fraction/` and
`plots/velocity/` contain per-bin figures. Combined fraction plots are directly
under `plots/`. `settings.json` records the command settings for this run.
"""


def bin_coordinate_label(samples, bin_number, bin_axis):
    row = samples.loc[samples["bin"] == bin_number].iloc[0]
    if "bin_center_angstrom" not in samples:
        return "coordinates unavailable"
    return (f"{bin_axis}={row.bin_center_angstrom:.4g} Å "
            f"[{row.bin_min_angstrom:.4g}, {row.bin_max_angstrom:.4g} Å]; {row.bin_position}")


def build_parser(parser, *, command):
    parser.set_defaults(command=command)
    parser.description = (
        "Measure effective growth from field onset to each bin's first observed fraction above a threshold. "
        "Read your calculated fractions from an xz grid CSV or separate x/z matrices. "
        "An x-bin curve gives z growth; a z-bin curve gives x growth. "
        "Writes transposed fraction tables, endpoint velocities, instantaneous rates and per-bin plots. "
        "Missing switched fractions are calculated from polarity counts using --initial-sign."
    )
    parser.add_argument("--input", type=Path, required=True, help="Modified projected-polarity 2D CSV, or a bin-by-time fraction matrix.")
    parser.add_argument("--second-input", type=Path, help="Other axis's fraction matrix when --input contains only one bin axis.")
    parser.add_argument("--value-column", default="switched_fraction", help="User-calculated fraction column in a grid CSV; also accepts 'switched fraction' by default.")
    parser.add_argument("--initial-sign", type=int, choices=[-1, 1], default=1, help="When switched_fraction is absent, use 1 - positive_count/defined_group_count for +1 (default), or 1 - negative_count/defined_group_count for -1.")
    parser.add_argument("--axis", choices=["x", "z", "both"], default="both", help="Velocity direction(s); x velocity uses z-bin curves and z velocity uses x-bin curves.")
    parser.add_argument("--threshold", type=float, required=True, help="Stop each bin at its first recorded fraction strictly greater than this value, e.g. 0.98.")
    parser.add_argument("--lag-threshold", type=float, help="Start the rise at the first recorded fraction above the field-onset fraction plus this increment, e.g. 0.05; report lag and rise times and use rise-only velocity.")
    parser.add_argument("--field-start-time", type=float, required=True, help="Field onset in original input time units; baseline fraction is interpolated if necessary.")
    parser.add_argument("--time-unit", choices=["fs", "ps", "ns", "s"], default="ps", help="Physical unit of input timestamps; used to convert rates to m/s.")
    parser.add_argument("--length", type=float, help="Physical length in angstrom for a single selected --axis.")
    parser.add_argument("--length-x", type=float, help="Physical x length in angstrom, required for x velocity when --axis both.")
    parser.add_argument("--length-z", type=float, help="Analyzed z thickness in angstrom, excluding vacuum, required for z velocity when --axis both.")
    parser.add_argument("--front-count", type=int, default=1, help="Assumed equally advancing fronts; divides average and instantaneous rates.")
    parser.add_argument("--figure-dpi", type=int, default=180, help="Resolution of automatically generated per-bin PNG plots.")
    parser.add_argument("--output-dir", type=Path, help="Destination; otherwise a unique workspace other/<command>/run_* folder.")
    return parser


def plot_fraction_growth(samples, endpoints, intervals, destination, *, bin_axis, velocity_axis, dpi):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, Normalize

    style = resolve_plot_style()
    written = []
    fraction_directory = destination / "fraction"
    velocity_directory = destination / "velocity"
    fraction_directory.mkdir(parents=True, exist_ok=True)
    velocity_directory.mkdir(parents=True, exist_ok=True)
    with plot_render_context({"plot_style": "publication" if style == "default" else style, "dpi": dpi}):
        for endpoint in endpoints.itertuples(index=False):
            group = samples[samples["bin"] == endpoint.bin].sort_values("time")
            rates = intervals[intervals["bin"] == endpoint.bin]
            label = str(int(endpoint.bin)) if float(endpoint.bin).is_integer() else str(endpoint.bin)
            safe_label = re.sub(r"[^A-Za-z0-9._-]", "_", label)
            title = f"{bin_axis}-bin {label}: {velocity_axis}-direction growth"
            title += "\n" + bin_coordinate_label(samples, endpoint.bin, bin_axis)
            figure, axes = plt.subplots(figsize=(8.2, 5.4))
            axes.plot(group["time"], group["switched_fraction"], color="#0072B2", marker="o",
                      markersize=3, markerfacecolor="#9ECAE1", markeredgecolor="#08306B", label="Switched fraction")
            axes.axhline(endpoint.threshold, color="#555555", linestyle="--", linewidth=1, label="Threshold")
            axes.axvline(endpoint.field_start_time, color="#7B2CBF", linestyle=":", label="Field onset")
            lag_mode = hasattr(endpoint, "lag_threshold")
            if lag_mode and pd.notna(endpoint.lag_target_fraction):
                axes.axhline(endpoint.lag_target_fraction, color="#009E73", linestyle="--", label="Baseline + lag threshold")
            if lag_mode and pd.notna(endpoint.rise_start_time):
                axes.axvline(endpoint.rise_start_time, color="#009E73", linestyle=":",
                             label=f"Rise start (lag {endpoint.lag_time:.4g} {endpoint.time_unit})")
            if endpoint.status == "ok":
                axes.plot([endpoint.rise_start_time if lag_mode else endpoint.field_start_time, endpoint.threshold_time],
                          [endpoint.rise_start_fraction if lag_mode else endpoint.field_fraction, endpoint.threshold_fraction], linestyle="None",
                          marker="o", color="#D62728", markersize=7, label="Average-velocity endpoints")
            axes.set(xlabel=f"Time ({endpoint.time_unit})", ylabel="Switched fraction", ylim=(-0.03, 1.03), title=title)
            axes.grid(False)
            axes.legend(frameon=False)
            figure.tight_layout()
            path = fraction_directory / f"{bin_axis}_bin_{safe_label}_fraction.png"
            figure.savefig(path, dpi=dpi, bbox_inches="tight")
            plt.close(figure)
            written.append(path)

            figure, axes = plt.subplots(figsize=(8.2, 5.4))
            axes.plot(rates["time_mid"], rates["velocity_m_per_s"], linestyle="None", marker="o",
                      markersize=4, color="#0072B2", markerfacecolor="#9ECAE1", markeredgecolor="#08306B",
                      label="Instantaneous (interval rate)")
            if endpoint.status == "ok":
                axes.axhline(endpoint.velocity_m_per_s, color="#D62728", linewidth=2,
                             label=f"{'Rise-to-threshold' if lag_mode else 'Onset-to-threshold'} average: {endpoint.velocity_m_per_s:.4g} m/s")
            else:
                axes.text(0.02, 0.97, f"Average unavailable: {endpoint.status.replace('_', ' ')}",
                          transform=axes.transAxes, va="top", fontsize=9)
            post_onset = group[group["time"] >= endpoint.field_start_time]
            if not post_onset.empty and post_onset["time"].max() > endpoint.field_start_time:
                axes.set_xlim(endpoint.field_start_time, post_onset["time"].max())
            axes.set(xlabel=f"Time ({endpoint.time_unit})", ylabel="Velocity (m/s)", title=title)
            axes.grid(False)
            axes.legend(frameon=False)
            figure.tight_layout()
            path = velocity_directory / f"{bin_axis}_bin_{safe_label}_velocity.png"
            figure.savefig(path, dpi=dpi, bbox_inches="tight")
            plt.close(figure)
            written.append(path)
        figure, axes = plt.subplots(figsize=(8.2, 5.4))
        bin_numbers = sorted(samples["bin"].unique())
        normalization = Normalize(vmin=bin_numbers[0], vmax=bin_numbers[-1] if len(bin_numbers) > 1 else bin_numbers[0] + 1)
        colormap = LinearSegmentedColormap.from_list("bin_gradient", ["#0000FF", "#FF0000"])
        for bin_number, group in samples.groupby("bin", sort=True):
            group = group.sort_values("time")
            axes.plot(group["time"], group["switched_fraction"],
                      color=colormap(normalization(bin_number)), linewidth=1.6)
        endpoint = endpoints.iloc[0]
        axes.axhline(endpoint.threshold, color="#555555", linestyle="--", linewidth=1, label="Threshold")
        axes.axvline(endpoint.field_start_time, color="#555555", linestyle=":", label="Field onset")
        colorbar = figure.colorbar(matplotlib.cm.ScalarMappable(norm=normalization, cmap=colormap), ax=axes)
        colorbar.set_label(f"{bin_axis}-bin number")
        if "bin_center_angstrom" in samples:
            ticks = bin_numbers[::max(1, (len(bin_numbers) + 6) // 7)]
            if ticks[-1] != bin_numbers[-1]:
                ticks.append(bin_numbers[-1])
            colorbar.set_ticks(ticks)
            centers = samples.drop_duplicates("bin").set_index("bin")["bin_center_angstrom"]
            colorbar.set_ticklabels([f"{number:g}: {centers.loc[number]:.4g} Å" for number in ticks])
            colorbar.set_label(f"{bin_axis}-bin: center coordinate")
        else:
            colorbar.set_ticks(bin_numbers[::max(1, (len(bin_numbers) + 6) // 7)])
        axes.set(xlabel=f"Time ({endpoint.time_unit})", ylabel="Switched fraction", ylim=(-0.03, 1.03),
                 title=f"All {bin_axis} bins: {velocity_axis}-direction growth")
        if "bin_center_angstrom" in samples:
            axes.set_title(axes.get_title() + f"\n{bin_axis}: {samples.bin_min_angstrom.min():.4g} to {samples.bin_max_angstrom.max():.4g} Å"
                           + (" (bottom = low z; top = high z)" if bin_axis == "z" else " (low x to high x)"))
        axes.grid(False)
        axes.legend(frameon=False)
        figure.tight_layout()
        path = destination / f"{bin_axis}_all_bins_fraction.png"
        figure.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(figure)
        written.append(path)
    return written


def run_main(command, args):
    if args.figure_dpi <= 0:
        raise ValueError("figure-dpi must be positive.")
    tables = read_fraction_tables(args.input, fraction_column=args.value_column, initial_sign=args.initial_sign)
    if args.second_input is not None:
        additional = read_fraction_tables(args.second_input, fraction_column=args.value_column, initial_sign=args.initial_sign)
        if tables.keys() & additional.keys():
            raise ValueError("Both inputs provide the same bin axis; supply one x and one z matrix.")
        tables.update(additional)
    directions = ["x", "z"] if args.axis == "both" else [args.axis]
    if args.axis == "both" and args.length is not None:
        raise ValueError("Use --length-x and --length-z for --axis both.")
    summaries, interval_tables = [], []
    results = []
    for direction in directions:
        bin_axis = "z" if direction == "x" else "x"
        if bin_axis not in tables:
            raise ValueError(f"{direction} velocity needs {bin_axis}-bin fraction curves. Supply --second-input or an xz grid CSV.")
        length = getattr(args, f"length_{direction}")
        if length is not None and args.length is not None:
            raise ValueError("Specify either --length or the direction-specific length, not both.")
        length = length if length is not None else args.length
        if length is None:
            raise ValueError(f"Supply --length-{direction} in angstrom (or --length for a single axis).")
        endpoints = threshold_velocity(tables[bin_axis], threshold=args.threshold,
                                       field_start_time=args.field_start_time, length=length,
                                       time_unit=args.time_unit, front_count=args.front_count,
                                       lag_threshold=args.lag_threshold)
        intervals = instantaneous_velocity(tables[bin_axis], endpoints)
        metadata_columns = [name for name in tables[bin_axis] if name.startswith("bin_")]
        if metadata_columns:
            metadata = tables[bin_axis][["bin", *metadata_columns]].drop_duplicates("bin")
            endpoints = endpoints.merge(metadata, on="bin", validate="one_to_one")
            intervals = intervals.merge(metadata, on="bin", validate="many_to_one")
        for table in (endpoints, intervals):
            table["bin_axis"] = bin_axis
            table["velocity_axis"] = direction
        summaries.append(endpoints)
        interval_tables.append(intervals)
        results.append((bin_axis, direction, endpoints, intervals))
    destination = Path(args.output_dir or default_project_root() / "other" / command / generate_run_id()).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    for bin_axis, samples in tables.items():
        matrix = samples.pivot(index="time", columns="bin", values="switched_fraction").sort_index()
        matrix.columns = [f"{bin_axis}_bin_{number:g}" + (
            f" ({bin_coordinate_label(samples, number, bin_axis)})" if "bin_center_angstrom" in samples else "")
            for number in matrix.columns]
        write_workflow_csv(matrix.reset_index(), destination / f"{bin_axis}_bins_fraction_vs_time.csv", index=False)
    velocities = pd.concat(summaries, ignore_index=True)
    write_workflow_csv(velocities, destination / "velocities.csv", index=False)
    summary_rows = []
    for direction, group in velocities.groupby("velocity_axis", sort=True):
        valid = group.loc[group["status"] == "ok", "velocity_m_per_s"].dropna()
        mean_velocity = valid.mean()
        std_velocity = valid.std(ddof=0)
        velocity_cv = std_velocity / mean_velocity if mean_velocity > 0 else float("nan")
        heterogeneity = std_velocity / (mean_velocity + std_velocity) if mean_velocity > 0 else float("nan")
        summary_rows.append(dict(
            velocity_axis=direction, bin_axis=group["bin_axis"].iloc[0],
            total_bin_count=len(group), valid_bin_count=len(valid),
            excluded_bin_count=len(group) - len(valid),
            mean_velocity_m_per_s=mean_velocity, median_velocity_m_per_s=valid.median(),
            min_velocity_m_per_s=valid.min(), max_velocity_m_per_s=valid.max(),
            std_velocity_m_per_s=std_velocity, velocity_cv=velocity_cv,
            speed_heterogeneity_index=heterogeneity,
        ))
    write_workflow_csv(pd.DataFrame(summary_rows), destination / "velocity_summary.csv", index=False)
    write_workflow_csv(pd.concat(interval_tables, ignore_index=True), destination / "velocity_intervals.csv", index=False)
    plot_directory = destination / "plots"
    plot_directory.mkdir(exist_ok=True)
    for bin_axis, direction, endpoints, intervals in results:
        plot_fraction_growth(tables[bin_axis], endpoints, intervals, plot_directory,
                             bin_axis=bin_axis, velocity_axis=direction, dpi=args.figure_dpi)
    settings = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    settings.update(method="threshold_fraction_growth", threshold_endpoint="first_recorded_fraction_strictly_above_threshold")
    if args.lag_threshold is not None:
        settings.update(method="lag_threshold_fraction_growth",
                        rise_start="first_recorded_fraction_strictly_above_field_fraction_plus_lag_threshold")
    (destination / "settings.json").write_text(json.dumps(settings, indent=2, default=str), encoding="utf-8")
    readme = OUTPUT_README
    if args.lag_threshold is not None:
        readme = readme.replace("onset-to-threshold", "rise-to-threshold")
        readme = readme.replace("threshold_fraction - field_fraction", "threshold_fraction - rise_start_fraction")
        readme = readme.replace("threshold_time - field_start_time", "rise_time")
        readme += "\n## Lag-aware timing\n\n" + LAG_README
    (destination / "README.md").write_text(readme, encoding="utf-8")
    print(f"Wrote fraction matrices, endpoint velocities, interval rates and per-bin plots to {destination}")
    return 0
