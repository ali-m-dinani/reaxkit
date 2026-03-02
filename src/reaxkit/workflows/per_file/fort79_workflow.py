"""
fort.79 sensitivity-analysis workflow for ReaxKit.

This workflow provides tools for analyzing parameter sensitivities from
ReaxFF `fort.79` files, which report how force-field parameters influence
training or objective-function errors.

It supports:
- Identifying the most sensitive (or least sensitive) force-field parameter
  based on minimum and maximum sensitivity metrics.
- Visualizing sensitivity evolution across epoch sets for a selected parameter.
- Generating tornado plots that summarize parameter influence ranges
  (min, median, and max effects) across the force field.
- Exporting processed sensitivity tables and subsets to CSV for downstream
  analysis or reporting.

The workflow is intended for diagnosing force-field training behavior,
prioritizing parameters for refinement, and interpreting sensitivity-driven
optimization results.
"""


import argparse
from pathlib import Path

from reaxkit.analysis.force_field import (
    ParameterOptimizationDiagnosticRequest,
    ParameterOptimizationDiagnosticTask,
)
from reaxkit.cli.path import resolve_output_path
from reaxkit.domain.data_models import ForceFieldOptimizationDiagnosticData
from reaxkit.engine.reaxff.adapter import ReaxFFAdapter
from reaxkit.presentation.plot import single_plot, tornado_plot


def _load_fort79_table(file_path: str):
    return ParameterOptimizationDiagnosticTask().run(
        ReaxFFAdapter().load(
            ForceFieldOptimizationDiagnosticData,
            {
                "fort79": file_path,
                "input": file_path,
            },
        ),
        ParameterOptimizationDiagnosticRequest(),
    ).table


def _fort79_most_sensitive_param_task(args: argparse.Namespace) -> int:
    """
    Compute and visualize the parameter with the minimum sensitivity in a fort.79 file.

    This task:
    - Parses the fort.79 output using the analysis task
    - Identifies the parameter with the lowest `min_sensitivity`
    - Plots its sensitivity values across epoch-sets (optional: --plot / --save)
    - Exports the full sensitivity table and the minimum-identifier subset (optional: --export)

    Returns
    -------
    int
        Exit status code (0 on success).
    """
    df_sens = _load_fort79_table(args.file)

    idx_min = df_sens["min_sensitivity"].idxmin()
    min_identifier = df_sens.loc[idx_min, "identifier"]
    min_value = df_sens.loc[idx_min, "min_sensitivity"]
    print(f"[Done] Minimum sensitivity value: {min_value:.6f}")
    print(f"Corresponding identifier: {min_identifier}")

    df_sens = df_sens.copy()
    df_sens["epoch-set"] = df_sens.index + 1

    ratio_cols = ["sensitivity1/3", "sensitivity2/3", "sensitivity4/3"]
    subset_min = df_sens[df_sens["identifier"] == min_identifier].copy()
    long_all = (
        subset_min[["epoch-set"] + ratio_cols]
        .melt(
            id_vars=["epoch-set"],
            value_vars=ratio_cols,
            var_name="sensitivity_name",
            value_name="sensitivity",
        )
        .dropna(subset=["sensitivity"])
    )

    series = [{
        "x": long_all["epoch-set"].to_numpy(),
        "y": long_all["sensitivity"].to_numpy(),
        "label": "sensitivities per epoch",
        "marker": "o",
        "linewidth": 0,
        "markersize": 20,
        "alpha": 1,
    }]

    title_for_save = "Ratios_per_EpochSet"
    workflow_name = args.kind

    if args.save:
        out_save = Path(resolve_output_path(args.save, workflow_name))
        out_save.parent.mkdir(parents=True, exist_ok=True)

        if out_save.suffix:
            title_for_save = out_save.stem

        single_plot(
            series=series,
            plot_type="scatter",
            title=title_for_save,
            xlabel="Epoch Set",
            ylabel="sensitivity (Error Response)",
            save=str(out_save),
            legend=True,
        )
    elif args.plot:
        single_plot(
            series=series,
            plot_type="scatter",
            title=title_for_save,
            xlabel="Epoch Set",
            ylabel="sensitivity (Error Response)",
            save=None,
            legend=True,
        )

    if args.export:
        export_base = Path(resolve_output_path(args.export, workflow_name))
        export_base.parent.mkdir(parents=True, exist_ok=True)

        base_no_suffix = export_base.with_suffix("")
        out_min = base_no_suffix.with_name(base_no_suffix.name + "_min").with_suffix(".csv")
        out_all = base_no_suffix.with_name(base_no_suffix.name + "_all").with_suffix(".csv")

        export_cols = ["epoch-set", "identifier"]
        export_cols += [c for c in ratio_cols if c in df_sens.columns]
        export_cols += [c for c in ["min_sensitivity", "max_sensitivity"] if c in df_sens.columns]

        subset_min = df_sens[df_sens["identifier"] == min_identifier].copy()
        subset_min[export_cols].to_csv(out_min, index=False)

        to_all = df_sens.copy()
        to_all[export_cols].to_csv(out_all, index=False)

        print(f"[Done] Exported min-identifier data to {out_min}")
        print(f"[Done] Exported all-identifier data to {out_all}")

    if not args.plot and not args.save and not args.export:
        print("No action selected. Use one or more of --plot, --save, --export.")
    return 0


def _fort79_tornado_task(args: argparse.Namespace) -> int:
    """
    Generate a tornado sensitivity plot from a fort.79 file.

    This task:
    - Parses sensitivity values from `fort.79`
    - Builds a tornado-style bar plot showing positive and negative effects
    - Allows saving or displaying the plot (via --plot / --save)
    - Supports exporting the processed tornado table (via --export)

    Returns
    -------
    int
        Exit status code (0 on success).
    """
    sensitivities = _load_fort79_table(args.file)

    grouped = (
        sensitivities.groupby("identifier", dropna=True)
        .agg(min_eff=("min_sensitivity", "min"), max_eff=("max_sensitivity", "max"))
        .reset_index()
    )

    eff_union = (
        sensitivities.melt(
            id_vars=["identifier"],
            value_vars=["min_sensitivity", "max_sensitivity"],
            var_name="kind",
            value_name="eff",
        )
        .dropna(subset=["eff"])
    )
    median_series = eff_union.groupby("identifier", dropna=True)["eff"].median()
    grouped["median_eff"] = grouped["identifier"].map(median_series)

    grouped["span"] = grouped["max_eff"] - grouped["min_eff"]
    grouped = grouped.sort_values("span", ascending=False)

    if grouped.empty:
        print("No data to plot (empty after grouping).")
        return 0

    grouped_top = grouped.head(args.top) if (args.top and args.top > 0) else grouped.copy()

    workflow_name = args.kind
    if args.export:
        out = resolve_output_path(args.export, workflow_name)
        export_cols = ["identifier", "min_eff", "median_eff", "max_eff", "span"]
        grouped_top[export_cols].to_csv(out, index=False)
        print(f"[Done] Exported tornado data to {out}")

    if args.save:
        out = resolve_output_path(args.save, workflow_name)
        tornado_plot(
            labels=grouped_top["identifier"].tolist(),
            min_vals=grouped_top["min_eff"].tolist(),
            max_vals=grouped_top["max_eff"].tolist(),
            median_vals=grouped_top["median_eff"].tolist(),
            title="Parameter Sensitivity Tornado Plot",
            xlabel="sensitivity (Error Response)",
            ylabel="Parameter",
            save=out,
            top=0,
            vline=1.0 if args.vline else None,
        )

    if args.plot:
        tornado_plot(
            labels=grouped_top["identifier"].tolist(),
            min_vals=grouped_top["min_eff"].tolist(),
            max_vals=grouped_top["max_eff"].tolist(),
            median_vals=grouped_top["median_eff"].tolist(),
            title="Parameter Sensitivity Tornado Plot",
            xlabel="sensitivity (Error Response)",
            ylabel="Parameter",
            save=None,
            top=0,
            vline=1.0 if args.vline else None,
        )

    if not args.plot and not args.save and not args.export:
        print("No action selected. Use one or more of --plot, --save, --export.")

    return 0


def _add_common_fort79_io_args(
    p: argparse.ArgumentParser,
    *,
    include_plot: bool = True,
) -> None:
    """Add shared CLI arguments to the provided parser."""
    p.add_argument("--file", default="fort.79", help="Path to fort.79 file.")

    if include_plot:
        p.add_argument("--plot", action="store_true", help="Show plot interactively.")

    p.add_argument(
        "--save",
        default=None,
        help="Path or directory to save the plot image (resolved via resolve_output_path).",
    )
    p.add_argument(
        "--export",
        default=None,
        help="Export processed data to CSV (path or directory, resolved via resolve_output_path).",
    )


def register_tasks(subparsers: argparse._SubParsersAction) -> None:
    """Register fort.79 workflow tasks."""
    most_sensitive = subparsers.add_parser(
        "most-sensitive",
        help="Identify the parameter with the minimum sensitivity and optionally plot or export results.",
        description=(
            "Examples:\n"
            "  reaxkit fort79 most-sensitive --plot\n"
            "  reaxkit fort79 most-sensitive --export result.csv\n"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )

    _add_common_fort79_io_args(most_sensitive)
    most_sensitive.set_defaults(_run=_fort79_most_sensitive_param_task)

    p_tornado = subparsers.add_parser(
        "tornado",
        help="Create a tornado plot of sensitivities from fort.79 and optionally save or export the data.",
        description=(
            "Examples:\n"
            "  reaxkit fort79 tornado --top 6 --save tornado.png --export tornado.csv\n"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )

    _add_common_fort79_io_args(p_tornado)

    p_tornado.add_argument(
        "--top",
        type=int,
        default=0,
        help="Only show/export top-N widest spans (0 = show all).",
    )
    p_tornado.add_argument(
        "--vline",
        type=float,
        default=1.0,
        help="Draw a vertical reference line at x = this value.",
    )

    p_tornado.set_defaults(_run=_fort79_tornado_task)
