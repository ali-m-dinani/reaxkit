"""used to read fort.79 data, find the most sensitive parameter in the ffield and plot tornado plots of sensitivities"""
import argparse
from pathlib import Path
import pandas as pd
from reaxkit.io.fort79_handler import Fort79Handler
from reaxkit.analysis.fort79_analyzer import diff_sensitivities
from reaxkit.analysis.plotter import single_plot, tornado_plot


def _fort79_most_sensitive_param_task(args: argparse.Namespace) -> int:
    """
    (unchanged)
    """
    handler = Fort79Handler(args.file)
    df_sens = diff_sensitivities(handler)

    idx_min = df_sens["min_sensitivity"].idxmin()
    min_identifier = df_sens.loc[idx_min, "identider"]
    min_value = df_sens.loc[idx_min, "min_sensitivity"]
    print(f"[Done] Minimum sensitivity value: {min_value:.6f}")
    print(f"📘 Corresponding identifier: {min_identifier}")

    df_sens = df_sens.copy()
    df_sens["epoch-set"] = df_sens.index + 1

    ratio_cols = ["sensitivity1/3", "sensitivity2/3", "sensitivity4/3"]
    subset_min = df_sens[df_sens["identider"] == min_identifier].copy()
    long_all = (
        subset_min[["epoch-set"] + ratio_cols]
        .melt(id_vars=["epoch-set"], value_vars=ratio_cols,
              var_name="sensitivity_name", value_name="sensitivity")
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
    if args.save:
        p = Path(args.save)
        if p.suffix:
            title_for_save = p.stem

    if args.save:
        single_plot(
            series=series,
            plot_type="scatter",
            title=title_for_save,
            xlabel="Epoch Set",
            ylabel="sensitivity (Error Response)",
            save=args.save,
            legend=True,
        )
        if args.plot:
            single_plot(
                series=series,
                plot_type="scatter",
                title=title_for_save,
                xlabel="Epoch Set",
                ylabel="sensitivity (Error Response)",
                save=None,
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
        export_path = Path(args.export)
        base = export_path.with_suffix("")
        out_min = base.parent / f"{base.name}_min.csv"
        out_all = base.parent / f"{base.name}_all.csv"

        export_cols = ["epoch-set", "identider", "ratio1/3", "ratio2/3", "ratio4/3", "min_ratio", "max_ratio"]

        subset_min = df_sens[df_sens["identider"] == min_identifier].copy()
        for c in export_cols:
            if c not in subset_min.columns:
                subset_min[c] = pd.NA
        subset_min[export_cols].to_csv(out_min, index=False)

        to_all = df_sens.copy()
        for c in export_cols:
            if c not in to_all.columns:
                to_all[c] = pd.NA
        to_all[export_cols].to_csv(out_all, index=False)

        print(f"[Done] Exported min-identifier data → {out_min}")
        print(f"[Done] Exported all-identifier data → {out_all}")

    if not args.plot and not args.save and not args.export:
        print("ℹ️ No action selected. Use one or more of --plot, --save, --export.")
    return 0


def _fort79_tornado_task(args: argparse.Namespace) -> int:
    """
    Build a tornado plot of identifier effects using:
      - min_eff  = min(min_sensitivity) per identider
      - max_eff  = max(max_sensitivity) per identider
      - median_eff = median over the union of all {min_sensitivity, max_sensitivity} values per identider

    Usage full_sim_examples:
      reaxkit fort79 tornado --file fort.79 --save tornado.png
      reaxkit fort79 tornado --file fort.79 --top 25 --export tornado.csv --plot
    """
    handler = Fort79Handler(args.file)
    sensitivities = diff_sensitivities(handler)  # has: identider, min_sensitivity, max_sensitivity, ...

    # Aggregate per identifier: min & max from their respective columns
    grouped = (
        sensitivities.groupby("identider", dropna=True)
        .agg(min_eff=("min_sensitivity", "min"),
             max_eff=("max_sensitivity", "max"))
        .reset_index()
    )

    # Median across ALL min & max values together (union) per parameter
    eff_union = (
        sensitivities.melt(
            id_vars=["identider"],
            value_vars=["min_sensitivity", "max_sensitivity"],
            var_name="kind",
            value_name="eff"
        )
        .dropna(subset=["eff"])
    )
    median_series = eff_union.groupby("identider", dropna=True)["eff"].median()
    grouped["median_eff"] = grouped["identider"].map(median_series)

    # Sort by span just like before
    grouped["span"] = grouped["max_eff"] - grouped["min_eff"]
    grouped = grouped.sort_values("span", ascending=False)

    if grouped.empty:
        print("No data to plot (empty after grouping).")
        return 0

    # Apply top-N before exporting/plotting
    grouped_top = grouped.head(args.top) if (args.top and args.top > 0) else grouped.copy()

    # Export (now includes median)
    if args.export:
        export_cols = ["identider", "min_eff", "median_eff", "max_eff", "span"]
        grouped_top[export_cols].to_csv(args.export, index=False)
        print(f"[Done] Exported tornado data to {args.export}")

    # Save figure if requested
    if args.save:
        tornado_plot(
            labels=grouped_top["identider"].tolist(),
            min_vals=grouped_top["min_eff"].tolist(),
            max_vals=grouped_top["max_eff"].tolist(),
            median_vals=grouped_top["median_eff"].tolist(),  # NEW
            title="Parameter Sensitivity Tornado Plot",
            xlabel="sensitivity (Error Response)",
            ylabel="Parameter",
            save=args.save,
            top=0,
            vline=1.0 if args.vline else None,
        )

    # Interactive plot if requested
    if args.plot:
        tornado_plot(
            labels=grouped_top["identider"].tolist(),
            min_vals=grouped_top["min_eff"].tolist(),
            max_vals=grouped_top["max_eff"].tolist(),
            median_vals=grouped_top["median_eff"].tolist(),  # NEW
            title="Parameter Sensitivity Tornado Plot",
            xlabel="sensitivity (Error Response)",
            ylabel="Parameter",
            save=None,
            top=0,
            vline=1.0 if args.vline else None,
        )

    if not args.plot and not args.save and not args.export:
        print("ℹ️ No action selected. Use one or more of --plot, --save, --export.")

    return 0


def register_tasks(subparsers: argparse._SubParsersAction) -> None:
    # --- tornado ---
    p_tornado = subparsers.add_parser("tornado", help="Tornado Plot of parameter effects.\n"
                                                      "reaxkit fort79 tornado --save fort79/tornado.png \n"
                                                      "reaxkit fort79 tornado --top 25 --export fort79/tornado.csv")
    p_tornado.add_argument("--file", default="fort.79", help="Path to fort.79 file")
    p_tornado.add_argument("--plot", action="store_true", help="Show plot interactively")
    p_tornado.add_argument("--save", default=None, help="Save figure (file path or directory)")
    p_tornado.add_argument("--export", default=None, help="Export tornado table to CSV")
    p_tornado.add_argument("--top", type=int, default=0, help="Only show/export top-N widest spans (0 = all)")
    p_tornado.add_argument("--vline", default=1, help="Draw vertical reference line at x=1.0")
    p_tornado.set_defaults(_run=_fort79_tornado_task)

    # --- most_sensitive_param ---
    most_sensitive = subparsers.add_parser("most-sensitive", help="Plot and export parameter effects for best and all identifiers\n"
                                                                  "reaxkit fort79 most-sensitive --plot")
    most_sensitive.add_argument("--file", default="fort.79", help="Path to fort.79 file")
    most_sensitive.add_argument("--plot", action="store_true", help="Show plot interactively")
    most_sensitive.add_argument("--save", default=None,
                            help="Full output image path or directory (e.g., out/ratios_plot.png)")
    most_sensitive.add_argument("--export", default=None, help="Path to export data as CSV (optional)")
    most_sensitive.set_defaults(_run=_fort79_most_sensitive_param_task)
