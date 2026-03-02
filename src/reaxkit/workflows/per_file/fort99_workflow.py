"""fort.99 optimization-report workflow for ReaxKit."""

from __future__ import annotations

import argparse
from pathlib import Path

from reaxkit.analysis.force_field import (
    ForceFieldOptimizationReportBulkModulusRequest,
    ForceFieldOptimizationReportBulkModulusTask,
    ForceFieldOptimizationReportEOSRequest,
    ForceFieldOptimizationReportEOSTask,
    ForceFieldOptimizationReportRequest,
    ForceFieldOptimizationReportTask,
)
from reaxkit.cli.path import resolve_output_path
from reaxkit.core.alias import normalize_choice
from reaxkit.domain.data_models import ForceFieldOptimizationReportData, GeometrySummaryData
from reaxkit.engine.reaxff.adapter import ReaxFFAdapter
from reaxkit.presentation.plot import single_plot


def _load_report(path: str) -> ForceFieldOptimizationReportData:
    return ReaxFFAdapter().load(
        ForceFieldOptimizationReportData,
        {"fort99": path, "input": path},
    )


def _load_geometry_summary(fort99_path: str, fort74_path: str | None) -> GeometrySummaryData:
    chosen = fort74_path or str(Path(fort99_path).with_name("fort.74"))
    return ReaxFFAdapter().load(
        GeometrySummaryData,
        {"fort74": chosen, "input": chosen},
    )


def _fort99_get_task(args: argparse.Namespace) -> int:
    data = _load_report(args.file)
    sortby = normalize_choice(args.sort, domain="sort")
    df = ForceFieldOptimizationReportTask().run(
        data,
        ForceFieldOptimizationReportRequest(
            sortby=sortby,
            ascending=bool(args.ascending),
        ),
    ).table

    if df.empty:
        print("[Warning] fort.99 result is empty.")
        return 0

    workflow_name = args.kind
    if args.export:
        out = resolve_output_path(args.export, workflow_name)
        df.to_csv(out, index=False)
        print(f"[Done] Exported fort.99 table to {out}")
    else:
        print(df.to_string(index=False))
    return 0


def _fort99_eos_task(args: argparse.Namespace) -> int:
    report = _load_report(args.fort99)
    geometry_summary = _load_geometry_summary(args.fort99, args.fort74)
    df = ForceFieldOptimizationReportEOSTask().run(
        report,
        ForceFieldOptimizationReportEOSRequest(
            geometry_summary=geometry_summary,
            iden=args.iden,
        ),
    ).table

    if df.empty:
        print("[Warning] No ENERGY vs volume data found.")
        return 0

    workflow_name = args.kind
    if args.export:
        out = resolve_output_path(args.export, workflow_name)
        df.to_csv(out, index=False)
        print(f"[Done] Exported ENERGY vs volume table to {out}")

    do_plot = args.plot or args.save
    if not do_plot:
        return 0

    save_dir = resolve_output_path(args.save, workflow_name) if args.save else None
    for iden1, group in df.groupby("iden1"):
        group = group.sort_values("V_iden2").copy()
        if args.flip:
            group["ffield_value"] = -group["ffield_value"]
            group["qm_value"] = -group["qm_value"]

        single_plot(
            series=[
                {"x": group["V_iden2"], "y": group["ffield_value"], "label": "Force-field"},
                {"x": group["V_iden2"], "y": group["qm_value"], "label": "QM"},
            ],
            title="",
            xlabel="Volume",
            ylabel="Relative Energy (kcal/mole)",
            save=save_dir,
            legend=True,
            figsize=(6, 4),
        )
    return 0


def _task_bulk_modulus(args: argparse.Namespace) -> int:
    report = _load_report(args.fort99)
    geometry_summary = _load_geometry_summary(args.fort99, args.fort74)
    res = ForceFieldOptimizationReportBulkModulusTask().run(
        report,
        ForceFieldOptimizationReportBulkModulusRequest(
            geometry_summary=geometry_summary,
            iden=args.iden,
            source=args.source,
        ),
    ).values

    print("\nBulk modulus (Vinet EOS fit)")
    print("-" * 40)
    print(f"Identifier : {res['iden']}")
    print(f"Source     : {res['source']}")
    print(f"Points     : {res['n_points']}")
    print(f"V0         : {res['V0_A3']:.4f} A^3")
    print(f"K0         : {res['K0_GPa']:.2f} GPa")
    print(f"C          : {res['C']:.3f}")
    return 0


def register_tasks(subparsers: argparse._SubParsersAction) -> None:
    p_get = subparsers.add_parser(
        "get",
        help="Compute fort.99 errors and sort/export the table.\n",
        description=(
            "Examples:\n"
            "  reaxkit fort99 get --sort error --ascending --export fort99_sorted_data.csv\n"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p_get.add_argument("--file", default="fort.99", help="Path to fort.99 file")
    p_get.add_argument("--sort", default="error", help="Column to sort by (e.g., error, ffield_value, qm_value)")
    p_get.add_argument("--ascending", action="store_true", help="Sort ascending (default: descending)")
    p_get.add_argument("--export", default=None, help="CSV file to export the sorted table")
    p_get.set_defaults(_run=_fort99_get_task)

    p_eos = subparsers.add_parser(
        "eos",
        help="Energy vs volume (EOS) from fort.99 + fort.74.\n",
        description=(
            "Examples:\n"
            "  reaxkit fort99 eos --iden all --save reaxkit_outputs/fort99/eos_plots/ --flip --export eos.csv\n"
            "  reaxkit fort99 eos --iden bulk_0 --save eos_bulk_0.png --export eos_bulk0.csv\n"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p_eos.add_argument("--fort99", default="fort.99", help="Path to fort.99 file")
    p_eos.add_argument("--fort74", "--fort74.md", dest="fort74", default=None, help="Path to fort.74 file")
    p_eos.add_argument("--iden", default="all", help="iden1 to include ('all' or a specific value)")
    p_eos.add_argument("--plot", action="store_true", help="Show plots interactively")
    p_eos.add_argument("--save", default=None, help="Directory or file path for saved plots")
    p_eos.add_argument("--export", default=None, help="CSV file to export ENERGY vs volume data")
    p_eos.add_argument("--flip", action="store_true", help="Flip the sign of both QM and force-field energies")
    p_eos.set_defaults(_run=_fort99_eos_task)

    bulk = subparsers.add_parser(
        "bulk",
        help="Compute bulk modulus from ENERGY vs volume (Vinet EOS).",
        description=(
            "Examples:\n"
            "  reaxkit fort99 bulk --iden Al2N2_w_opt2\n"
            "  reaxkit fort99 bulk --iden Al2N2_w_opt2 --source qm\n"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    bulk.add_argument("--fort99", default="fort.99", help="fort.99 file to use")
    bulk.add_argument("--fort74", "--fort74.md", dest="fort74", default="fort.74", help="fort.74 file to use")
    bulk.add_argument("--iden", required=True, help="ENERGY identifier (iden1) to use for EOS fitting")
    bulk.add_argument("--source", default="ffield", choices=["ffield", "qm"], help="Energy source to use")
    bulk.set_defaults(_run=_task_bulk_modulus)
