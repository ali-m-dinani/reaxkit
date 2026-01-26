"""workflow for fort.99 files"""

from __future__ import annotations
import argparse
from pathlib import Path

from reaxkit.io.fort99_handler import Fort99Handler
from reaxkit.analysis import fort99_analyzer
from reaxkit.utils.plotter import single_plot
from reaxkit.utils.alias import normalize_choice
from reaxkit.io.fort74_handler import Fort74Handler
from reaxkit.utils.path import resolve_output_path

# ---------- tasks ----------

def fort99_get_task(args: argparse.Namespace) -> int:
    handler = Fort99Handler(args.file)
    sortby = normalize_choice(args.sort, domain="sort")

    df = fort99_analyzer.get_fort99(
        handler,
        sortby=sortby,
        ascending=args.ascending,
    )


    if df.empty:
        print("[Warning] fort.99 result is empty.")
        return 0

    workflow_name = args.kind
    if args.export:
        out = resolve_output_path(args.export, workflow_name)
        df.to_csv(out, index=False)
        print(f"[Done] Exported fort.99 table to {out}")
    else:
        # If no export requested, just print to stdout
        print(df.to_string(index=False))

    return 0


def fort99_eos_task(args: argparse.Namespace) -> int:
    """
     Steps:
      1) Run fort99_energy_vs_volume to obtain:
            iden1, iden2, ffield_value, qm_value, V_iden2
      2) Filter by --iden (either 'all' or a specific iden1)
      3) Optionally flip energy sign using --flip
      4) For each iden1 group, plot single_plot (both lines on same axes)
      5) Optionally export table
    """
    fort99_handler = Fort99Handler(args.fort99)

    # Guess fort.74 path if not provided
    fort74_path = (Path(args.fort99).with_name("fort.74")
                    if args.fort74 is None else Path(args.fort74))
    fort74_handler = Fort74Handler(str(fort74_path))

    df = fort99_analyzer.fort99_energy_vs_volume(
        fort99_handler=fort99_handler,
        fort74_handler=fort74_handler,
    )

    if df.empty:
        print("[Warning] No ENERGY vs volume data found.")
        return 0

    # Filter by iden1
    if args.iden.lower() != "all":
        df = df[df["iden1"] == args.iden]
        if df.empty:
            print(f"[Warning] No rows found for iden1 == {args.iden!r}.")
            return 0

    workflow_name = args.kind
    # Export table
    if args.export:
        out = resolve_output_path(args.export, workflow_name)
        df.to_csv(out, index=False)
        print(f"[Done] Exported ENERGY vs volume table to {out}")

    # No plot/save = nothing else to do
    do_plot = args.plot or args.save
    if not do_plot:
        return 0

    save_dir = None
    if args.save:
        save_dir = resolve_output_path(args.save, workflow_name)

    # ----------- PLOTTING -------------
    for iden1, g in df.groupby("iden1"):
        # Sort by volume for proper curve ordering
        g = g.sort_values("V_iden2")

        # Flip sign if requested
        if args.flip:
            g["ffield_value"] = -g["ffield_value"]
            g["qm_value"]     = -g["qm_value"]

        x = g["V_iden2"]
        y_ff = g["ffield_value"]
        y_qm = g["qm_value"]

        single_plot(
            series=[
                {"x": x, "y": y_ff, "label": "Force-field"},
                {"x": x, "y": y_qm, "label": "QM"},
            ],
            title="",
            xlabel="Volume",
            ylabel="Relative Energy (kcal/mole)",
            save=save_dir,
            legend=True,
            figsize=(6,4),
        )

    return 0

def task_bulk_modulus(args, fort99_handler, fort74_handler) -> None:
    """
    Compute and print the bulk modulus from ENERGY vs volume data
    using a Vinet equation-of-state fit.
    """
    from reaxkit.analysis.fort99_analyzer import fort99_bulk_modulus

    res = fort99_bulk_modulus(
        fort99_handler=fort99_handler,
        fort74_handler=fort74_handler,
        iden=args.iden,
        source=args.source,
    )

    print("\nBulk modulus (Vinet EOS fit)")
    print("-" * 40)
    print(f"Identifier : {res['iden']}")
    print(f"Source     : {res['source']}")
    print(f"Points     : {res['n_points']}")
    print(f"V0         : {res['V0_A3']:.4f} Å^3")
    print(f"K0         : {res['K0_GPa']:.2f} GPa")
    print(f"C          : {res['C']:.3f}")

# ---------- registration ----------

def register_tasks(subparsers: argparse._SubParsersAction) -> None:

    # ---- fort99 get ----
    p_get = subparsers.add_parser(
        "get",
        help="Compute fort.99 ENERGY errors and sort/export the table \n",
        description=(
            "Examples:\n"
            "  reaxkit fort99 get --sort error --ascending --export fort99_sorted_data.csv \n"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p_get.add_argument("--file", default="fort.99", help="Path to fort.99 file")
    p_get.add_argument("--sort", default="error", help="Column to sort by (e.g., error, ffield_value, qm_value)")
    p_get.add_argument("--ascending", action="store_true", help="Sort ascending (default: descending)")
    p_get.add_argument("--export", default=None, help="CSV file to export the sorted table (optional)")
    p_get.set_defaults(_run=fort99_get_task)

    # ---- fort99 eos ----
    p_eos = subparsers.add_parser(
        "eos", help="Energy vs volume (EOS) plots from fort.99 + fort.74 \n",
        description=(
            "Examples:\n"
            "  reaxkit fort99 eos --iden all --save reaxkit_outputs/fort99/eos_plots/ --flip --export eos_plots.csv \n"
            "  reaxkit fort99 eos --iden bulk_0 --save eos_bulk_0.png --export eos_bulk0.csv"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p_eos.add_argument("--fort99", default="fort.99", help="Path to fort.99 file")
    p_eos.add_argument("--fort74", default='fort.74', help="Path to fort.74 (default: same directory as fort.99)")
    p_eos.add_argument("--iden", default="all", help="iden1 to include ('all' or specific e.g. bulk_0)")
    p_eos.add_argument("--plot", action="store_true", help="Show plots interactively")
    p_eos.add_argument("--save", default=None, help="Directory to save plots as <iden1>.png")
    p_eos.add_argument("--export", default=None, help="CSV file to export ENERGY vs volume table")
    p_eos.add_argument("--flip", action="store_true",
                       help="Flip the sign of both QM and force-field energies before plotting")
    p_eos.set_defaults(_run=fort99_eos_task)

    bulk = subparsers.add_parser(
        "bulk",
        help="Compute bulk modulus from ENERGY vs volume (Vinet EOS)",
        description=(
            "Examples:\n"
            "  reaxkit fort99 bulk --iden Al2N2_w_opt2 \n"
            "  reaxkit fort99 bulk --iden Al2N2_w_opt2 --source qm\n"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )

    bulk.add_argument("--iden",required=True,help="ENERGY identifier (iden1) to use for EOS fitting")

    bulk.add_argument("--source", default="ffield", choices=["ffield", "qm"],
        help="Energy source to use (ffield or qm)",
    )

    bulk.set_defaults(func=task_bulk_modulus)
