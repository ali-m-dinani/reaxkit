"""Workflow for ReaxFF trainset files."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

from reaxkit.io.trainset_handler import TrainsetHandler
from reaxkit.analysis.trainset_analyzer import trainset_group_comments
from reaxkit.utils.path import resolve_output_path

# ----------------------------------------------------------------------
# Task 1: reaxkit trainset get --file ... --section ...
# ----------------------------------------------------------------------
def get_task(args: argparse.Namespace) -> int:
    """
    Read trainset and save section DataFrames to CSV files.
    """
    handler = TrainsetHandler(args.file)
    meta: Dict[str, Any] = handler.metadata()
    tables: Dict[str, Any] = meta.get("tables", {})

    # -------------------------------------
    # Determine output directory
    # -------------------------------------
    if args.export:
        outdir = Path(args.export)
    else:
        outdir = Path("trainset_analysis")

    outdir.mkdir(parents=True, exist_ok=True)

    section = args.section.lower()

    if section == "all":
        items = list(tables.items())
    else:
        try:
            df = handler.section(section)
        except KeyError:
            print(f"[Error] Section '{section}' not found in trainset.")
            return 1

        canon_name = section.upper()
        if canon_name in ("CELL", "CELL PARAMETERS"):
            canon_name = "CELL_PARAMETERS"

        items = [(canon_name, df)]

    stem = Path(args.file).stem

    if not items:
        print("[Info] No sections found in trainset.")
        return 0

    for sec_name, df in items:
        if df is None or df.empty:
            print(f"[Skip] Section {sec_name} is empty or not parsed.")
            continue

        fname = f"{stem}_{sec_name.lower()}.csv"
        outpath = outdir / fname
        df.to_csv(outpath, index=False)
        print(f"[Done] Exported section '{sec_name}' to {outpath}")

    return 0


# ----------------------------------------------------------------------
# Task 2: reaxkit trainset category --file ... --section ...
# ----------------------------------------------------------------------
def category_task(args: argparse.Namespace) -> int:
    """
    Print or export unique group comments (categories) for trainset sections.
    """
    handler = TrainsetHandler(args.file)
    df = trainset_group_comments(handler, sort=args.sort)   # columns: section, group_comment

    if df.empty:
        print("[Info] No categories found in trainset.")
        return 0

    section = args.section.lower()

    if section != "all":
        df = df[df["section"].str.lower() == section]
        if df.empty:
            print(f"[Info] No categories found for section '{section}'.")
            return 0

    # ---------------------------------
    # EXPORT OPTION
    # ---------------------------------
    workflow_name = args.kind
    if args.export:
        outpath = resolve_output_path(args.export, workflow_name)
        df.to_csv(outpath, index=False)
        print(f"[Done] Exported categories to: {outpath}")
        return 0

    # ---------------------------------
    # PRINT OPTION
    # ---------------------------------
    for _, row in df.iterrows():
        print(f"{row['section']} {row['group_comment']}")

    return 0


# ----------------------------------------------------------------------
# Register tasks with the CLI
# ----------------------------------------------------------------------
def register_tasks(subparsers: argparse._SubParsersAction) -> None:

    # ---- get ----
    p_get = subparsers.add_parser(
        "get",
        help="Save trainset sections as CSV files. \n",
        description=(
            "Examples:\n"
            "  reaxkit trainset get --section all --export reaxkit_outputs/trainset\n"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p_get.add_argument("--file", default="trainset.in", help="Path to trainset/fort.99 file")
    p_get.add_argument("--section", default="all",
                       help="Section to export: all, charge, heatfo, geometry, cell_parameters, energy")
    p_get.add_argument("--export", help="Directory to save CSVs into (default: trainset_analysis/)")
    p_get.set_defaults(_run=get_task)

    # ---- category ----
    p_cat = subparsers.add_parser(
        "category",
        help="List or export unique trainset categories (group comments) || ",
        description=(
            "Examples:\n"
            "  reaxkit trainset category --section all --export trainset_categories.csv\n"
            "  reaxkit trainset category --section all --sort"
            "  reaxkit trainset category --section energy --export energy_categories.csv"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p_cat.add_argument("--file", default="trainset.in", help="Path to trainset/fort.99 file")
    p_cat.add_argument("--section", default="all",
                       help="Section to analyze: all, charge, heatfo, geometry, cell_parameters, energy",
    )
    p_cat.add_argument("--export", help="Optional CSV file to write categories into (e.g. trainset_categories.csv)")
    p_cat.add_argument("--sort", action="store_true", help="Sort labels alphabetically (default: off)")
    p_cat.set_defaults(_run=category_task)
