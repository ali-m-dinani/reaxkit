"""
fort.74 summary-table workflow for ReaxKit.

This workflow provides utilities for reading and exporting data from ReaxFF
`fort.74` files, which contain per-structure or per-configuration summary
quantities produced during force-field training or evaluation runs.

It supports:
- Exporting all available columns from a `fort.74` file to CSV.
- Selecting and exporting a single column (with optional alias resolution,
  e.g. `Density` → `D`) while always preserving the structure identifier.
- Writing outputs to a standardized ReaxKit output directory for
  reproducible data organization.

The workflow is designed for lightweight inspection and downstream analysis
of ReaxFF summary metrics such as energies, volumes, and densities.
"""

from __future__ import annotations

import argparse

from reaxkit.io.handlers.fort74_handler import Fort74Handler
from reaxkit.analysis.per_file.fort74_analyzer import get_fort74_data
from reaxkit.utils.path import resolve_output_path
from reaxkit.utils.alias import normalize_choice, resolve_alias_from_columns

def _get_task(args: argparse.Namespace) -> int:
    handler = Fort74Handler(args.file)
    df = get_fort74_data(handler)

    col_raw = (args.col or "all").strip()

    if col_raw != "all":
        canonical = normalize_choice(col_raw)  # e.g., "Density" -> "D"
        resolved = resolve_alias_from_columns(df.columns, canonical)
        if resolved is None:
            raise SystemExit(
                f"❌ Column '{col_raw}' not found (and no alias matched). "
                f"Available: {', '.join(df.columns)}"
            )

        cols = []
        if "identifier" in df.columns:
            cols.append("identifier")
        if resolved != "identifier":
            cols.append(resolved)

        out_df = df[cols].copy()

        # Optional: make exported header match what user asked for
        if resolved != col_raw:
            out_df = out_df.rename(columns={resolved: col_raw})
    else:
        out_df = df

    out_path = resolve_output_path(args.export, workflow="fort74.md")
    out_df.to_csv(out_path, index=False)
    print(f"[Done] Successfully exported the requested data to {out_path}")
    return 0




def register_tasks(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "get",
        help="Export one column or all columns from fort.74 to CSV.",
        description=(
            "Examples:\n"
            "  reaxkit fort74.md get --export fort74_all_data.csv\n"
            "  reaxkit fort74.md get --col Emin --export fort74_all_Emin_data.csv\n"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )

    p.add_argument("--file", default="fort.74", help="Path to fort.74 file.")
    p.add_argument("--col", default="all", help="Column to export, or 'all'.")
    p.add_argument(
        "--export",
        required=True,
        help=(
            "CSV output path. If a bare filename is given, it will be saved under "
            "reaxkit_outputs/fort74.md/ (via path.resolve_output_path)."
        ),
    )

    p.set_defaults(_run=_get_task)
