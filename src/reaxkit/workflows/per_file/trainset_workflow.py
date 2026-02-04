"""
Trainset workflow for ReaxKit.

This workflow provides tools for inspecting, categorizing, generating, and
exporting ReaxFF trainset files used in force-field training and validation.

It supports:
- Reading an existing trainset file and exporting individual sections
  (e.g. charge, heat of formation, geometry, cell parameters, energy)
  as CSV tables for inspection or downstream analysis.
- Extracting and listing unique group comments (categories) defined in
  trainset sections, with optional sorting and CSV export.
- Generating a template trainset settings YAML file populated with
  default values for elastic and structural targets.
- Generating complete elastic-energy trainsets and associated tables
  from either:
    • a user-provided YAML settings file, or
    • Materials Project data via a material ID and API key.
- Optionally generating and post-processing strained geometry files
  associated with elastic trainset construction.

The workflow is designed to bridge high-level training specifications
(YAML, Materials Project data) with concrete ReaxFF trainset inputs in a
reproducible, CLI-driven manner.
"""


from __future__ import annotations

import os
import argparse
from pathlib import Path
from typing import Any, Dict

from reaxkit.io.handlers.trainset_handler import TrainsetHandler
from reaxkit.analysis.per_file.trainset_analyzer import get_trainset_group_comments
from reaxkit.utils.path import resolve_output_path
from reaxkit.io.generators.trainset_generator import (
    write_trainset_settings_yaml,
    generate_trainset_from_yaml,
    generate_trainset_settings_yaml_from_mp_simple,
)

# ----------------------------------------------------------------------
# Task 1: reaxkit trainset get --file ... --section ...
# ----------------------------------------------------------------------
def _get_task(args: argparse.Namespace) -> int:
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
def _category_task(args: argparse.Namespace) -> int:
    """
    Print or export unique group comments (categories) for trainset sections.
    """
    handler = TrainsetHandler(args.file)
    df = get_trainset_group_comments(handler, sort=args.sort)   # columns: section, group_comment

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
# Task 3: reaxkit trainset gen-settings --out ...
# ----------------------------------------------------------------------
def _gen_settings_task(args: argparse.Namespace) -> int:
    """
    Generate a sample trainset settings YAML using default values.

    Where the generated file is stored:
      - The YAML is written to: <resolved --out path> (typically under reaxkit_outputs/trainset/).
    """
    base_dir = Path("reaxkit_generated_inputs")
    base_dir.mkdir(parents=True, exist_ok=True)

    out_yaml = base_dir / args.out

    write_trainset_settings_yaml(out_path=str(out_yaml))

    print(f"[Done] Wrote sample settings YAML to: {out_yaml}")
    return 0

# ----------------------------------------------------------------------
# Task 4: reaxkit trainset generate --yaml ... OR --mp-id ... --api-key ...
# ----------------------------------------------------------------------
def _generate_task(args: argparse.Namespace) -> int:
    """
    Generate elastic-energy trainset + tables (and optional geo if geo.enable=true in YAML).

    Two modes:
      A) Use an existing YAML file: --yaml trainset_settings.yaml
      B) Build YAML from Materials Project: --mp-id mp-XXXX --api-key <KEY> [--bulk-mode vrh]

    Where the generated files are stored:
      - Elastic-energy trainset + tables are written to: <resolved --out-dir> (typically under reaxkit_outputs/trainset/).
      - Geo outputs (if geo.enable=true) are written under the YAML folder (trainset_generator writes geo to yaml_path.parent).
    """
    workflow_name = args.kind

    yaml_path = args.yaml

    # -------------------------
    # Mode B: build YAML from MP
    # -------------------------
    if not yaml_path:
        if not args.mp_id:
            print("❌ You must provide either --yaml <settings.yaml> OR --mp-id <mp-####>.")
            return 2

        api_key = args.api_key or os.getenv("MP_API_KEY")
        args.out_dir = f"{args.out_dir}_mp"

        if not api_key:
            print("❌ Missing Materials Project API key. Provide --api-key or set MP_API_KEY env var.")
            return 2

        # Where to write the generated YAML (and associated structure files)
        out_yaml = resolve_output_path(args.out_yaml, workflow_name)
        out_yaml_p = Path(out_yaml)
        out_yaml_p.parent.mkdir(parents=True, exist_ok=True)

        structure_dir = args.structure_dir or str(out_yaml_p.parent / "downloaded_structures")
        Path(structure_dir).mkdir(parents=True, exist_ok=True)

        res = generate_trainset_settings_yaml_from_mp_simple(
            mp_id=args.mp_id,
            out_yaml=out_yaml,
            structure_dir=structure_dir,
            bulk_mode=args.bulk_mode,
            api_key=api_key,
            verbose=bool(args.verbose),
        )

        yaml_path = res["yaml"]
        print(f"\n[Done] Generated settings from Materials Project:")
        print(f"       YAML: {res['yaml']}")
        print(f"       CIF:  {res['cif']}")
        print(f"       XYZ:  {res['xyz']}\n")

    else:
        args.out_dir = f"{args.out_dir}_yaml"

    # -------------------------
    # Run YAML -> trainset + tables (+ optional geo)
    # -------------------------

    out_dir = resolve_output_path(args.out_dir, workflow_name)
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    generate_trainset_from_yaml(yaml_path=yaml_path, out_dir=out_dir)

    print(f"[Done] Elastic-energy trainset + tables written to: {out_dir}")
    print(f"[Info] Geo outputs (if enabled in YAML) are written under the same folder in two separate sub-folders:\n"
          f" geo_strained and xyz_strained which contain .bgf and .xyz files, respectively.")

    # ------------------------------------------------------------------
    # Concatenate all strained geo (.bgf) files into one
    # ------------------------------------------------------------------
    geo_dir = Path(out_dir) / "geo_strained"
    all_geo_file = geo_dir / "all_trainset_geo.bgf"

    if geo_dir.exists():
        bgf_files = sorted(geo_dir.glob("*.bgf"))

        if bgf_files:
            with open(all_geo_file, "w") as fout:
                for bgf in bgf_files:
                    fout.write(f"# ===== BEGIN {bgf.name} =====\n")
                    with open(bgf, "r") as fin:
                        fout.write(fin.read())
                    fout.write(f"\n# ===== END {bgf.name} =====\n\n")

            print(
                f"[Post] All strained geometry (.bgf) files were concatenated into: all_trainset_geo.bgf"
            )
        else:
            print("[Post] geo_strained folder exists but contains no .bgf files.")
    else:
        print("[Post] No geo_strained folder found; skipping geometry concatenation.")

    return 0

# ----------------------------------------------------------------------
# Register tasks with the CLI
# ----------------------------------------------------------------------

def register_tasks(subparsers: argparse._SubParsersAction) -> None:
    # ---- get ---- (existing)
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
    p_get.set_defaults(_run=_get_task)

    # ---- category ---- (existing)
    p_cat = subparsers.add_parser(
        "category",
        help="List or export unique trainset categories (group comments) || ",
        description=(
            "Examples:\n"
            "  reaxkit trainset category --section all --export trainset_categories.csv\n"
            "  reaxkit trainset category --section all --sort\n"
            "  reaxkit trainset category --section energy --export energy_categories.csv\n"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p_cat.add_argument("--file", default="trainset.in", help="Path to trainset/fort.99 file")
    p_cat.add_argument("--section", default="all",
                       help="Section to analyze: all, charge, heatfo, geometry, cell_parameters, energy",
    )
    p_cat.add_argument("--export", help="Optional CSV file to write categories into (e.g. trainset_categories.csv)")
    p_cat.add_argument("--sort", action="store_true", help="Sort labels alphabetically (default: off)")
    p_cat.set_defaults(_run=_category_task)

    # ------------------------------------------------------------------
    # gen-settings
    # ------------------------------------------------------------------
    p_gens = subparsers.add_parser(
        "gen-settings",
        help="Generate a sample trainset settings YAML (default values).",
        description=(
            "Examples:\n"
            "  reaxkit trainset gen-settings\n"
            "  reaxkit trainset gen-settings --out reaxkit_outputs/trainset/trainset_settings.yaml\n"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p_gens.add_argument(
        "--out",
        default="trainset_settings.yaml",
        help="Output YAML filename/path (resolved under reaxkit_outputs/trainset/ if relative).",
    )
    p_gens.set_defaults(_run=_gen_settings_task)

    # ------------------------------------------------------------------
    # generate
    # ------------------------------------------------------------------
    p_gen = subparsers.add_parser(
        "generate",
        help="Generate elastic-energy trainset + tables (and optional geo) from YAML or Materials Project.",
        description=(
            "YAML mode:\n"
            "  reaxkit trainset generate --yaml trainset_settings.yaml\n"
            "\n"
            "Materials Project mode:\n"
            "  reaxkit trainset generate --mp-id mp-661 --api-key YOUR_KEY\n"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # Mode A
    p_gen.add_argument("--yaml", default=None, help="Path to an existing trainset_settings.yaml file.")

    # Mode B
    p_gen.add_argument("--mp-id", default=None, help="Materials Project material id (e.g., mp-661).")
    p_gen.add_argument("--api-key", default=None, help="Materials Project API key (or set MP_API_KEY env var).")
    p_gen.add_argument("--bulk-mode", default="voigt", choices=["voigt", "reuss", "vrh"],
                       help="Which MP bulk modulus to use (default: vrh).")
    p_gen.add_argument("--out-yaml", default="reaxkit_generated_inputs/trainset_mp/trainset_settings_mp.yaml",
                       help="Where to write the generated YAML in MP mode (resolved under outputs if relative).")
    p_gen.add_argument("--structure-dir", default=None,
                       help="Directory to write MP-downloaded structure files (default: next to out-yaml).")
    p_gen.add_argument("--verbose", action="store_true", help="Verbose MP fetching/logging.")

    # Common output
    p_gen.add_argument("--out-dir", default="reaxkit_generated_inputs/trainset",
                       help="Directory to write elastic-energy trainset + tables (resolved under outputs if relative).")

    p_gen.set_defaults(_run=_generate_task)