"""Workflow for creating ReaxFF trainset files from completed isomer jobs."""

from __future__ import annotations

import argparse

from reaxkit.analysis.molecular_analysis.isomer_trainset import (
    create_isomer_trainset,
)
from reaxkit.core.resolve.command_alias_resolver import resolve_command_name

ALL_COMMANDS = ("create-isomer-trainset",)
ALL_LEGACY_COMMANDS = (
    "create_isomer_trainset",
    "make-isomer-trainset",
    "make_isomer_trainset",
)
COMMAND_ALIASES = {"create-isomer-trainset": ALL_LEGACY_COMMANDS}


def build_parser(parser: argparse.ArgumentParser, *, command: str) -> argparse.ArgumentParser:
    """Build the parser for Isomer trainset generation."""
    canonical = resolve_command_name(command, task_names=ALL_COMMANDS, aliases=COMMAND_ALIASES)
    parser.set_defaults(command=canonical)
    parser.formatter_class = argparse.RawTextHelpFormatter
    parser.description = (
        "Create ReaxFF geo/trainset files from completed isomer jobs.\n"
        "This workflow expects per-isomer job folders containing supported hf.out files.\n"
        "Currently supported output format: Jaguar hf.out.\n"
        "Incomplete or missing hf.out files are skipped by default and reported in the log.\n\n"
        "Examples:\n"
        "   reaxkit create-isomer-trainset --job-dir isomer_jobs --output-dir trainset_outputs\n"
        "   reaxkit create-isomer-trainset --job-dir isomer_jobs --output-dir trainset_outputs --require-all-complete"
    )
    parser.add_argument(
        "--job-dir",
        default="isomer_jobs",
        help="Directory containing per-isomer job folders. Example: --job-dir isomer_jobs, which reads the per-isomer calculation folders there.",
    )
    parser.add_argument(
        "--output-dir",
        default="isomer_trainset",
        help="Output directory for geo, trainset.in, composition.txt, and log files. Example: --output-dir analysis/charges, which writes generated artifacts beneath that directory.",
    )
    parser.add_argument("--hf-output-name", default="hf.out", help="Supported output filename inside each job folder. Example: --hf-output-name hf.out, which reads that filename in each isomer job folder.")
    parser.add_argument("--geo-file", default="geo", help="Output geo filename. Example: --geo-file geo, which writes generated geometries to geo.")
    parser.add_argument("--trainset-file", default="trainset.in", help="Output trainset filename. Example: --trainset-file trainset.in, which writes generated training entries to trainset.in.")
    parser.add_argument("--composition-file", default="composition.txt", help="Output composition filename. Example: --composition-file composition.txt, which writes relative compositions to that file.")
    parser.add_argument("--log-file", default="out_trainset_log.txt", help="Output log filename. Example: --log-file trainset.log, which writes generation diagnostics to that log.")
    parser.add_argument("--weight", type=float, default=1.0, help="Weight for generated ENERGY trainset lines. Example: --weight 1.0, which assigns unit weight to generated training entries.")
    parser.add_argument(
        "--reference-composition",
        type=float,
        default=100.0,
        help="Composition value assigned to the lowest-energy reference structure. Example: --reference-composition 100, which sets the lowest-energy reference composition to 100.",
    )
    parser.add_argument("--temperature", type=float, default=273.0, help="Temperature used for relative composition. Example: --temperature 300, which evaluates relative compositions at 300 K.")
    parser.add_argument("--gas-constant", type=float, default=1.987, help="Gas constant used for relative composition. Example: --gas-constant 1.987, which uses 1.987 cal/(mol K) in the composition calculation.")
    parser.add_argument(
        "--require-all-complete",
        action="store_true",
        help="Fail if any isomer job folder is missing a completed hf.out. Example: --require-all-complete, which fails if any expected isomer output is incomplete.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Allow overwriting files in an existing non-empty output directory. Example: --force, which allows replacement of existing generated output.",
    )
    return parser


def run_main(command: str, args: argparse.Namespace) -> int:
    """Run Isomer trainset generation."""
    canonical = resolve_command_name(command, task_names=ALL_COMMANDS, aliases=COMMAND_ALIASES)
    result = create_isomer_trainset(
        job_dir=args.job_dir,
        output_dir=args.output_dir,
        hf_output_name=args.hf_output_name,
        geo_filename=args.geo_file,
        trainset_filename=args.trainset_file,
        composition_filename=args.composition_file,
        log_filename=args.log_file,
        weight=float(args.weight),
        reference_composition=float(args.reference_composition),
        temperature=float(args.temperature),
        gas_constant=float(args.gas_constant),
        require_all_complete=bool(args.require_all_complete),
        force=bool(args.force),
    )
    print(
        f"{canonical}: processed {len(result.records) + len(result.skipped)} isomers"
        f"; included {len(result.records)}; skipped {len(result.skipped)}; "
        f"trainset {result.trainset_path}; geo {result.geo_path}"
    )
    return 0


__all__ = [
    "ALL_COMMANDS",
    "ALL_LEGACY_COMMANDS",
    "build_parser",
    "run_main",
]
