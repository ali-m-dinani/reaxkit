"""Direct command workflow for ReaxFF isomer representative detection."""

from __future__ import annotations

import argparse

from reaxkit.analysis.molecular_analysis.reaxff_isomer_representatives_detection import (
    detect_reaxff_isomer_representatives,
)
from reaxkit.core.resolve.command_alias_resolver import resolve_command_name

ALL_COMMANDS = ("detect-isomer-representatives",)
ALL_LEGACY_COMMANDS = ("detect_isomer_representatives", "detect-isomers", "detect_isomers")
COMMAND_ALIASES = {"detect-isomer-representatives": ALL_LEGACY_COMMANDS}


def build_parser(parser: argparse.ArgumentParser, *, command: str) -> argparse.ArgumentParser:
    """Build the parser for ReaxFF isomer representative detection."""
    canonical = resolve_command_name(command, task_names=ALL_COMMANDS, aliases=COMMAND_ALIASES)
    parser.set_defaults(command=canonical)
    parser.formatter_class = argparse.RawTextHelpFormatter
    parser.description = (
        "Detect coarse target-formula isomer representatives from ReaxFF `fort.7` and `xmolout` files.\n"
        "This uses fast bond-type count signatures, not full graph isomorphism, to keep the number of\n"
        "downstream Jaguar (DFT) jobs smaller and cheaper for ReaxFF training workflows.\n"
        "The target formula, atom type map, prefix, and legacy folder behavior are read from `control_params`.\n\n"
        "Examples:\n"
        "  1. Run in a ReaxFF output directory and write outputs to ./isomer_outputs:\n"
        "   reaxkit detect-isomer-representatives --fort7 fort.7 --xmolout xmolout --control control_params --output-dir isomer_outputs\n\n"
        "  2. Extract at most 10 representatives for cheaper Jaguar screening:\n"
        "   reaxkit detect-isomer-representatives --output-dir isomer_outputs --max-representatives 10"
    )
    parser.add_argument(
        "--fort7",
        default="fort.7",
        help="Input fort.7 file. Example: --fort7 runs/job1/fort.7.",
    )
    parser.add_argument(
        "--xmolout",
        default="xmolout",
        help="Input xmolout file. Example: --xmolout runs/job1/xmolout.",
    )
    parser.add_argument(
        "--control",
        default="control_params",
        help="Input control_params file. Example: --control runs/job1/control_params.",
    )
    parser.add_argument(
        "--output-dir",
        default="isomer_outputs",
        help="Output directory for xmolout_isomers, isomer_run_log.txt, and optional isomers folders.",
    )
    parser.add_argument(
        "--max-representatives",
        type=int,
        default=None,
        help="Optional cap on representatives to extract. Example: --max-representatives 10, which limits downstream Jaguar jobs.",
    )
    folder_group = parser.add_mutually_exclusive_group()
    folder_group.add_argument(
        "--write-isomer-dirs",
        action="store_true",
        help="Force writing per-isomer folders under output-dir/isomers.",
    )
    folder_group.add_argument(
        "--no-isomer-dirs",
        action="store_true",
        help="Disable per-isomer folders even when control_params has isomer_run=2.",
    )
    return parser


def run_main(command: str, args: argparse.Namespace) -> int:
    """Run ReaxFF isomer representative detection."""
    canonical = resolve_command_name(command, task_names=ALL_COMMANDS, aliases=COMMAND_ALIASES)
    write_isomer_dirs = None
    if bool(getattr(args, "write_isomer_dirs", False)):
        write_isomer_dirs = True
    if bool(getattr(args, "no_isomer_dirs", False)):
        write_isomer_dirs = False

    result = detect_reaxff_isomer_representatives(
        fort7_path=args.fort7,
        xmolout_path=args.xmolout,
        control_path=args.control,
        output_dir=args.output_dir,
        write_isomer_dirs=write_isomer_dirs,
        max_representatives=getattr(args, "max_representatives", None),
    )
    print(
        f"{canonical}: detected {len(result.records)} isomer representatives; "
        f"wrote {result.output_xmolout_isomers}"
    )
    if result.isomer_dir is not None:
        print(f"Per-isomer xmolout folders: {result.isomer_dir}")
    if result.log_path is not None:
        print(f"Log: {result.log_path}")
    return 0


__all__ = ["ALL_COMMANDS", "ALL_LEGACY_COMMANDS", "build_parser", "run_main"]
