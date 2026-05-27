"""Direct command workflow for generating ``addmol`` files."""

from __future__ import annotations

import argparse
from pathlib import Path

from reaxkit.core.generator_runtime import (
    maybe_copy_output_to_dot,
    persist_generator_metadata,
    prepare_generator_output,
    print_saved_dirs,
)
from reaxkit.core.storage_layout import add_storage_cli_arguments
from reaxkit.engine.reaxff.generators.addmol_generator import gen_template_addmol

ALL_COMMANDS = ("gen_template_addmol",)
ALL_LEGACY_COMMANDS = ("make-addmol", "make_addmol")


def build_parser(parser: argparse.ArgumentParser, *, command: str) -> argparse.ArgumentParser:
    _ = command
    parser.set_defaults(command="gen_template_addmol")
    parser.formatter_class = argparse.RawTextHelpFormatter
    parser.description = (
        "Write template addmol files for ReaxFF workflows (addmol.bgf and addmol.vel).\n"
        "This command generates starter templates only. It does not run molecular dynamics or modify\n"
        "existing simulation data. Use the generated files as a clean starting point, then edit them\n"
        "based on your target system before running downstream workflows.\n\n"
        "Examples:\n"
        "  1. Generate templates using default output name ('addmol.bgf'):\n"
        "   reaxkit gen_template_addmol\n\n"
        "  2. Generate templates with a custom BGF filename:\n"
        "   reaxkit gen_template_addmol --output custom_addmol.bgf\n\n"
        "  3. Generate templates and also copy them to the current directory:\n"
        "   reaxkit gen_template_addmol --output custom_addmol.bgf --copy-to-dot"
    )
    parser.add_argument(
        "--output",
        default="addmol.bgf",
        help="Output addmol.bgf filename to write under <project_root>/input/",
    )
    parser.add_argument(
        "--copy-to-dot",
        action="store_true",
        help="Also copy generated outputs to the current directory.",
    )
    add_storage_cli_arguments(parser)
    return parser


def run_main(command: str, args: argparse.Namespace) -> int:
    out_path, layout = prepare_generator_output(args, command=command, output_value=str(args.output))
    bgf_path = gen_template_addmol(out_path)
    vel_path = Path(bgf_path).with_name("addmol.vel")

    persist_generator_metadata(
        args,
        command=command,
        output_path=bgf_path,
        layout=layout,
        extra={"secondary_output": {"name": vel_path.name, "path": str(vel_path)}},
        copy_to_dot=bool(getattr(args, "copy_to_dot", False)),
    )

    copied_bgf = maybe_copy_output_to_dot(bgf_path, enabled=bool(getattr(args, "copy_to_dot", False)))
    copied_vel = maybe_copy_output_to_dot(vel_path, enabled=bool(getattr(args, "copy_to_dot", False)))
    dirs = [bgf_path.parent]
    if copied_bgf is not None:
        dirs.append(copied_bgf.parent)
    if copied_vel is not None:
        dirs.append(copied_vel.parent)
    print_saved_dirs(dirs)
    return 0
