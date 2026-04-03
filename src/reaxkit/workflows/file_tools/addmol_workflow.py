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
from reaxkit.engine.reaxff.generators.addmol_generator import write_addmol


def build_parser(parser: argparse.ArgumentParser, *, command: str) -> argparse.ArgumentParser:
    _ = command
    parser.set_defaults(command="make-addmol")
    parser.formatter_class = argparse.RawTextHelpFormatter
    parser.description = (
        "Generate ReaxFF addmol templates (addmol.bgf and addmol.vel).\n\n"
        "Examples:\n"
        "  reaxkit make-addmol\n"
        "  reaxkit make-addmol --output addmol.bgf\n"
        "  reaxkit make-addmol --output custom_addmol.bgf --copy-to-dot"
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
    bgf_path = write_addmol(out_path)
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
