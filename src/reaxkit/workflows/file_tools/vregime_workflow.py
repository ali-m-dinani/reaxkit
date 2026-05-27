"""Direct command workflow for vregime generation."""

from __future__ import annotations

import argparse

from reaxkit.core.generator_runtime import (
    maybe_copy_output_to_dot,
    persist_generator_metadata,
    prepare_generator_output,
    print_saved_dirs,
)
from reaxkit.core.storage_layout import add_storage_cli_arguments
from reaxkit.engine.reaxff.generators.vregime_generator import gen_template_vregime

ALL_COMMANDS = ("gen_template_vregime",)
ALL_LEGACY_COMMANDS = ("make-vregime", "make_vregime")


def build_parser(parser: argparse.ArgumentParser, *, command: str) -> argparse.ArgumentParser:
    _ = command
    parser.set_defaults(command="gen_template_vregime")
    parser.formatter_class = argparse.RawTextHelpFormatter
    parser.description = (
        "Write a template `vregime.in` file for ReaxFF workflows.\n"
        "This command generates a starter vregime file with configurable sample-row count.\n"
        "It is intended as a baseline template and does not execute simulation steps.\n\n"
        "Examples:\n"
        "  1. Generate a template using defaults:\n"
        "   reaxkit gen_template_vregime\n\n"
        "  2. Generate a template with explicit row count and output name:\n"
        "   reaxkit gen_template_vregime --rows 5 --output vregime.in"
    )
    parser.add_argument(
        "--output",
        default="vregime.in",
        help="Output vregime path. Example: --output vregime_custom.in, which writes the generated template using that filename.",
    )
    parser.add_argument(
        "--copy-to-dot",
        action="store_true",
        help="Also copy generated output to current directory. Example: --copy-to-dot, which keeps a convenience copy where you run the command.",
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=5,
        help="Number of sample rows. Example: --rows 8, which generates eight template data rows.",
    )
    add_storage_cli_arguments(parser)
    return parser


def run_main(command: str, args: argparse.Namespace) -> int:
    out, layout = prepare_generator_output(args, command=command, output_value=str(args.output))
    gen_template_vregime(out, n_rows=args.rows)
    persist_generator_metadata(
        args,
        command=command,
        output_path=out,
        layout=layout,
        copy_to_dot=bool(getattr(args, "copy_to_dot", False)),
    )
    copied = maybe_copy_output_to_dot(out, enabled=bool(getattr(args, "copy_to_dot", False)))
    dirs = [out.parent]
    if copied is not None:
        dirs.append(copied.parent)
    print_saved_dirs(dirs)
    return 0
