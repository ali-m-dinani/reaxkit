"""Direct command workflow for generating ``charges`` files."""

from __future__ import annotations

import argparse

from reaxkit.core.generator_runtime import (
    maybe_copy_output_to_dot,
    persist_generator_metadata,
    prepare_generator_output,
    print_saved_dirs,
)
from reaxkit.core.storage_layout import add_storage_cli_arguments
from reaxkit.engine.reaxff.generators.charges_generator import gen_template_charges


def build_parser(parser: argparse.ArgumentParser, *, command: str) -> argparse.ArgumentParser:
    _ = command
    parser.set_defaults(command="gen_template_charges")
    parser.formatter_class = argparse.RawTextHelpFormatter
    parser.description = (
        "Generate a ReaxFF charges template file.\n\n"
        "Examples:\n"
        "  reaxkit gen_template_charges\n"
        "  reaxkit gen_template_charges --output charges\n"
        "  reaxkit gen_template_charges --output charges.template --copy-to-dot"
    )
    parser.add_argument(
        "--output",
        default="charges",
        help="Output filename to write under <project_root>/input/",
    )
    parser.add_argument(
        "--copy-to-dot",
        action="store_true",
        help="Also copy generated output to the current directory.",
    )
    add_storage_cli_arguments(parser)
    return parser


def run_main(command: str, args: argparse.Namespace) -> int:
    out_path, layout = prepare_generator_output(args, command=command, output_value=str(args.output))
    written = gen_template_charges(out_path)
    persist_generator_metadata(
        args,
        command=command,
        output_path=written,
        layout=layout,
        copy_to_dot=bool(getattr(args, "copy_to_dot", False)),
    )
    copied = maybe_copy_output_to_dot(written, enabled=bool(getattr(args, "copy_to_dot", False)))
    dirs = [written.parent]
    if copied is not None:
        dirs.append(copied.parent)
    print_saved_dirs(dirs)
    return 0
