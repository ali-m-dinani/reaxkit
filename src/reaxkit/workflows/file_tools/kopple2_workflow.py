"""Direct command workflow for generating ``kopple2`` files."""

from __future__ import annotations

import argparse

from reaxkit.core.generator_runtime import (
    maybe_copy_output_to_dot,
    persist_generator_metadata,
    prepare_generator_output,
    print_saved_dirs,
)
from reaxkit.core.storage_layout import add_storage_cli_arguments
from reaxkit.engine.reaxff.generators.kopple2_generator import gen_template_kopple2

ALL_COMMANDS = ("gen_template_kopple2",)
ALL_LEGACY_COMMANDS = ("make-kopple2", "make_kopple2")


def build_parser(parser: argparse.ArgumentParser, *, command: str) -> argparse.ArgumentParser:
    _ = command
    parser.set_defaults(command="gen_template_kopple2")
    parser.formatter_class = argparse.RawTextHelpFormatter
    parser.description = (
        "Write a template `kopple2` file for ReaxFF workflows.\n"
        "This command generates a starter `kopple2` file only. It does not run simulation steps\n"
        "or modify existing run outputs. Use the generated template as a base and adjust values\n"
        "for your target system.\n\n"
        "Examples:\n"
        "  1. Generate a template using default output name ('kopple2'):\n"
        "   reaxkit gen_template_kopple2\n\n"
        "  2. Generate a template with a custom filename:\n"
        "   reaxkit gen_template_kopple2 --output kopple2.template\n\n"
        "  3. Generate a template and also copy it to the current directory:\n"
        "   reaxkit gen_template_kopple2 --output kopple2.template --copy-to-dot"
    )
    parser.add_argument(
        "--output",
        default="kopple2",
        help="Output filename to write under <project_root>/input/. Example: --output kopple2.template, which writes the template using that filename.",
    )
    parser.add_argument(
        "--copy-to-dot",
        action="store_true",
        help="Also copy generated output to the current directory. Example: --copy-to-dot, which keeps a convenience copy where you run the command.",
    )
    add_storage_cli_arguments(parser)
    return parser


def run_main(command: str, args: argparse.Namespace) -> int:
    out_path, layout = prepare_generator_output(args, command=command, output_value=str(args.output))
    written = gen_template_kopple2(out_path)
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
