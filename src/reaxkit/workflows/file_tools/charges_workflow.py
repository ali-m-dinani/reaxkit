"""Direct command workflow for generating ``charges`` files.

This module implements CLI workflow orchestration for its command family, including argument parsing, request construction, execution dispatch, and result presentation handoff.

**Usage context**

- Command routing: Resolve CLI aliases and normalized command names.
- Task execution: Build request objects and invoke registered tasks.
- Output handling: Forward results to table, plot, export, or report flows.
"""

from __future__ import annotations

import argparse

from reaxkit.core.runtime.generator_runtime import (
    maybe_copy_output_to_dot,
    persist_generator_metadata,
    prepare_generator_output,
    print_saved_dirs,
)
from reaxkit.core.storage.storage_layout import add_storage_cli_arguments
from reaxkit.engine.reaxff.generators.charges_generator import gen_template_charges

ALL_COMMANDS = ("gen_template_charges",)
ALL_LEGACY_COMMANDS = ("make-charges", "make_charges")


def build_parser(parser: argparse.ArgumentParser, *, command: str) -> argparse.ArgumentParser:
    """Build parser.

    Execute the workflow function for this command path and return the
    computed result for downstream CLI handling.

    Parameters
    -----
    parser : Any
        Function argument.
    command : Any
        Function argument.

    Returns
    -----
    argparse.ArgumentParser
        Function return value.

    Examples
    -----
    >>> # See workflow CLI usage for concrete examples.
    """
    _ = command
    parser.set_defaults(command="gen_template_charges")
    parser.formatter_class = argparse.RawTextHelpFormatter
    parser.description = (
        "Write a template charges file for ReaxFF workflows.\n"
        "This command generates a starter charges file only. It does not run simulation steps\n"
        "or apply charges to an existing trajectory/run. Use the generated file as a base, then\n"
        "edit it according to your system and workflow requirements.\n\n"
        "Examples:\n"
        "  1. Generate a template using default output name ('charges'):\n"
        "   reaxkit gen_template_charges\n\n"
        "  2. Generate a template with a custom filename:\n"
        "   reaxkit gen_template_charges --output charges.template\n\n"
        "  3. Generate a template and also copy it to the current directory:\n"
        "   reaxkit gen_template_charges --output charges.template --copy-to-dot"
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
    """Run main.

    Execute the workflow function for this command path and return the
    computed result for downstream CLI handling.

    Parameters
    -----
    command : Any
        Function argument.
    args : Any
        Function argument.

    Returns
    -----
    int
        Function return value.

    Examples
    -----
    >>> # See workflow CLI usage for concrete examples.
    """
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
