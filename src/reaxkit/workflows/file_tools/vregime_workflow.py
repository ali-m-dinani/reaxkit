"""Direct command workflow for vregime generation.

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
from reaxkit.engine.reaxff.generators.vregime_generator import gen_template_vregime

ALL_COMMANDS = ("gen_template_vregime",)
ALL_LEGACY_COMMANDS = ("make-vregime", "make_vregime")


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
