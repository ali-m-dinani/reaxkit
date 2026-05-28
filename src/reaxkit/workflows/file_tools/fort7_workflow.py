"""Direct command workflow for repairing corrupted fort.7 files.

This module implements CLI workflow orchestration for its command family, including argument parsing, request construction, execution dispatch, and result presentation handoff.

**Usage context**

- Command routing: Resolve CLI aliases and normalized command names.
- Task execution: Build request objects and invoke registered tasks.
- Output handling: Forward results to table, plot, export, or report flows.
"""

from __future__ import annotations

import argparse

from reaxkit.core.generator_runtime import (
    maybe_copy_output_to_dot,
    persist_generator_metadata,
    prepare_generator_output,
    print_saved_dirs,
)
from reaxkit.core.storage_layout import add_storage_cli_arguments
from reaxkit.engine.reaxff.generators.fort7_repair import repair_fort7

ALL_COMMANDS = ("repair_fort7",)
ALL_LEGACY_COMMANDS = ()


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
    parser.set_defaults(command="repair_fort7")
    parser.formatter_class = argparse.RawTextHelpFormatter
    parser.description = (
        "Repair corrupted `fort.7` atom lines where fused integer columns break tokenization.\n"
        "This command rewrites a repaired output file while preserving lines that do not need\n"
        "changes. It is useful when malformed spacing/column fusion causes downstream parsing\n"
        "failures, which happen in +9999-atom-simulations using Standalone ReaxFF, in trajectory or "
        "analysis workflows.\n\n"
        "Examples:\n"
        "  1. Repair a specific input file and write to a named output:\n"
        "   reaxkit repair_fort7 --file fort.7 --output fort7_fixed\n\n"
        "  2. Repair using custom output and also copy result to current directory:\n"
        "   reaxkit repair_fort7 --output fort7_repaired --copy-to-dot"
    )
    parser.add_argument(
        "--file",
        default="fort.7",
        help="Input fort.7 file. Example: --file runs/job1/fort.7, which reads that file as repair source.",
    )
    parser.add_argument(
        "--output",
        default="fort7_fixed",
        help="Output repaired fort.7 file. Example: --output fort7_repaired, which writes repaired content using that output name.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=5000,
        help="Progress interval in lines (stored for metadata compatibility). Example: --progress-every 10000, which records progress in larger line-step chunks.",
    )
    parser.add_argument(
        "--copy-to-dot",
        action="store_true",
        help="Also copy generated output to current directory. Example: --copy-to-dot, which keeps a convenience copy where you run the command.",
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
    stats = repair_fort7(input_file=args.file, output_file=out_path, progress_every=int(args.progress_every))
    persist_generator_metadata(
        args,
        command=command,
        output_path=out_path,
        layout=layout,
        extra={"repair_stats": stats},
        copy_to_dot=bool(getattr(args, "copy_to_dot", False)),
    )
    copied = maybe_copy_output_to_dot(out_path, enabled=bool(getattr(args, "copy_to_dot", False)))
    dirs = [out_path.parent]
    if copied is not None:
        dirs.append(copied.parent)
    print_saved_dirs(dirs)
    print(
        "Repaired fort.7:"
        f" lines={stats['lines']:,}"
        f" frames={stats['frames']:,}"
        f" fixed={stats['fixed']:,}"
        f" unchanged={stats['unchanged']:,}"
        f" unresolved={stats['unresolved']:,}"
        f" skipped={stats['skipped']:,}"
    )
    return 0
