"""Direct command workflow for repairing corrupted fort.7 files."""

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


def build_parser(parser: argparse.ArgumentParser, *, command: str) -> argparse.ArgumentParser:
    _ = command
    parser.set_defaults(command="repair_fort7")
    parser.formatter_class = argparse.RawTextHelpFormatter
    parser.description = (
        "Repair corrupted fort.7 atom lines where fused integer columns break tokenization.\n\n"
        "Examples:\n"
        "  reaxkit repair_fort7 --file fort.7 --output fort7_fixed\n"
        "  reaxkit repair_fort7 --output fort7_repaired --copy-to-dot"
    )
    parser.add_argument("--file", default="fort.7", help="Input fort.7 file")
    parser.add_argument("--output", default="fort7_fixed", help="Output repaired fort.7 file")
    parser.add_argument(
        "--progress-every",
        type=int,
        default=5000,
        help="Progress interval in lines (stored for metadata compatibility).",
    )
    parser.add_argument("--copy-to-dot", action="store_true", help="Also copy generated output to current directory")
    add_storage_cli_arguments(parser)
    return parser


def run_main(command: str, args: argparse.Namespace) -> int:
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
