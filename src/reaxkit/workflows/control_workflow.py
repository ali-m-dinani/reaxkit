"""workflow for control file: query control keys like 'nmdit' """

from __future__ import annotations
import argparse

from reaxkit.io.control_handler import ControlHandler
from reaxkit.analysis.control_analyzer import get_control


def _format_value(value):
    """
    Format numeric values with commas (e.g., 150000 -> 150,000).
    Works for int or float-like inputs.
    """
    # Try integer first
    try:
        iv = int(value)
        return f"{iv:,}"
    except (ValueError, TypeError):
        pass

    # Try float next (only apply formatting if large enough)
    try:
        fv = float(value)
        # Only add commas to the integer part for large floats
        if abs(fv) >= 1000:
            int_part, dot, frac = f"{fv}".partition(".")
            int_part = f"{int(int_part):,}"
            return int_part + (("." + frac) if frac else "")
        return value
    except (ValueError, TypeError):
        return value


def control_get_task(args: argparse.Namespace) -> int:
    """CLI task: get a single control key value (e.g., nmdit)."""
    handler = ControlHandler(args.file)
    value = get_control(handler, args.key, section=args.section, default=None)

    if value is None:
        print(f"❌ Key '{args.key}' not found in control file '{args.file}'.")
        return 1

    formatted = _format_value(value)
    print(f'{args.key} = {formatted}')
    return 0


def register_tasks(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "get",
        help="Get the value of a control key (e.g. nmdit)",
        description=(
            "Examples:\n"
            "  reaxkit control get nmdit\n"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument("key", help="Control key to look up, e.g. 'nmdit'.")
    p.add_argument(
        "--file", default="control",
        help="Path to control file (default: 'control').",
    )
    p.add_argument(
        "--section", default=None,
        help="Optional control section (general, md, mm, ff, outdated).",
    )
    p.set_defaults(_run=control_get_task)
