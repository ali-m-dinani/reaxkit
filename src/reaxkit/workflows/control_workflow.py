"""workflow for control file"""

from __future__ import annotations
import argparse

from reaxkit.io.control_handler import ControlHandler
from reaxkit.analysis.control_analyzer import get_control
from reaxkit.io.control_generator import write_control_template

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

def control_make_task(args: argparse.Namespace) -> int:
    """
    CLI task: generate a default control file with all sections and default values.
    """
    output = write_control_template(args.output)
    print(f"[Done] control file written to {output}")
    return 0

def register_tasks(subparsers: argparse._SubParsersAction) -> None:
    # --- get ---
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

    # --- make ---
    m = subparsers.add_parser(
        "make",
        help="Generate a default control file with all sections and default values",
        description=(
            "Examples:\n"
            "  reaxkit control make\n"
            "  reaxkit control make --output control\n"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    m.add_argument(
        "--output",
        default="reaxkit_generated_inputs/control",
        help="Output path for the generated control file (default: 'control').",
    )
    m.set_defaults(_run=control_make_task)