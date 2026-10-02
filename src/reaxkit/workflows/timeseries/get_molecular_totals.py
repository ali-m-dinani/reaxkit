"""Dedicated workflow for molecular-total time series."""

from reaxkit.workflows.timeseries.common import build_molecular_totals_request, configure_parser, run_task

COMMAND = "get_molecular_totals"
QUANTITIES = ("total_molecules", "total_atoms", "total_molecular_mass")


def build_parser(parser, *, command: str):
    configure_parser(parser, command=command, description=(
            "Get selected molecular totals as time series.\n"
            "\n"
            "Check molecular totals in existing molecular-analysis output.\n"
            "\n"
            "Examples:\n"
            "  1. Selected totals:\n"
            "     reaxkit get-molecular-totals --molfra runs/heating/molfra.out --quantities total_molecules total_atoms --plot subplot\n"
            "\n"
            "  2. All totals:\n"
            "     reaxkit get-molecular-totals --molfra runs/heating/molfra.out --export molecular_totals.csv"
        ), inputs=("molfra",))
    parser.add_argument("--quantities", nargs="+", choices=list(QUANTITIES), default=QUANTITIES, help="Molecular totals to include. Example: --quantities total_molecules total_atoms, which excludes total molecular mass.")
    return parser


def build_request(args):
    return build_molecular_totals_request(args)


def run_main(command: str, args) -> int:
    return run_task(COMMAND, "molecular_totals_series", build_request(args), args)

