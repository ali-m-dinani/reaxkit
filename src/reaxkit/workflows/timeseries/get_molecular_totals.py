"""Dedicated workflow for molecular-total time series."""

from reaxkit.workflows.timeseries.common import build_molecular_totals_request, configure_parser, run_task

COMMAND = "get_molecular_totals"
QUANTITIES = ("total_molecules", "total_atoms", "total_molecular_mass")


def build_parser(parser, *, command: str):
    configure_parser(parser, command=command, description="Get selected molecular totals as time series.", inputs=("molfra",))
    parser.add_argument("--quantities", nargs="+", choices=list(QUANTITIES), default=QUANTITIES)
    return parser


def build_request(args):
    return build_molecular_totals_request(args)


def run_main(command: str, args) -> int:
    return run_task(COMMAND, "molecular_totals_series", build_request(args), args)

