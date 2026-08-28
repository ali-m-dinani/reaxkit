"""Dedicated workflow for total molecular-mass time series."""

from reaxkit.workflows.timeseries.common import build_molecular_totals_request, configure_parser, run_task

COMMAND = "get_total_molecular_mass"


def build_parser(parser, *, command: str):
    return configure_parser(parser, command=command, description="Get total molecular mass as a time series.", inputs=("molfra",))


def build_request(args):
    return build_molecular_totals_request(args, ("total_molecular_mass",))


def run_main(command: str, args) -> int:
    return run_task(COMMAND, "molecular_totals_series", build_request(args), args)

