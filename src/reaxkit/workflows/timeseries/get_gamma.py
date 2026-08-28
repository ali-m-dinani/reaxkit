"""Dedicated workflow for cell angle gamma."""

from reaxkit.workflows.timeseries.common import build_simulation_request, configure_parser, run_task

COMMAND = "get_gamma"


def build_parser(parser, *, command: str):
    return configure_parser(parser, command=command, description="Get cell angle gamma as a time series.", inputs=("xmolout", "summary"))


def build_request(args):
    return build_simulation_request(args, "gamma")


def run_main(command: str, args) -> int:
    return run_task(COMMAND, "simulation_series", build_request(args), args)

