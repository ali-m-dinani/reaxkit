"""Dedicated workflow for density time series."""

from reaxkit.workflows.timeseries.common import build_simulation_request, configure_parser, run_task

COMMAND = "get_density"


def build_parser(parser, *, command: str):
    return configure_parser(parser, command=command, description="Get simulation density as a time series.", inputs=("xmolout", "summary"))


def build_request(args):
    return build_simulation_request(args, "density")


def run_main(command: str, args) -> int:
    return run_task(COMMAND, "simulation_series", build_request(args), args)

