"""Dedicated workflow for potential-energy time series."""

from reaxkit.workflows.timeseries.common import build_simulation_request, configure_parser, run_task

COMMAND = "get_potential_energy"


def build_parser(parser, *, command: str):
    return configure_parser(
        parser,
        command=command,
        description="Get potential energy as a time series.\n\nExample:\n  reaxkit get_potential_energy --summary summary.txt --plot single",
        inputs=("xmolout", "summary"),
    )


def build_request(args):
    return build_simulation_request(args, "potential_energy")


def run_main(command: str, args) -> int:
    return run_task(COMMAND, "simulation_series", build_request(args), args)

