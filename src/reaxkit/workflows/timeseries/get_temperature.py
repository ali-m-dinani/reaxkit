"""Dedicated workflow for temperature time series."""

from reaxkit.workflows.timeseries.common import build_simulation_request, configure_parser, run_task

COMMAND = "get_temperature"


def build_parser(parser, *, command: str):
    return configure_parser(
        parser,
        command=command,
        description=(
            "Get simulation temperature as a time series.\n"
            "\n"
            "Inspect changes in an existing simulation; no simulation is run.\n"
            "\n"
            "Examples:\n"
            "  1. Plot the series:\n"
            "     reaxkit get-temperature --xmolout runs/heating/xmolout --plot single\n"
            "\n"
            "  2. Export sampled values:\n"
            "     reaxkit get-temperature --xmolout runs/heating/xmolout --every 5 --export temperature.csv"
        ),
        inputs=("xmolout", "summary"),
    )


def build_request(args):
    return build_simulation_request(args, "temperature")


def run_main(command: str, args) -> int:
    return run_task(COMMAND, "simulation_series", build_request(args), args)

