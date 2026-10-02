"""Dedicated workflow for cell length a."""

from reaxkit.workflows.timeseries.common import build_simulation_request, configure_parser, run_task

COMMAND = "get_a"


def build_parser(parser, *, command: str):
    return configure_parser(
        parser,
        command=command,
        description=(
            "Get cell length a as a time series.\n"
            "\n"
            "Inspect changes in an existing simulation; no simulation is run.\n"
            "\n"
            "Examples:\n"
            "  1. Plot the series:\n"
            "     reaxkit get-a --xmolout runs/heating/xmolout --plot single\n"
            "\n"
            "  2. Export sampled values:\n"
            "     reaxkit get-a --xmolout runs/heating/xmolout --every 5 --export a.csv"
        ),
        inputs=("xmolout", "summary"),
    )


def build_request(args):
    return build_simulation_request(args, "a")


def run_main(command: str, args) -> int:
    return run_task(COMMAND, "simulation_series", build_request(args), args)

