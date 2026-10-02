"""Dedicated workflow for elapsed-time series."""

from reaxkit.workflows.timeseries.common import build_simulation_request, configure_parser, run_task

COMMAND = "get_elapsed_time"


def build_parser(parser, *, command: str):
    return configure_parser(
        parser,
        command=command,
        description=(
            "Get recorded elapsed time as a series.\n"
            "\n"
            "Inspect changes in an existing simulation; no simulation is run.\n"
            "\n"
            "Examples:\n"
            "  1. Plot the series:\n"
            "     reaxkit get-elapsed-time --xmolout runs/heating/xmolout --plot single\n"
            "\n"
            "  2. Export sampled values:\n"
            "     reaxkit get-elapsed-time --xmolout runs/heating/xmolout --every 5 --export elapsed_time.csv"
        ),
        inputs=("xmolout", "summary"),
    )


def build_request(args):
    return build_simulation_request(args, "elapsed_time")


def run_main(command: str, args) -> int:
    return run_task(COMMAND, "simulation_series", build_request(args), args)

