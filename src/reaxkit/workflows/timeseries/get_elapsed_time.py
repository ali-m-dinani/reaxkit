"""Dedicated workflow for elapsed-time series."""

from reaxkit.workflows.timeseries.common import build_simulation_request, configure_parser, run_task

COMMAND = "get_elapsed_time"


def build_parser(parser, *, command: str):
    return configure_parser(
        parser,
        command=command,
        description=(
            "Get recorded elapsed time as a series.\n"
            "Includes elapsed_time_per_iter: elapsed time divided by iteration.\n"
            "Zero or missing iteration values have no rate (NaN).\n"
            "Plots include elapsed time and elapsed time per iteration.\n"
            "\n"
            "Inspect changes in an existing simulation; no simulation is run.\n"
            "\n"
            "Examples:\n"
            "  1. Plot both series separately:\n"
            "     reaxkit get-elapsed-time --engine reaxff --summary summary.txt --plot separate --save elapsed_time_plots"
        ),
        inputs=("xmolout", "summary"),
    )


def build_request(args):
    return build_simulation_request(args, "elapsed_time")


def run_main(command: str, args) -> int:
    return run_task(COMMAND, "simulation_series", build_request(args), args)

