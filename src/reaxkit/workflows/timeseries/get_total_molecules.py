"""Dedicated workflow for total-molecule time series."""

from reaxkit.workflows.timeseries.common import build_molecular_totals_request, configure_parser, run_task

COMMAND = "get_total_molecules"


def build_parser(parser, *, command: str):
    return configure_parser(
        parser,
        command=command,
        description=(
            "Get total molecule count as a time series.\n"
            "\n"
            "Inspect totals recorded by molecular analysis of an existing simulation.\n"
            "\n"
            "Examples:\n"
            "  1. Plot the series:\n"
            "     reaxkit get-total-molecules --molfra runs/heating/molfra.out --plot single\n"
            "\n"
            "  2. Export sampled values:\n"
            "     reaxkit get-total-molecules --molfra runs/heating/molfra.out --every 5 --export total_molecules.csv"
        ),
        inputs=("molfra",),
    )


def build_request(args):
    return build_molecular_totals_request(args, ("total_molecules",))


def run_main(command: str, args) -> int:
    return run_task(COMMAND, "molecular_totals_series", build_request(args), args)

