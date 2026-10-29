"""Dedicated workflow for total molecular-atom time series."""

from reaxkit.workflows.timeseries.common import build_molecular_totals_request, configure_parser, run_task

COMMAND = "get_total_atoms"


def build_parser(parser, *, command: str):
    return configure_parser(
        parser,
        command=command,
        description=(
            "Get total atom count from molecular analysis as a time series.\n"
            "\n"
            "Inspect totals recorded by molecular analysis of an existing simulation.\n"
            "\n"
            "Examples:\n"
            "  1. Plot the series:\n"
            "     reaxkit get-total-atoms --molfra runs/heating/molfra.out --plot single\n"
            "\n"
            "  2. Export sampled values:\n"
            "     reaxkit get-total-atoms --molfra runs/heating/molfra.out --every 5 --export total_atoms.csv"
        ),
        inputs=("molfra",),
    )


def build_request(args):
    return build_molecular_totals_request(args, ("total_atoms",))


def run_main(command: str, args) -> int:
    return run_task(COMMAND, "molecular_totals_series", build_request(args), args)

