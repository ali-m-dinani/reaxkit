"""Dedicated workflow for partial-energy time series."""

from reaxkit.workflows.timeseries.common import build_partial_energy_request, configure_parser, run_task

COMMAND = "get_partial_energy"


def build_parser(parser, *, command: str):
    configure_parser(
        parser,
        command=command,
        description=(
            "Extract partial-energy components from a fort.73 time series.\n"
            "Use this command to inspect how individual energy contributions change "
            "during a simulation.\n"
            "Omit --components to include every available component.\n\n"
            "Examples:\n"
            "  1. Plot bond and atomic energy components against physical time:\n"
            "     reaxkit get-partial-energy --fort73 fort.73 --components Ebond Eatom "
            "--xaxis time --plot single\n"
            "     Reads only Ebond and Eatom from fort.73 and plots both series using "
            "the physical-time x-axis."
        ),
        inputs=("fort73",),
    )
    parser.add_argument(
        "--components",
        nargs="*",
        default=None,
        help=(
            "Partial-energy components to include. Example: --components Ebond Eatom, "
            "which includes only the bond and atomic energy series; omit this option "
            "to include all available components."
        ),
    )
    return parser


def build_request(args):
    return build_partial_energy_request(args)


def run_main(command: str, args) -> int:
    return run_task(COMMAND, "partial_energy_series", build_request(args), args)
