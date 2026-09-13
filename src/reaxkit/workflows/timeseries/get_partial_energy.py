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
            "  1. Save every partial-energy component as its own plot:\n"
            "     reaxkit get-partial-energy --fort73 .\\energylog "
            "--save partial_energies --plot separate\n"
            "     Reads all components from energylog and saves one figure per "
            "component in the partial_energies directory."
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
