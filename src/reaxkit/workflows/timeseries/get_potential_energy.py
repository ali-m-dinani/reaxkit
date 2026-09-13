"""Dedicated workflow for potential-energy time series."""

from reaxkit.workflows.timeseries.common import build_simulation_request, configure_parser, run_task

COMMAND = "get_potential_energy"


def build_parser(parser, *, command: str):
    configure_parser(
        parser,
        command=command,
        description=(
            "Get potential energy as a time series.\n\n"
            "Examples:\n"
            "  reaxkit get_potential_energy --summary summary.txt --plot single\n"
            "  reaxkit get_potential_energy --summary summary.txt --per-atom "
            "--plot single --save pot_en_per_atom.png"
        ),
        inputs=("xmolout", "summary"),
    )
    parser.add_argument(
        "--per-atom",
        action="store_true",
        help=(
            "Divide potential energy by the number of atoms in each frame. "
            "The atom counts are read from xmolout and may vary across frames."
        ),
    )
    return parser


def build_request(args):
    return build_simulation_request(args, "potential_energy")


def run_main(command: str, args) -> int:
    return run_task(COMMAND, "simulation_series", build_request(args), args)

