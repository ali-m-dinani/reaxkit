"""Dedicated workflow for potential-energy time series."""

from reaxkit.workflows.timeseries.common import build_simulation_request, configure_parser, run_task

COMMAND = "get_potential_energy"


def build_parser(parser, *, command: str):
    configure_parser(
        parser,
        command=command,
        description=(
            "Get potential energy as a time series.\n"
            "\n"
            "Compare total or per-atom potential energy in existing simulation output.\n"
            "Per-atom normalization requires frame atom counts from xmolout.\n"
            "\n"
            "Examples:\n"
            "  1. Total energy:\n"
            "     reaxkit get-potential-energy --summary runs/heating/summary.txt --plot single\n"
            "\n"
            "  2. Energy per atom:\n"
            "     reaxkit get-potential-energy --summary runs/heating/summary.txt --xmolout runs/heating/xmolout --per-atom --plot single --save energy_per_atom.png"
        ),
        inputs=("xmolout", "summary"),
    )
    parser.add_argument(
        "--per-atom",
        action="store_true",
        help="Divide potential energy by each frame's atom count from xmolout. Example: --per-atom, which reports energy per atom instead of total energy.",
    )
    return parser


def build_request(args):
    return build_simulation_request(args, "potential_energy")


def run_main(command: str, args) -> int:
    return run_task(COMMAND, "simulation_series", build_request(args), args)

