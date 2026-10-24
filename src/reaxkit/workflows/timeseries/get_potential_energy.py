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
            "Per-atom normalization uses --num-atoms N or frame atom counts from xmolout.\n"
            "Use --relative to also output E(frame) - E(reference_frame).\n"
            "The reference defaults to source frame 0, before sampling; change it with --reference-frame.\n"
            "With --per-atom, normalize each frame before subtracting the reference.\n"
            "CSV exports retain total energy; --relative with --num-atoms or --per-atom includes total, relative total, per-atom, and relative per-atom columns.\n"
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
    parser.add_argument(
        "--num-atoms",
        type=int,
        default=None,
        help="Positive atom count used to normalize every frame; implies --per-atom and overrides xmolout atom counts. Example: --num-atoms 100 --relative.",
    )
    parser.add_argument(
        "--relative",
        action="store_true",
        help="Also output E(frame) - E(reference_frame), after any per-atom normalization. Example: --relative.",
    )
    parser.add_argument(
        "--reference-frame",
        type=int,
        default=0,
        help="Zero-based source frame for --relative, before --frames/--every sampling (default: 0). Example: --relative --reference-frame 10.",
    )
    return parser


def build_request(args):
    return build_simulation_request(args, "potential_energy")


def run_main(command: str, args) -> int:
    return run_task(COMMAND, "simulation_series", build_request(args), args)

